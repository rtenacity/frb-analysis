import argparse
import json
import os
import shutil
from os.path import join

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

BASE_FEATURES = ['bonsai_dm', 'dm_exc_ne2001', 'dm_exc_ymw16', 'bc_width',
                 'high_freq', 'low_freq', 'peak_freq']
ERROR_FEATURES = ['dm_fitb', 'fluence', 'flux', 'sp_idx', 'sp_run']
INNER_CV_SEED = 42  # fixed across trials, as in the original objective


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class SupervisedVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.3,
                 activation=nn.LeakyReLU(0.1)):
        super(SupervisedVAE, self).__init__()
        self.activation = activation

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            self.activation, nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            self.activation, nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            self.activation, nn.Dropout(dropout_rate),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            self.activation, nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            self.activation, nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            self.activation, nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, self.classifier(mu)


def get_activation_function(name):
    return {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(0.1),
        'ELU': nn.ELU(),
        'SELU': nn.SELU(),
        'GELU': nn.GELU(),
    }[name]


def loss_function(recon_x, x, mu, logvar, class_logits, labels, beta, gamma,
                  class_weight, classification_multiplier, device):
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    pos_weight = torch.tensor([class_weight], dtype=torch.float32, device=device)
    class_loss = classification_multiplier * nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
        class_logits, labels.unsqueeze(1).float())
    return recon_loss + beta * kl_loss + gamma * class_loss


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(path):
    """Same cleaning as the original script. Returns (X_df, y, names)."""
    df = pd.read_csv(path)
    df['repeater'] = df['repeater_name'].apply(lambda n: 1 if n != "-9999" else 0)

    df["mjd_400"] = pd.to_numeric(df["mjd_400"], errors="coerce")
    df["mjd_400_r"] = df["mjd_400"].round(6)
    mask = df["repeater"] == 0
    df = pd.concat([
        df[mask].drop_duplicates(subset=["tns_name", "mjd_400_r"], keep="first"),
        df[~mask],
    ]).sort_index()

    for f in BASE_FEATURES + ERROR_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[f]):
            df[f] = pd.to_numeric(df[f])

    for f in ERROR_FEATURES:
        df[f"{f}_lower"] = df[f] - df[f"{f}_err"]
        df[f"{f}_upper"] = df[f] + df[f"{f}_err"]

    features = ([f"{f}_lower" for f in ERROR_FEATURES]
                + [f"{f}_upper" for f in ERROR_FEATURES]
                + BASE_FEATURES)

    X = df[features].dropna()
    n_dropped = len(df) - len(X)
    if n_dropped:
        print(f"[data] dropped {n_dropped} rows with NaN features")
    sub = df.loc[X.index]
    y = sub['repeater'].to_numpy()
    names = sub['tns_name'].astype(str).to_numpy()
    return X, y, names


def make_loader(Xs, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(Xs, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    # BatchNorm cannot train on a batch of size 1; drop a trailing singleton batch
    drop_last = shuffle and (len(ds) % batch_size == 1)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


# --------------------------------------------------------------------------- #
# Training pieces
# --------------------------------------------------------------------------- #
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, optimizer, loader, hp, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        recon, mu, logvar, logits = model(x)
        loss = loss_function(recon, x, mu, logvar, logits, y, hp['beta'], hp['gamma'],
                             hp['class_weight'], hp['classification_multiplier'], device)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def validation_loss(model, loader, hp, device):
    """Per-sample loss on the inner-validation fold; drives ReduceLROnPlateau
    (same as the original procedure)."""
    model.eval()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        recon, mu, logvar, logits = model(x)
        total += loss_function(recon, x, mu, logvar, logits, y, hp['beta'], hp['gamma'],
                               hp['class_weight'], hp['classification_multiplier'],
                               device).item()
    return total / len(loader.dataset)


@torch.no_grad()
def predict_probs(model, Xs, device):
    model.eval()
    x = torch.tensor(Xs, dtype=torch.float32, device=device)
    return torch.sigmoid(model(x)[-1]).squeeze(1).cpu().numpy()


def build_model(input_dim, hp, device):
    return SupervisedVAE(
        input_dim,
        hidden_dim=hp['hidden_dim'],
        latent_dim=hp['latent_dim'],
        dropout_rate=hp['dropout_rate'],
        activation=get_activation_function(hp['activation']),
    ).to(device)


def fit_fold(hp, X_tr, y_tr, X_va, y_va, args, device):
    """Train on one inner fold. Scaler is fit on the inner-train part only."""
    scaler = StandardScaler().fit(X_tr)
    Xtr_s, Xva_s = scaler.transform(X_tr), scaler.transform(X_va)
    train_loader = make_loader(Xtr_s, y_tr, args.batch_size, shuffle=True)
    val_loader = make_loader(Xva_s, y_va, args.batch_size, shuffle=False)

    model = build_model(Xtr_s.shape[1], hp, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=hp['scheduler_patience'])
    # Fixed number of epochs, no early stopping (the original never fired).
    for _ in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, hp, device)
        scheduler.step(validation_loss(model, val_loader, hp, device))
    return model, scaler, predict_probs(model, Xva_s, device)


# --------------------------------------------------------------------------- #
# Optuna (one search per outer fold, on the outer-train part only)
# --------------------------------------------------------------------------- #
def make_objective(X_dev, y_dev, outer_idx, args, device, ckpt_root):
    inner = StratifiedKFold(n_splits=args.inner_folds, shuffle=True,
                            random_state=INNER_CV_SEED)
    splits = list(inner.split(X_dev, y_dev))

    def objective(trial):
        hp = dict(
            hidden_dim=trial.suggest_int('hidden_dim', 128, 2048),
            latent_dim=trial.suggest_int('latent_dim', 5, 40),
            beta=trial.suggest_float('beta', 0.1, 2.0),
            gamma=trial.suggest_float('gamma', 0.1, 2.0),
            dropout_rate=trial.suggest_float('dropout_rate', 0.1, 0.5),
            lr=trial.suggest_float('lr', 1e-4, 1e-2),
            scheduler_patience=trial.suggest_int('scheduler_patience', 2, 7),
            class_weight=trial.suggest_float('class_weight', 0.05, 1),
            activation=trial.suggest_categorical(
                'activation', ['ReLU', 'LeakyReLU', 'ELU', 'SELU', 'GELU']),
            classification_multiplier=trial.suggest_float(
                'classification_multiplier', 5000, 15000),
        )
        set_seed(args.seed + 100000 * outer_idx + trial.number)

        trial_dir = join(ckpt_root, f'trial_{trial.number}')
        os.makedirs(trial_dir, exist_ok=True)

        fold_f1 = []
        for k, (tr, va) in enumerate(splits):
            model, scaler, probs = fit_fold(hp, X_dev[tr], y_dev[tr],
                                            X_dev[va], y_dev[va], args, device)
            f1 = f1_score(y_dev[va], (probs > 0.5).astype(int), zero_division=0)
            fold_f1.append(float(f1))
            torch.save({
                'state_dict': {n: v.detach().cpu() for n, v in model.state_dict().items()},
                'scaler_mean': torch.tensor(scaler.mean_),
                'scaler_scale': torch.tensor(scaler.scale_),
                'hp': hp,
            }, join(trial_dir, f'model_fold_{k + 1}.pth'))
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        trial.set_user_attr('fold_f1', fold_f1)
        mean_f1 = float(np.mean(fold_f1))
        print(f"[outer {outer_idx}] trial {trial.number}: inner F1 = {mean_f1:.4f} "
              f"(folds: {', '.join(f'{v:.3f}' for v in fold_f1)})")
        return mean_f1

    return objective


def make_cleanup_callback(ckpt_root):
    """Keep only the checkpoints of the current best trial (saves tens of GB)."""
    def cb(study, trial):
        try:
            keep = f'trial_{study.best_trial.number}'
        except ValueError:  # no completed trial yet
            keep = None
        for d in os.listdir(ckpt_root):
            if d.startswith('trial_') and d != keep:
                shutil.rmtree(join(ckpt_root, d), ignore_errors=True)
    return cb


def make_log_callback(odir, outer_idx):
    path = join(odir, 'trial_log.jsonl')

    def cb(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        with open(path, 'a') as f:
            f.write(json.dumps({'trial': trial.number, 'inner_f1': trial.value,
                                'inner_f1_per_fold': trial.user_attrs.get('fold_f1')}) + '\n')
        best = study.best_trial
        print(f"[outer {outer_idx}] [best so far] trial {best.number}: "
              f"inner F1 = {best.value:.4f}")
    return cb


# --------------------------------------------------------------------------- #
# Evaluation of one outer fold (run once, after its search is finished)
# --------------------------------------------------------------------------- #
def evaluate_outer(study, ckpt_root, odir, outer_idx, X_test, y_test, names_test,
                   row_ids_test, args, device):
    best = study.best_trial
    trial_dir = join(ckpt_root, f'trial_{best.number}')
    if not os.path.isdir(trial_dir):
        raise FileNotFoundError(f"Checkpoints for best trial {best.number} not found "
                                f"in {trial_dir}")

    probs = []
    for k in range(args.inner_folds):
        ck = torch.load(join(trial_dir, f'model_fold_{k + 1}.pth'),
                        map_location='cpu', weights_only=True)
        model = build_model(X_test.shape[1], ck['hp'], device)
        model.load_state_dict(ck['state_dict'])
        # each model is scored with ITS OWN training-fold scaler
        Xs = (X_test - ck['scaler_mean'].numpy()) / ck['scaler_scale'].numpy()
        probs.append(predict_probs(model, Xs, device))
    probs = np.stack(probs)                              # (inner_folds, n_test)
    preds = (probs > 0.5).astype(int)
    ens_prob = probs.mean(axis=0)
    ens_pred = (ens_prob > 0.5).astype(int)

    f1_each = [float(f1_score(y_test, p, zero_division=0)) for p in preds]
    inner_f1 = float(best.value)
    results = {
        'outer_fold': outer_idx,
        'best_trial': best.number,
        'best_params': best.params,
        'n_completed_trials': sum(t.state == optuna.trial.TrialState.COMPLETE
                                  for t in study.trials),
        'inner_cv_f1_mean': inner_f1,
        'inner_cv_f1_per_fold': best.user_attrs.get('fold_f1'),
        'n_test': int(len(y_test)),
        'n_test_repeaters': int(y_test.sum()),
        'outer_f1_per_model': f1_each,
        'outer_f1_mean_of_models': float(np.mean(f1_each)),
        'outer_f1_ensemble': float(f1_score(y_test, ens_pred, zero_division=0)),
        'outer_f2_ensemble': float(fbeta_score(y_test, ens_pred, beta=2, zero_division=0)),
        'outer_precision_ensemble': float(precision_score(y_test, ens_pred, zero_division=0)),
        'outer_recall_ensemble': float(recall_score(y_test, ens_pred, zero_division=0)),
    }
    results['optimism_vs_mean_of_models'] = inner_f1 - results['outer_f1_mean_of_models']
    results['optimism_vs_ensemble'] = inner_f1 - results['outer_f1_ensemble']
    with open(join(odir, 'heldout_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    pred_df = pd.DataFrame({'row_id': row_ids_test, 'tns_name': names_test,
                            'label': y_test, 'outer_fold': outer_idx})
    for k in range(probs.shape[0]):
        pred_df[f'prob_model{k + 1}'] = probs[k]
    pred_df['prob_ensemble'] = ens_prob
    pred_df['pred_ensemble'] = ens_pred
    pred_df.to_csv(join(odir, 'heldout_predictions.csv'), index=False)

    print(f"\n=== Outer fold {outer_idx}: best trial {best.number} ===")
    print(f"  inner-CV F1 (selected)           : {inner_f1:.4f}")
    print(f"  outer F1, mean of 3 models       : {results['outer_f1_mean_of_models']:.4f}")
    print(f"  outer F1, ensemble               : {results['outer_f1_ensemble']:.4f}")
    print(f"  optimism (inner - outer, ens)    : {results['optimism_vs_ensemble']:+.4f}\n")
    return results


# --------------------------------------------------------------------------- #
# Aggregation over the outer folds
# --------------------------------------------------------------------------- #
def _counts(y, p):
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    return tp, fp, fn, tn


def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if 4 * prec + rec else 0.0
    return prec, rec, f1, f2


def aggregate(out_dir, args):
    rows, preds = [], []
    for i in range(args.outer_folds):
        rp = join(out_dir, f'outer_{i}', 'heldout_results.json')
        pp = join(out_dir, f'outer_{i}', 'heldout_predictions.csv')
        if not (os.path.exists(rp) and os.path.exists(pp)):
            print(f"[aggregate] outer fold {i} not finished; skipping aggregation")
            return
        rows.append(json.load(open(rp)))
        preds.append(pd.read_csv(pp))
    P = pd.concat(preds, ignore_index=True)
    assert P['row_id'].is_unique, "a burst appears in more than one outer test fold"
    y = P['label'].to_numpy()
    p = P['pred_ensemble'].to_numpy()

    def ms(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

    # ---- per-outer-fold summary ---- #
    print("\n" + "=" * 78)
    print(f"NESTED CV ({args.outer_folds} outer x full Optuna x {args.inner_folds} inner)"
          )
    print("=" * 78)
    print(f"{'outer':>5} {'n':>5} {'rep':>4} {'trial':>6} {'innerF1':>8} "
          f"{'outF1(ens)':>10} {'outF1(mean)':>11} {'prec':>6} {'rec':>6} {'optim':>7}")
    for r in rows:
        print(f"{r['outer_fold']:>5} {r['n_test']:>5} {r['n_test_repeaters']:>4} "
              f"{r['best_trial']:>6} {r['inner_cv_f1_mean']:>8.4f} "
              f"{r['outer_f1_ensemble']:>10.4f} {r['outer_f1_mean_of_models']:>11.4f} "
              f"{r['outer_precision_ensemble']:>6.3f} {r['outer_recall_ensemble']:>6.3f} "
              f"{r['optimism_vs_ensemble']:>+7.4f}")
    summary = {k: ms(k) for k in ['inner_cv_f1_mean', 'outer_f1_ensemble',
                                  'outer_f1_mean_of_models', 'outer_f2_ensemble',
                                  'outer_precision_ensemble', 'outer_recall_ensemble',
                                  'optimism_vs_ensemble', 'optimism_vs_mean_of_models']}
    print("\nMean +/- std over outer folds (std is only a rough guide: folds overlap in training):")
    for k, (m, s) in summary.items():
        print(f"  {k:<30}: {m:.4f} +/- {s:.4f}")

    # ---- pooled out-of-fold metrics ---- #
    tp, fp, fn, tn = _counts(y, p)
    prec, rec, f1, f2 = _prf(tp, fp, fn)
    acc = (tp + tn) / len(y)

    rng = np.random.default_rng(args.seed)
    n = len(y)
    boot = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'f2': []}
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, n)
        btp, bfp, bfn, btn = _counts(y[idx], p[idx])
        bp, br, bf1, bf2 = _prf(btp, bfp, bfn)
        boot['accuracy'].append((btp + btn) / n)
        boot['precision'].append(bp)
        boot['recall'].append(br)
        boot['f1'].append(bf1)
        boot['f2'].append(bf2)
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
          for k, v in boot.items()}
    pooled = {'n': int(n), 'n_repeaters': int(y.sum()),
              'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'f2': f2,
              'confusion_matrix [[TN, FP], [FN, TP]]': [[tn, fp], [fn, tp]],
              'ci95': ci}

    print("\nPOOLED out-of-fold metrics (each burst predicted by the ensemble of 3 models that")
    print("never saw it, from a search that never saw it; repeater = positive class):")
    print(f"  N = {n} ({int(y.sum())} repeaters)")
    print(f"  confusion [[TN, FP], [FN, TP]] = [[{tn}, {fp}], [{fn}, {tp}]]")
    for name, v in [('accuracy', acc), ('precision', prec), ('recall', rec),
                    ('F1', f1), ('F2', f2)]:
        lo, hi = ci[name.lower()]
        print(f"  {name:<10}: {v:.4f}   95% bootstrap CI [{lo:.3f}, {hi:.3f}]")
    inner_mean = summary['inner_cv_f1_mean'][0]
    print(f"\nSelection optimism: mean inner-CV F1 of selected trials = {inner_mean:.4f} "
          f"vs pooled outer F1 = {f1:.4f}  ->  {inner_mean - f1:+.4f}")

    print("\nSelected hyperparameters per outer fold:")
    for r in rows:
        bp = r['best_params']
        print(f"  outer {r['outer_fold']} (trial {r['best_trial']}): "
              + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in bp.items()))

    out = {'settings': {'outer_folds': args.outer_folds, 'inner_folds': args.inner_folds,
                        'n_trials': args.n_trials, 'seed': args.seed,
                        'epochs': args.epochs},
           'per_outer_fold': rows,
           'mean_std_over_outer_folds': {k: {'mean': m, 'std': s}
                                         for k, (m, s) in summary.items()},
           'pooled': pooled}
    with open(join(out_dir, 'nested_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    P.to_csv(join(out_dir, 'nested_predictions.csv'), index=False)
    print(f"\nSaved {join(out_dir, 'nested_results.json')} and nested_predictions.csv")


# --------------------------------------------------------------------------- #
def run_outer(i, dev_pos, test_pos, X, y, names, row_ids, args, device, out_dir):
    odir = join(out_dir, f'outer_{i}')
    ckpt_root = join(odir, 'checkpoints')
    os.makedirs(ckpt_root, exist_ok=True)

    X_dev, y_dev = X[dev_pos], y[dev_pos]
    X_test, y_test = X[test_pos], y[test_pos]
    print(f"[outer {i}] dev: {len(y_dev)} ({int(y_dev.sum())} repeaters) | "
          f"test: {len(y_test)} ({int(y_test.sum())} repeaters)")

    storage = f"sqlite:///{join(odir, 'study.db')}"
    study = optuna.create_study(
        direction='maximize',
        study_name=f'svae_nested_seed{args.seed}_outer{i}of{args.outer_folds}',
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=args.seed + i),
        load_if_exists=True,
    )
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, args.n_trials - done)
    print(f"[outer {i}] [optuna] {done} completed trials found, running {remaining} more")
    if remaining:
        callbacks = [make_log_callback(odir, i)]
        if not args.keep_all_checkpoints:
            callbacks.append(make_cleanup_callback(ckpt_root))
        study.optimize(make_objective(X_dev, y_dev, i, args, device, ckpt_root),
                       n_trials=remaining, callbacks=callbacks)

    return evaluate_outer(study, ckpt_root, odir, i, X_test, y_test, names[test_pos],
                          row_ids[test_pos], args, device)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='chimefrbcat1.csv')
    p.add_argument('--out_dir',
                   default='/scratch/gpfs/MLISANTI/ra0438/old_frb_paper/nested_cv_seed42',
                   help='per-outer-fold checkpoints, study.db, results all go here')
    p.add_argument('--n_trials', type=int, default=350,
                   help='TOTAL completed trials wanted PER OUTER FOLD; on resume only '
                        'the rest are run (0 = skip search, just evaluate)')
    p.add_argument('--outer_folds', type=int, default=5)
    p.add_argument('--inner_folds', type=int, default=3)
    p.add_argument('--outer_fold', type=int, default=None,
                   help='run only this outer fold (0-indexed), e.g. for a SLURM array')
    p.add_argument('--aggregate_only', action='store_true',
                   help='skip all training; aggregate finished outer folds')
    p.add_argument('--seed', type=int, default=42, help='outer split / sampler / init seed')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_boot', type=int, default=2000)
    p.add_argument('--keep_all_checkpoints', action='store_true',
                   help="keep every trial's checkpoints (default: only the best so far)")
    args = p.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.aggregate_only:
        aggregate(out_dir, args)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"[device] {torch.cuda.get_device_name(0)}")
    set_seed(args.seed)

    X_df, y, names = load_data(args.data)
    X = X_df.to_numpy(dtype=np.float64)
    row_ids = X_df.index.to_numpy()

    outer = StratifiedKFold(n_splits=args.outer_folds, shuffle=True,
                            random_state=args.seed)
    outer_splits = list(outer.split(X, y))
    with open(join(out_dir, 'outer_splits.json'), 'w') as f:
        json.dump({'seed': args.seed,
                   'folds':[{'outer_fold': i,
                              'train_row_ids': row_ids[tr].tolist(),
                              'test_row_ids': row_ids[te].tolist()}
                             for i, (tr, te) in enumerate(outer_splits)]}, f)

    todo = range(args.outer_folds) if args.outer_fold is None else [args.outer_fold]
    for i in todo:
        dev_pos, test_pos = outer_splits[i]
        run_outer(i, np.sort(dev_pos), np.sort(test_pos), X, y, names, row_ids,
                  args, device, out_dir)

    aggregate(out_dir, args)


if __name__ == '__main__':
    main()