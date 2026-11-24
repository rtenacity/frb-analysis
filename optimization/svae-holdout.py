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
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn


class SupervisedVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.3, activation=nn.LeakyReLU(0.1)):
        super(SupervisedVAE, self).__init__()

        self.activation = activation

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),  # Additional dense layer
            nn.BatchNorm1d(hidden_dim),
            self.activation,
            nn.Dropout(dropout_rate)
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),  # Additional dense layer
            nn.BatchNorm1d(hidden_dim),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )

        # Classification head for binary classification - tune hyperparameters
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # Added extra linear layer
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1),
        )


    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        class_prob = self.classifier(mu)
        return recon_x, mu, logvar, class_prob



def loss_function(recon_x, x, mu, logvar, class_prob, labels, beta, gamma, class_weight, classification_multiplier, device):
    reconstruction_loss_fn = nn.MSELoss(reduction='sum')
    pos_weight = torch.tensor([class_weight], dtype=torch.float32, device=device)
    classification_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    recon_loss = reconstruction_loss_fn(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    class_loss = classification_multiplier * classification_loss_fn(class_prob, labels.unsqueeze(1).float())
    total_loss = recon_loss + beta * kl_loss + gamma * class_loss
    return total_loss, recon_loss, kl_loss, class_loss

BASE_FEATURES = ['bonsai_dm', 'dm_exc_ne2001', 'dm_exc_ymw16', 'bc_width',
                 'high_freq', 'low_freq', 'peak_freq']
ERROR_FEATURES = ['dm_fitb', 'fluence', 'flux', 'sp_idx', 'sp_run']
INNER_CV_SEED = 42  # fixed across trials, as in the original objective


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
    # labels/names are taken from the SAME rows as X (index-aligned)
    y = df.loc[X.index, 'repeater'].to_numpy()
    names = df.loc[X.index, 'tns_name'].to_numpy()
    return X, y, names


# --------------------------------------------------------------------------- #
# Model pieces (identical maths to the original)
# --------------------------------------------------------------------------- #
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


def make_loader(Xs, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(Xs, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    # BatchNorm cannot train on a batch of size 1; drop a trailing singleton batch
    drop_last = shuffle and (len(ds) % batch_size == 1)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


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
# Optuna
# --------------------------------------------------------------------------- #
def make_objective(X_dev, y_dev, X_test, y_test, args, device, ckpt_root):
    skf = StratifiedKFold(n_splits=args.inner_folds, shuffle=True,
                          random_state=INNER_CV_SEED)
    splits = list(skf.split(X_dev, y_dev))

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
        set_seed(args.seed + trial.number)

        trial_dir = join(ckpt_root, f'trial_{trial.number}')
        os.makedirs(trial_dir, exist_ok=True)

        fold_f1 = []
        test_probs = []
        for k, (tr, va) in enumerate(splits):
            model, scaler, probs = fit_fold(hp, X_dev[tr], y_dev[tr],
                                            X_dev[va], y_dev[va], args, device)
            f1 = f1_score(y_dev[va], (probs > 0.5).astype(int), zero_division=0)
            fold_f1.append(float(f1))
            # monitoring only: score this fold model on the held-out set, using
            # its own training-fold scaler. The result never enters the objective.
            test_probs.append(predict_probs(model, scaler.transform(X_test), device))
            torch.save({
                'state_dict': {n: v.detach().cpu() for n, v in model.state_dict().items()},
                'scaler_mean': torch.tensor(scaler.mean_),
                'scaler_scale': torch.tensor(scaler.scale_),
                'hp': hp,
            }, join(trial_dir, f'model_fold_{k + 1}.pth'))
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"  trial {trial.number} fold {k + 1}/{len(splits)} inner F1 = {f1:.4f}")

        trial.set_user_attr('fold_f1', fold_f1)
        mean_f1 = float(np.mean(fold_f1))

        test_probs = np.stack(test_probs)
        ho_each = [float(f1_score(y_test, (p > 0.5).astype(int), zero_division=0))
                   for p in test_probs]
        ho_mean = float(np.mean(ho_each))
        ho_ens = float(f1_score(y_test, (test_probs.mean(axis=0) > 0.5).astype(int),
                                zero_division=0))
        trial.set_user_attr('heldout_f1_per_fold', ho_each)
        trial.set_user_attr('heldout_f1_mean', ho_mean)
        trial.set_user_attr('heldout_f1_ens', ho_ens)

        print(f"Trial {trial.number}: inner F1 = {mean_f1:.4f} | held-out F1 "
              f"mean = {ho_mean:.4f}, ensemble = {ho_ens:.4f} (monitoring only)")
        return mean_f1

    return objective


def make_log_callback(out_dir):
    """After every completed trial: append a record to trial_log.jsonl and print
    the held-out F1 of the trial currently selected by inner-CV F1."""
    path = join(out_dir, 'trial_log.jsonl')

    def cb(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        ua = trial.user_attrs
        with open(path, 'a') as f:
            f.write(json.dumps({
                'trial': trial.number,
                'inner_f1': trial.value,
                'inner_f1_per_fold': ua.get('fold_f1'),
                'heldout_f1_mean': ua.get('heldout_f1_mean'),
                'heldout_f1_ens': ua.get('heldout_f1_ens'),
                'heldout_f1_per_fold': ua.get('heldout_f1_per_fold'),
            }) + '\n')
        best = study.best_trial
        bh = best.user_attrs.get('heldout_f1_mean')
        if bh is None:  # trial from a study created before this logging existed
            print(f"[best so far] trial {best.number}: inner F1 = {best.value:.4f}")
        else:
            print(f"[best so far] trial {best.number}: inner F1 = {best.value:.4f}, "
                  f"held-out F1 (mean) = {bh:.4f}, optimism = {best.value - bh:+.4f}")
    return cb


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


# --------------------------------------------------------------------------- #
# Held-out evaluation (run once, after the search is finished)
# --------------------------------------------------------------------------- #
def evaluate_heldout(study, ckpt_root, X_test, y_test, names_test, row_ids_test,
                     args, device, out_dir):
    best = study.best_trial
    trial_dir = join(ckpt_root, f'trial_{best.number}')
    if not os.path.isdir(trial_dir):
        raise FileNotFoundError(f"Checkpoints for best trial {best.number} not found "
                                f"in {trial_dir}")

    probs = []
    for k in range(args.inner_folds):
        ck = torch.load(join(trial_dir, f'model_fold_{k + 1}.pth'), map_location=device)
        model = build_model(X_test.shape[1], ck['hp'], device)
        model.load_state_dict(ck['state_dict'])
        # each fold model is scored with ITS OWN training-fold scaler
        Xs = (X_test - ck['scaler_mean'].numpy()) / ck['scaler_scale'].numpy()
        probs.append(predict_probs(model, Xs, device))
    probs = np.stack(probs)                      # (folds, n_test)
    preds = (probs > 0.5).astype(int)
    ens_prob = probs.mean(axis=0)
    ens_pred = (ens_prob > 0.5).astype(int) 

    f1_each = [float(f1_score(y_test, p, zero_division=0)) for p in preds]
    f1_mean = float(np.mean(f1_each))
    f1_ens = float(f1_score(y_test, ens_pred, zero_division=0))

    # bootstrap CI over held-out samples (few repeaters => wide)
    rng = np.random.default_rng(args.seed)
    n = len(y_test)
    boot_mean, boot_ens = [], []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, n)
        boot_mean.append(np.mean([f1_score(y_test[idx], p[idx], zero_division=0)
                                  for p in preds]))
        boot_ens.append(f1_score(y_test[idx], ens_pred[idx], zero_division=0))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    inner_f1 = float(best.value)
    completed = sorted([t.value for t in study.trials if t.value is not None], reverse=True)
    results = {
        'best_trial': best.number,
        'best_params': best.params,
        'n_completed_trials': len(completed),
        'inner_cv_f1_mean': inner_f1,
        'inner_cv_f1_per_fold': best.user_attrs.get('fold_f1'),
        'top10_inner_cv_f1': completed[:10],
        'heldout_n': int(n),
        'heldout_n_repeaters': int(y_test.sum()),
        'heldout_f1_per_fold_model': f1_each,
        'heldout_f1_mean_of_fold_models': f1_mean,
        'heldout_f1_mean_ci95': ci(boot_mean),
        'heldout_f1_ensemble': f1_ens,
        'heldout_f1_ensemble_ci95': ci(boot_ens),
        'optimism_vs_mean_of_fold_models': inner_f1 - f1_mean,
        'optimism_vs_ensemble': inner_f1 - f1_ens,
        'heldout_accuracy_ensemble': float(accuracy_score(y_test, ens_pred)),
        'heldout_precision_ensemble': float(precision_score(y_test, ens_pred, zero_division=0)),
        'heldout_recall_ensemble': float(recall_score(y_test, ens_pred, zero_division=0)),
        'heldout_confusion_matrix_ensemble': confusion_matrix(y_test, ens_pred).tolist(),
    }
    with open(join(out_dir, 'heldout_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    pred_df = pd.DataFrame({'row_id': row_ids_test, 'tns_name': names_test, 'label': y_test})
    for k in range(probs.shape[0]):
        pred_df[f'prob_fold{k + 1}'] = probs[k]
    pred_df['prob_ensemble'] = ens_prob
    pred_df['pred_ensemble'] = ens_pred
    pred_df.to_csv(join(out_dir, 'heldout_predictions.csv'), index=False)

    print("\n=== Held-out evaluation (best trial %d) ===" % best.number)
    print(f"Held-out set: {n} bursts, {int(y_test.sum())} repeaters")
    print(f"Inner-CV F1 (selected, optimistic)      : {inner_f1:.4f}")
    print(f"Held-out F1, mean of 3 fold models      : {f1_mean:.4f}  "
          f"95% CI {ci(boot_mean)[0]:.3f}-{ci(boot_mean)[1]:.3f}")
    print(f"Held-out F1, ensemble of 3 fold models  : {f1_ens:.4f}  "
          f"95% CI {ci(boot_ens)[0]:.3f}-{ci(boot_ens)[1]:.3f}")
    print(f"Optimism (inner - held-out, mean models): {inner_f1 - f1_mean:+.4f}")
    print(f"Optimism (inner - held-out, ensemble)   : {inner_f1 - f1_ens:+.4f}")
    print(classification_report(y_test, ens_pred,
                                target_names=['Non-Repeater', 'Repeater'], zero_division=0))
    print(f"Saved results to {out_dir}")


# --------------------------------------------------------------------------- #
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='chimefrbcat1.csv')
    p.add_argument('--out_dir',
                   default='/scratch/gpfs/MLISANTI/ra0438/old_frb_paper/heldout_run_seed42',
                   help='checkpoints, study.db, split and results all go here')
    p.add_argument('--n_trials', type=int, default=350,
                   help='TOTAL completed trials wanted; on resume only the rest are run '
                        '(0 = skip search, just evaluate)')
    p.add_argument('--seed', type=int, default=42, help='outer split / sampler / init seed')
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--inner_folds', type=int, default=3)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_boot', type=int, default=2000)
    p.add_argument('--keep_all_checkpoints', action='store_true',
                   help='keep every trial\'s checkpoints (default: only the best so far)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.abspath(args.out_dir)
    ckpt_root = join(out_dir, 'checkpoints')
    os.makedirs(ckpt_root, exist_ok=True)
    set_seed(args.seed)

    # ---- data and the ONE outer split ------------------------------------- #
    X_df, y, names = load_data(args.data)
    X = X_df.to_numpy(dtype=np.float64)
    row_ids = X_df.index.to_numpy()
    dev_pos, test_pos = train_test_split(np.arange(len(y)), test_size=args.test_size,
                                         stratify=y, random_state=args.seed)
    dev_pos, test_pos = np.sort(dev_pos), np.sort(test_pos)
    X_dev, y_dev = X[dev_pos], y[dev_pos]
    X_test, y_test = X[test_pos], y[test_pos]
    print(f"[split] dev: {len(y_dev)} ({int(y_dev.sum())} repeaters) | "
          f"held-out: {len(y_test)} ({int(y_test.sum())} repeaters)")
    with open(join(out_dir, 'split.json'), 'w') as f:
        json.dump({'seed': args.seed, 'test_size': args.test_size,
                   'dev_row_ids': row_ids[dev_pos].tolist(),
                   'test_row_ids': row_ids[test_pos].tolist()}, f)

    # ---- Optuna on the dev part only --------------------------------------- #
    storage = f"sqlite:///{join(out_dir, 'study.db')}"
    study = optuna.create_study(
        direction='maximize',
        study_name=f'svae_heldout_seed{args.seed}_test{int(round(args.test_size * 100))}',
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=True,
    )
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, args.n_trials - done)
    print(f"[optuna] {done} completed trials found, running {remaining} more")
    if remaining:
        callbacks = [make_log_callback(out_dir)]
        if not args.keep_all_checkpoints:
            callbacks.append(make_cleanup_callback(ckpt_root))
        study.optimize(make_objective(X_dev, y_dev, X_test, y_test, args, device, ckpt_root),
                       n_trials=remaining, callbacks=callbacks)

    # ---- one-time held-out evaluation -------------------------------------- #
    evaluate_heldout(study, ckpt_root, X_test, y_test, names[test_pos], row_ids[test_pos],
                     args, device, out_dir)


if __name__ == '__main__':
    main()