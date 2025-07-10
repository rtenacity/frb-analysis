import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, recall_score
from itertools import product
from sklearn.model_selection import StratifiedKFold
import optuna
import plotly.express as px
from collections import Counter
import umap.umap_ as umap
import matplotlib
from sklearn.manifold import Isomap
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
import sklearn.metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from os.path import join
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})
def fill_repeater_from_source(row, data):
    if row['Source'] == 'FRB20220912A':
        return 1
    else:
        return row['Repeater']
frb_data = pd.read_csv('frb-data.csv')
frb_data['Repeater'] = frb_data['Repeater'].map({'Yes': 1, 'No': 0})
frb_data['Repeater'] = frb_data['Repeater'].fillna(0)
frb_data['Repeater'] = frb_data['Repeater'].astype(int)
frb_data['Repeater'] = frb_data.apply(fill_repeater_from_source, axis=1, data=frb_data)

frb_data['Repeater'].isna().sum()
labels = frb_data['Repeater']

# Function to clean numerical strings and convert to float
def clean_numeric_value(value):
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return np.nan
        try:
            # Remove special characters and split if necessary
            for char in ['/', '+', '<', '>', '~']:
                value = value.replace(char, '')
            if '-' in value:
                value = value.split('-')[0]
            return float(value)
        except ValueError:
            return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan
    
error_features = [
    'DM_SNR', 'DM_alig', 'Flux_density', 'Fluence', 'Energy',
    'Polar_l', 'Polar_c', 'RM_syn', 'RM_QUfit', 'Scatt_t', 
    #'Scin_f'
]
base_features = [
    'Observing_band', 
    # 'GL', 'GB', 'SNR', 
    'Freq_high',
    'Freq_low', 'Freq_peak', 
    'Width'
    # 'Repeater',
    #'MJD'
]

source_counts = frb_data.groupby('Source').size()            # # signals per source
unique_sources = frb_data['Source'].nunique()                # N
repeater_sources = frb_data.loc[frb_data['Repeater']==1,'Source'].nunique()   # n_r
nonrep_sources = frb_data.loc[frb_data['Repeater']==0,'Source'].nunique()     # n_nr

# 2) Broadcast n_s to each row
frb_data['n_s'] = frb_data['Source'].map(source_counts)

# 3) Compute the per-row weight
def compute_weight(row):
    if row['Repeater'] == 1:
        return (1.0 / row['n_s']) * (repeater_sources / unique_sources)
    else:
        return nonrep_sources / unique_sources

frb_data['weight'] = frb_data.apply(compute_weight, axis=1)
# weights = np.ones(len(frb_data['weight']))
weights = frb_data['weight']


for feature in base_features + error_features:
    frb_data[feature] = frb_data[feature].apply(clean_numeric_value)

for feature in error_features:
    frb_data[f'{feature}_err'] = frb_data[f'{feature}_err'].apply(clean_numeric_value)

for feature in error_features:
    frb_data[f'{feature}_upper'] = frb_data[feature] + frb_data[f'{feature}_err']
    frb_data[f'{feature}_lower'] = frb_data[feature] - frb_data[f'{feature}_err']
    frb_data[f'{feature}_lower'] = frb_data[f'{feature}_lower'].clip(lower=0)

features = (
    base_features +
    error_features +
    [f'{feature}_upper' for feature in error_features] +
    [f'{feature}_lower' for feature in error_features]
)
frb_data_clean = frb_data[features].fillna(0)
scaler = StandardScaler()
frb_data_scaled = scaler.fit_transform(frb_data_clean)

# Retain the original indices
indices = frb_data_clean.index

# Split the data and retain indices
train_data, val_data, train_labels, val_labels, train_indices, val_indices = train_test_split(
    frb_data_scaled, labels, indices, test_size=0.2, random_state=42, stratify=labels
)

# Convert to PyTorch tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32)
val_tensor = torch.tensor(val_data, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.long)
val_labels_tensor = torch.tensor(val_labels.values, dtype=torch.long)

# Create datasets and dataloaders
batch_size = 64
train_dataset = TensorDataset(train_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_tensor, val_labels_tensor)


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

def loss_function(recon_x, x, mu, logvar, class_prob, labels, beta, gamma, class_weight, classification_multiplier):
    reconstruction_loss_fn = nn.MSELoss(reduction='sum')
    pos_weight = torch.tensor([class_weight], dtype=torch.float32, device=device)
    classification_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # check this loss function
    recon_loss = reconstruction_loss_fn(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    class_loss = classification_multiplier * classification_loss_fn(class_prob, labels.unsqueeze(1).float())
    total_loss = recon_loss + beta * kl_loss + gamma * class_loss
    return total_loss, recon_loss, kl_loss, class_loss

def weighted_loss_function(recon_x, x, mu, logvar, class_prob, 
                           labels, sample_weights, class_weight,
                           beta, gamma, classification_multiplier):
    
    recon_per_elem = F.mse_loss(recon_x, x, reduction='none')
    recon_per_sample = recon_per_elem.view(recon_per_elem.size(0), -1).sum(dim=1)

    # KL: closed-form per-sample
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    # classification: per-sample
    class_per_sample = F.binary_cross_entropy_with_logits(
        class_prob.squeeze(), 
        labels.float(), 
        reduction='none',
        pos_weight=torch.tensor([class_weight], dtype=torch.float32, device=device)
    )

    # combine terms
    loss_per_sample = (
        recon_per_sample
        + beta * kl_per_sample
        + gamma * classification_multiplier * class_per_sample
    )

    # now weight each sample
    # Option A: simple mean of weighted losses
    total_loss = torch.mean(sample_weights * loss_per_sample)
    recon_loss = torch.mean(sample_weights * recon_per_sample)
    kl_loss = torch.mean(sample_weights * kl_per_sample)
    class_loss = torch.mean(sample_weights * class_per_sample)
    
    return total_loss, recon_loss, kl_loss, class_loss


input_dim = val_tensor.shape[1]
stop_patience = 8


def evaluate_classifier(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)
            class_logits = model(data)[-1]
            preds = (class_logits > 0.5).float().cpu().numpy().squeeze()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=["Non-Repeater", "Repeater"])
    conf_matrix = confusion_matrix(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted')

    
    false_positives = np.sum((all_labels == 0) & (all_preds == 1))

    return accuracy, class_report, conf_matrix, recall, false_positives  # Return F1 score as well

def get_activation_function(name):
    if name == 'ReLU':
        return nn.ReLU()
    elif name == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif name == 'ELU':
        return nn.ELU()
    elif name == 'SELU':
        return nn.SELU()
    elif name == 'GELU':
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation function: {name}")
    
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
import sklearn.metrics

def evaluate_classifier_full(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels, _ in dataloader:
            data = data.to(device)
            class_logits = model(data)[-1]
            preds = (class_logits > 0.5).float().cpu().numpy().squeeze()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = sklearn.metrics.f1_score(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=["Non-Repeater", "Repeater"])
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, class_report, conf_matrix, all_preds, all_labels

original_data = pd.read_csv('frb-data.csv')
original_data['Repeater'] = original_data['Repeater'].map({'Yes': 1, 'No': 0})
print(original_data['Repeater'].isna().sum())

print(f"Number of NaN values in 'Repeater' column before processing: {original_data['Repeater'].isna().sum()}")
# Apply the function row-wise
original_data['Repeater'] = original_data.apply(fill_repeater_from_source, axis=1, data=original_data)

print(f"Number of NaN values in 'Repeater' column after processing: {original_data['Repeater'].isna().sum()}")
best_params = {'hidden_dim': 1082, 'latent_dim': 18, 'beta': 1.149574612306723, 'gamma': 1.9210647260496314, 'dropout_rate': 0.13093239424733344, 'lr': 0.0011823749066137313, 'scheduler_patience': 7, 'class_weight': 0.35488674730648145, 'activation': 'ReLU', 'classification_multiplier': 7817.124805902009}

beta = best_params["beta"]
gamma = best_params["gamma"]
lr = best_params["lr"]
scheduler_patience = best_params["scheduler_patience"]
num_epochs = 150
def train_supervised(model, optimizer, scheduler, epoch, beta, gamma, class_weight, classification_multiplier, train_loader):
    model.train()
    train_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    classification_loss_total = 0
    
    correct = 0
    total = 0
    
    for batch_idx, (data, labels, sample_weights) in enumerate(train_loader):
        data, labels, sample_weights = data.to(device), labels.to(device), sample_weights.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, class_logits = model(data)
        
        # Supervised loss function
        loss, recon_loss, kl_loss, classification_loss = weighted_loss_function(
            recon_batch, data, mu, logvar, class_logits, labels, sample_weights, class_weight,
            beta, gamma, classification_multiplier
        )
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        recon_loss_total += recon_loss.item()
        kl_loss_total += kl_loss.item()
        classification_loss_total += classification_loss.item()
        
        predicted = (class_logits > 0.5).float()
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()
        
        # if batch_idx % 100 == 0:
            # print(classification_loss)
            # print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
            #       f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    # Calculate average loss and accuracy for the epoch
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon = recon_loss_total / len(train_loader.dataset)
    avg_kl = kl_loss_total / len(train_loader.dataset)
    avg_class = classification_loss_total / len(train_loader.dataset)
    accuracy = correct / total
    
    # print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, '
    #       f'Class: {avg_class:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, avg_recon, avg_kl, avg_class, accuracy

def validate_supervised(model, scheduler, optimizer, epoch, beta, gamma, class_weight, classification_multiplier, val_loader):
    model.eval()
    val_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    classification_loss_total = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels, sample_weights in val_loader:
            data, labels, sample_weights = data.to(device), labels.to(device) , sample_weights.to(device)
            recon_batch, mu, logvar, class_logits = model(data)
            
            loss, recon_loss, kl_loss, classification_loss = weighted_loss_function(
                recon_batch, data, mu, logvar, class_logits, labels, sample_weights, class_weight,
                beta, gamma, classification_multiplier
            )
            
            val_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
            classification_loss_total += classification_loss.item()
            
            predicted = (class_logits > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()
    
    avg_loss = val_loss / len(val_loader.dataset)
    avg_recon = recon_loss_total / len(val_loader.dataset)
    avg_kl = kl_loss_total / len(val_loader.dataset)
    avg_class = classification_loss_total / len(val_loader.dataset)
    accuracy = correct / total
    
    # print(f'====> Validation loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, '
    #       f'Class: {avg_class:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, avg_recon, avg_kl, avg_class, accuracy


def early_stopping(val_losses, patience):
    if len(val_losses) > patience:
        if all(val_losses[-i-1] <= val_losses[-i] for i in range(1, patience+1)):
            return True
    return False


garcia_list = '''
FRB20180907E
FRB20180920B
FRB20180928A
FRB20181017B
FRB20181022E
FRB20181125A
FRB20181125A
FRB20181125A
FRB20181214A
FRB20181220A
FRB20181226E
FRB20181229B
FRB20190112A
FRB20190128C
FRB20190206B
FRB20190206A
FRB20190218B
FRB20190223A
FRB20190308C
FRB20190308C
FRB20190323D
FRB20190329A
FRB20190410A
FRB20190412B
FRB20190423B
FRB20190423B
FRB20190429B
FRB20190430A
FRB20190527A
FRB20190527A
FRB20190601C
FRB20190601C
FRB20190617B
FRB20180910A
FRB20190210C
FRB20200726D
'''.split()

luo_list = '''
FRB20181229B
FRB20190423B
FRB20190410A
FRB20181017B
FRB20181128C
FRB20190422A
FRB20190409B
FRB20190329A
FRB20190423B
FRB20190206A
FRB20190128C
FRB20190106A
FRB20190129A
FRB20181030E
FRB20190527A
FRB20190218B
FRB20190609A
FRB20190412B
FRB20190125B
FRB20181231B
FRB20181221A
FRB20190112A
FRB20190125A
FRB20181218C
FRB20190429B
FRB20190109B
FRB20190206B
'''.split()

zhu_ge_list = '''
FRB20180911A
FRB20180915B
FRB20180920B
FRB20180923A
FRB20180923C
FRB20180928A
FRB20181013E
FRB20181017B
FRB20181030E
FRB20181125A
FRB20181125A
FRB20181125A
FRB20181130A
FRB20181214A
FRB20181220A
FRB20181221A
FRB20181226E
FRB20181229B
FRB20181231B
FRB20190106B
FRB20190109B
FRB20190110C
FRB20190111A
FRB20190112A
FRB20190129A
FRB20190204A
FRB20190206A
FRB20190218B
FRB20190220A
FRB20190221A
FRB20190222B
FRB20190223A
FRB20190228A
FRB20190308C
FRB20190308C
FRB20190308B
FRB20190308B
FRB20190323D
FRB20190329A
FRB20190403E
FRB20190409B
FRB20190410A
FRB20190412B
FRB20190418A
FRB20190419A
FRB20190422A
FRB20190422A
FRB20190423A
FRB20190423B
FRB20190423B
FRB20190429B
FRB20190430A
FRB20190517C
FRB20190527A
FRB20190527A
FRB20190531C
FRB20190601B
FRB20190601C
FRB20190601C
FRB20190609A
FRB20190617A
FRB20190617B
FRB20190618A
FRB20190625A
'''.split()

# all_false_positives = []
# all_false_negatives = []
# all_true_positives = []
# all_true_negatives = []

# num_epochs = 100

# n_folds = 5
# skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# full_data_tensor = torch.tensor(frb_data_scaled, dtype=torch.float32)
# full_labels_tensor = torch.tensor(labels.values, dtype=torch.long)

# for fold, (train_index, val_index) in enumerate(skf.split(frb_data_scaled, labels)):
#     print(f"\n=== Fold {fold + 1}/{n_folds} ===")
    
#     train_data, val_data = frb_data_scaled[train_index], frb_data_scaled[val_index]
#     train_labels, val_labels = labels.iloc[train_index], labels.iloc[val_index]
    
    
#     train_tensor = torch.tensor(train_data, dtype=torch.float32)
#     val_tensor = torch.tensor(val_data, dtype=torch.float32)
#     train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.long)
#     val_labels_tensor = torch.tensor(val_labels.values, dtype=torch.long)
    
#     train_weights = torch.tensor(weights.iloc[train_index].values, dtype=torch.float32)
#     val_weights   = torch.tensor(weights.iloc[val_index].values,   dtype=torch.float32)
    
#     # train_weights = torch.tensor(weights[train_index], dtype=torch.float32)
#     # val_weights   = torch.tensor(weights[val_index],   dtype=torch.float32)


    
#     train_dataset = TensorDataset(train_tensor, train_labels_tensor, train_weights)
#     val_dataset   = TensorDataset(val_tensor,   val_labels_tensor,   val_weights)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    
#     best_model = SupervisedVAE(
#         input_dim,
#         best_params["hidden_dim"],
#         best_params["latent_dim"],
#         best_params["dropout_rate"],
#         get_activation_function(best_params["activation"])
#     ).to(device)
    
#     optimizer = torch.optim.Adam(best_model.parameters(), lr=lr)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=scheduler_patience)
    
#     # Train the model
#     for epoch in range(1, num_epochs + 1):
#         train_loss, _, _, _, train_accuracy = train_supervised(best_model, optimizer, scheduler, epoch, beta, gamma, best_params['class_weight'], best_params['classification_multiplier'])
#         val_loss, _, _, _, val_accuracy = validate_supervised(best_model, optimizer, scheduler, epoch, beta, gamma, best_params['class_weight'], best_params['classification_multiplier'])
#         # train_loss, _, _, _, train_accuracy = train_supervised(best_model, optimizer, scheduler, epoch, beta, gamma, 1, best_params['classification_multiplier'])
#         # val_loss, _, _, _, val_accuracy = validate_supervised(best_model, optimizer, scheduler, epoch, beta, gamma, 1, best_params['classification_multiplier'])
#         scheduler.step(val_loss)
        
#         # Early stopping
#         if early_stopping([val_loss], stop_patience):
#             print(f"Early stopping triggered at epoch {epoch}")
#             break

#     val_accuracy, val_class_report, val_conf_matrix, val_preds, val_labels = evaluate_classifier_full(best_model, val_loader, device)
    
#     # print(f"Validation Accuracy: {val_accuracy:.4f}")
#     # print("Classification Report:\n", val_class_report)
#     # print("Confusion Matrix:\n", val_conf_matrix)
    
#     misclassified_non_repeaters = (val_labels == 0) & (val_preds == 1)
#     misclassified_indices = val_index[misclassified_non_repeaters]
#     misclassified_sources = original_data.loc[misclassified_indices, "Source"].drop_duplicates()
    
#     false_positives_fold = original_data.loc[val_index[(val_labels == 0) & (val_preds == 1)], "Source"]
#     false_negatives_fold = original_data.loc[val_index[(val_labels == 1) & (val_preds == 0)], "Source"]
#     true_positives_fold = original_data.loc[val_index[(val_labels == 1) & (val_preds == 1)], "Source"]
#     true_negatives_fold = original_data.loc[val_index[(val_labels == 0) & (val_preds == 0)], "Source"]
    
#     # # fold_false_positives = []
#     # for source in misclassified_sources:
#     #     # fold_false_positives.append(source)
#     #     if source in garcia_list or source in luo_list or source in zhu_ge_list:
#     #         print(f"False positive in fold {fold + 1}: {source}")
            
#     all_false_negatives.extend(false_negatives_fold)
#     all_true_positives.extend(true_positives_fold)
#     all_true_negatives.extend(true_negatives_fold)
#     all_false_positives.extend(false_positives_fold)
    
    
# all_false_positives = pd.Series(all_false_positives)
# all_false_negatives = pd.Series(all_false_negatives)
# all_true_positives = pd.Series(all_true_positives)
# all_true_negatives = pd.Series(all_true_negatives)
# print("")

# print("\n=== Summary ===")
# print(f"Total False Positives: {all_false_positives.size}")
# print(f"Total False Negatives: {all_false_negatives.size}")
# print(f"Total True Positives: {all_true_positives.size}")
# print(f"Total True Negatives: {all_true_negatives.size}")

# conf_mat_dups = np.zeros((2, 2))
# conf_mat_dups[0, 0] = all_true_negatives.size
# conf_mat_dups[0, 1] = all_false_positives.size
# conf_mat_dups[1, 0] = all_false_negatives.size
# conf_mat_dups[1, 1] = all_true_positives.size


# conf_mat_dups = pd.DataFrame(conf_mat_dups, index=["Non-Repeater", "Repeater"], columns=["Non-Repeater", "Repeater"])
# print("\nConfusion Matrix (with duplicates):")
# print(conf_mat_dups)

# print("accuracy_score")
# accuracy = (all_true_positives.size + all_true_negatives.size) / (all_false_positives.size + all_false_negatives.size + all_true_positives.size + all_true_negatives.size)
# print(accuracy)
i = 0
accuracy_max = 0

def objective(trial):
    global i, accuracy_max
    hidden_dim = trial.suggest_int('hidden_dim', 128, 2048)
    latent_dim = trial.suggest_int('latent_dim', 5, 40)
    beta = trial.suggest_float('beta', 0.1, 2.0)
    gamma = trial.suggest_float('gamma', 0.1, 2.0)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2)
    scheduler_patience = trial.suggest_int('scheduler_patience', 2, 7)
    class_weight = trial.suggest_float('class_weight', 0.05, 3)

    activation_name = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'ELU', 'SELU', 'GELU'])
    activation = get_activation_function(activation_name)
    classification_multiplier = trial.suggest_float('classification_multiplier', 5000, 15000)

    pos_weight = torch.tensor([class_weight], dtype=torch.float32, device=device)

    all_false_positives = []
    all_false_negatives = []
    all_true_positives = []
    all_true_negatives = []

    num_epochs = 100

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    full_data_tensor = torch.tensor(frb_data_scaled, dtype=torch.float32)
    full_labels_tensor = torch.tensor(labels.values, dtype=torch.long)

    for fold, (train_index, val_index) in enumerate(skf.split(frb_data_scaled, labels)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        
        train_data, val_data = frb_data_scaled[train_index], frb_data_scaled[val_index]
        train_labels, val_labels = labels.iloc[train_index], labels.iloc[val_index]
        
        
        train_tensor = torch.tensor(train_data, dtype=torch.float32)
        val_tensor = torch.tensor(val_data, dtype=torch.float32)
        train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.long)
        val_labels_tensor = torch.tensor(val_labels.values, dtype=torch.long)
        
        train_weights = torch.tensor(weights.iloc[train_index].values, dtype=torch.float32)
        val_weights   = torch.tensor(weights.iloc[val_index].values,   dtype=torch.float32)
        
        # train_weights = torch.tensor(weights[train_index], dtype=torch.float32)
        # val_weights   = torch.tensor(weights[val_index],   dtype=torch.float32)


        
        train_dataset = TensorDataset(train_tensor, train_labels_tensor, train_weights)
        val_dataset   = TensorDataset(val_tensor,   val_labels_tensor,   val_weights)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        
        best_model = SupervisedVAE(
            input_dim,
            hidden_dim,
            latent_dim,
            dropout_rate,
            activation
        ).to(device)
        
        optimizer = torch.optim.Adam(best_model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=scheduler_patience)
        
        # Train the model
        for epoch in range(1, num_epochs + 1):
            train_loss, _, _, _, train_accuracy = train_supervised(best_model, optimizer, scheduler, epoch, beta, gamma, class_weight, classification_multiplier, train_loader)
            val_loss, _, _, _, val_accuracy = validate_supervised(best_model, optimizer, scheduler, epoch, beta, gamma, class_weight, classification_multiplier, val_loader)
            # train_loss, _, _, _, train_accuracy = train_supervised(best_model, optimizer, scheduler, epoch, beta, gamma, 1, best_params['classification_multiplier'])
            # val_loss, _, _, _, val_accuracy = validate_supervised(best_model, optimizer, scheduler, epoch, beta, gamma, 1, best_params['classification_multiplier'])
            scheduler.step(val_loss)
            
            # Early stopping
            if early_stopping([val_loss], stop_patience):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        val_accuracy, val_class_report, val_conf_matrix, val_preds, val_labels = evaluate_classifier_full(best_model, val_loader, device)
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("Classification Report:\n", val_class_report)
        print("Confusion Matrix:\n", val_conf_matrix)
        
        misclassified_non_repeaters = (val_labels == 0) & (val_preds == 1)
        misclassified_indices = val_index[misclassified_non_repeaters]
        misclassified_sources = original_data.loc[misclassified_indices, "Source"].drop_duplicates()
        
        false_positives_fold = original_data.loc[val_index[(val_labels == 0) & (val_preds == 1)], "Source"]
        false_negatives_fold = original_data.loc[val_index[(val_labels == 1) & (val_preds == 0)], "Source"]
        true_positives_fold = original_data.loc[val_index[(val_labels == 1) & (val_preds == 1)], "Source"]
        true_negatives_fold = original_data.loc[val_index[(val_labels == 0) & (val_preds == 0)], "Source"]
        
        # # fold_false_positives = []
        # for source in misclassified_sources:
        #     # fold_false_positives.append(source)
        #     if source in garcia_list or source in luo_list or source in zhu_ge_list:
        #         print(f"False positive in fold {fold + 1}: {source}")
                
        all_false_negatives.extend(false_negatives_fold)
        all_true_positives.extend(true_positives_fold)
        all_true_negatives.extend(true_negatives_fold)
        all_false_positives.extend(false_positives_fold)
        
        
    all_false_positives = pd.Series(all_false_positives)
    all_false_negatives = pd.Series(all_false_negatives)
    all_true_positives = pd.Series(all_true_positives)
    all_true_negatives = pd.Series(all_true_negatives)
    print("")

    print("\n=== Summary ===")
    print(f"Total False Positives: {all_false_positives.size}")
    print(f"Total False Negatives: {all_false_negatives.size}")
    print(f"Total True Positives: {all_true_positives.size}")
    print(f"Total True Negatives: {all_true_negatives.size}")

    conf_mat_dups = np.zeros((2, 2))
    conf_mat_dups[0, 0] = all_true_negatives.size
    conf_mat_dups[0, 1] = all_false_positives.size
    conf_mat_dups[1, 0] = all_false_negatives.size
    conf_mat_dups[1, 1] = all_true_positives.size


    conf_mat_dups = pd.DataFrame(conf_mat_dups, index=["Non-Repeater", "Repeater"], columns=["Non-Repeater", "Repeater"])
    print("\nConfusion Matrix (with duplicates):")
    print(conf_mat_dups)

    print("accuracy_score")
    accuracy = (all_true_positives.size + all_true_negatives.size) / (all_false_positives.size + all_false_negatives.size + all_true_positives.size + all_true_negatives.size)
    print(accuracy)
    
    # print all parameters
    
    print(f"Trial {i + 1}: hidden_dim={hidden_dim}, latent_dim={latent_dim}, beta={beta}, gamma={gamma}, dropout_rate={dropout_rate}, lr={lr}, scheduler_patience={scheduler_patience}, class_weight={class_weight}, activation={activation_name}, classification_multiplier={classification_multiplier}")
    
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=350)