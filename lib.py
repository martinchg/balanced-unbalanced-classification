import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- 1. DATA LOADING & PREPARATION ---

def load_rice_dataset(verbose=True):
    """
    Charge et prépare le dataset Rice.
    Retourne X, y et le DataFrame original.
    """
    from ucimlrepo import fetch_ucirepo
    
    if verbose: print("Fetching Rice dataset via ucimlrepo...")
    ds = fetch_ucirepo(id=545)
    
    X = ds.data.features.copy().astype(np.float32)
    y_raw = ds.data.targets.copy()
    
    # Encodage binaire (Cammeo=0, Osmancik=1)
    y = np.zeros(len(y_raw))
    # Selon le notebook, 0:1629 est une classe, le reste l'autre
    y[1630:] = 1 
    
    feature_names = list(X.columns)
    
    if verbose:
        print(f"Features: {feature_names}")
        print(f"Dataset shape: {X.shape}")
        
    return X, y, feature_names

def load_covtype_dataset(subsample_ratio=0.1, random_state=42, verbose=True):
    """
    Charge et sous-échantillonne le dataset CovType.
    """
    from sklearn.datasets import fetch_covtype
    
    if verbose: print("Downloading CovType dataset...")
    covtype_data = fetch_covtype()
    X_full = covtype_data.data
    y_full = covtype_data.target
    
    # Ajustement des labels pour PyTorch (1-7 -> 0-6)
    y_full = y_full - 1 
    
    # Shuffle et Subsample
    X_full, y_full = shuffle(X_full, y_full, random_state=random_state)
    n_subsample = int(X_full.shape[0] * subsample_ratio)
    
    X_red = X_full[:n_subsample]
    y_red = y_full[:n_subsample]
    
    if verbose:
        print(f"Original shape: {X_full.shape}")
        print(f"Reduced shape ({int(subsample_ratio*100)}%): {X_red.shape}")
        
    return X_red, y_red

def get_train_val_test_loaders(X, y, batch_size=128, test_size=0.2, val_size=0.1, random_state=42):
    """
    Prépare les DataLoaders PyTorch avec Standardisation.
    Gère la division Train/Val/Test.
    """
    # Conversion numpy si nécessaire
    if hasattr(X, 'to_numpy'): X = X.to_numpy().astype(np.float32)
    else: X = np.array(X, dtype=np.float32)
    
    y = np.array(y) # Les types seront castés en tensor plus tard

    # Split 1: Séparer le Test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Split 2: Séparer Train et Val depuis le reste
    # Ajustement de la proportion pour que Val représente val_size du TOTAL
    # Si test=0.1 et val=0.1 (total 0.2), il reste 0.9. Val doit être 1/9 du reste.
    # Ici on simplifie : val_size est une proportion de X_temp
    real_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=real_val_size, stratify=y_temp, random_state=random_state
    )

    # Standardisation (Fit sur Train uniquement)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Conversion Tenseurs
    # Classification binaire: y float pour BCEWithLogits, Multi: y long pour CrossEntropy
    is_multiclass = len(np.unique(y)) > 2
    dtype_y = torch.long if is_multiclass else torch.float32
    
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train, dtype=dtype_y)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val, dtype=dtype_y)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test, dtype=dtype_y)

    # Datasets & Loaders
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_train.shape[1]


# --- 2. VISUALIZATION UTILS ---

def plot_correlation_matrix(X, feature_names=None):
    """Affiche la matrice de corrélation."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)
        
    corr = X.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Matrice de Corrélation")
    plt.show()

def plot_scatter_classes(X, y, feat1_idx, feat2_idx, feature_names=None):
    """Affiche un scatter plot 2D coloré par classe."""
    if hasattr(X, 'iloc'): X = X.values
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, feat1_idx], X[:, feat2_idx], c=y, cmap='viridis', alpha=0.6, edgecolor='k')
    
    xlabel = feature_names[feat1_idx] if feature_names else f"Feature {feat1_idx}"
    ylabel = feature_names[feat2_idx] if feature_names else f"Feature {feat2_idx}"
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(scatter, label='Classe')
    plt.title(f"{xlabel} vs {ylabel}")
    plt.show()

def plot_history(history):
    """Trace les courbes de Loss et Accuracy pour PyTorch."""
    epochs = history["epoch"]
    
    plt.figure(figsize=(15, 5))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)
    
    # Learning Rate
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["lr"], label="LR", color='green')
    plt.xlabel("Epochs")
    plt.title("Learning Rate")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --- 3. SKLEARN MODELS & EVALUATION ---

def train_eval_sklearn(model_name, model_instance, X_train, y_train, X_test, y_test, use_scaler=True):
    """
    Entraîne et évalue un modèle sklearn (avec ou sans scaler).
    """
    steps = []
    if use_scaler:
        steps.append(('scaler', StandardScaler()))
    steps.append(('model', model_instance))
    
    pipe = Pipeline(steps)
    
    start = time.time()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    end = time.time()
    
    acc = accuracy_score(y_test, y_pred)
    print(f"--- {model_name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Temps d'exécution: {end - start:.4f}s")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return pipe, acc

def feature_importance_permutation(pipe, X, y, feature_names, n_repeats=20, groups=None, random_state=42):
    """
    Calcule l'importance des features par permutation (individuelle ou par groupe).
    """
    if hasattr(X, 'to_numpy'): X = X.to_numpy()
    
    base_preds = pipe.predict(X)
    base_acc = accuracy_score(y, base_preds)
    
    importances = []
    names = []
    rng = np.random.default_rng(random_state)
    
    # Si pas de groupes, on fait par feature individuelle
    if groups is None:
        iteration_list = [[i] for i in range(X.shape[1])]
        names_list = feature_names
    else:
        iteration_list = groups
        # Génération de noms pour les groupes si non fournis
        names_list = [f"Group {i}" for i in range(len(groups))]

    for i, indices in enumerate(iteration_list):
        scores_shuffled = []
        for _ in range(n_repeats):
            X_temp = X.copy()
            # Permutation globale appliquée à toutes les colonnes du groupe simultanément
            perm_idx = rng.permutation(X.shape[0])
            
            # Important : on applique la MEME permutation aux colonnes corrélées
            # X_temp[:, indices] = X_temp[perm_idx][:, indices] ne marche pas directement en numpy pur pour le slicing avancé
            # On doit le faire proprement :
            subset = X_temp[:, indices]
            X_temp[:, indices] = subset[perm_idx]
            
            shuffled_preds = pipe.predict(X_temp)
            scores_shuffled.append(accuracy_score(y, shuffled_preds))
        
        mean_drop = base_acc - np.mean(scores_shuffled)
        importances.append(mean_drop)
        names.append(names_list[i])
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(names, importances, color='orange')
    plt.xlabel("Baisse de précision moyenne")
    plt.title("Feature Importance (Permutation)")
    plt.gca().invert_yaxis()
    plt.show()
    
    return importances


# --- 4. PYTORCH DEEP LEARNING ---

class GenericMLP(nn.Module):
    """
    MLP générique pour binaire ou multiclasse.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 32], dropout_rate=0.2):
        super(GenericMLP, self).__init__()
        layers = []
        
        # Couches cachées
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) # BatchNorm pour stabilité
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        # Couche de sortie
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_torch_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=20, device='cpu', print_every=5):
    """
    Boucle d'entraînement PyTorch générique.
    """
    model.to(device)
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    
    is_binary = isinstance(criterion, nn.BCEWithLogitsLoss)

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(Xb)
            
            # Gestion dimension pour binaire
            if is_binary: outputs = outputs.squeeze()
            
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * Xb.size(0)
            
            # Calcul Accuracy
            if is_binary:
                preds = (torch.sigmoid(outputs) >= 0.5).float()
            else:
                preds = torch.argmax(outputs, dim=1)
                
            train_correct += (preds == yb).sum().item()
            train_total += Xb.size(0)
            
        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                out_v = model(Xv)
                
                if is_binary: out_v = out_v.squeeze()
                
                l_v = criterion(out_v, yv)
                val_loss += l_v.item() * Xv.size(0)
                
                if is_binary:
                    preds_v = (torch.sigmoid(out_v) >= 0.5).float()
                else:
                    preds_v = torch.argmax(out_v, dim=1)
                
                val_correct += (preds_v == yv).sum().item()
                val_total += Xv.size(0)
        
        # --- Metrics & Scheduler ---
        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Logging
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
    return history