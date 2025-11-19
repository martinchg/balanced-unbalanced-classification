# lib.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ==========================
# 1. Chargement des données
# ==========================

def load_spam(path: str) -> pd.DataFrame:
    """Charge le dataset SPAM de UCI."""
    # spambase.data n'a pas d'en-têtes, la dernière colonne = label
    df = pd.read_csv(path, header=None)
    return df


def load_diabetes(path: str) -> pd.DataFrame:
    """Charge le dataset Diabetes Kaggle."""
    df = pd.read_csv(path)
    return df


def make_X_y_spam(df: pd.DataFrame):
    """Sépare X, y pour SPAM (dernière colonne = cible)."""
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def make_X_y_diabetes(df: pd.DataFrame, target_col: str = "Diabetes_binary"):
    """Sépare X, y pour Diabetes (colonne cible = Diabetes_binary)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# ====================================
# 2. Split train / test + pré-traitement
# ====================================

def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    """Wrapper pour un split stratifié (conserve la proportion des classes)."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def make_preprocess():
    """Pré-traitement générique : imputation + standardisation."""
    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return preprocess


# =====================
# 3. Évaluation (TP)
# =====================

def eval_clf(y_true, y_pred, labels=None):
    """Fonction d’éval copiée du TP (accuracy + matrice de confusion)."""
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy = {acc:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()


def compute_metrics(y_true, y_pred, y_proba=None):
    """Renvoie un dict de métriques utiles (accuracy, f1, roc_auc...)."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = None
    return metrics


# ==========================================
# 4. Modèles de base (inspirés du TP)
# ==========================================

def build_knn_pipeline(n_neighbors=5):
    return Pipeline([
        ("preprocess", make_preprocess()),
        ("clf", KNeighborsClassifier(n_neighbors=n_neighbors)),
    ])


def build_logreg_pipeline(class_weight=None):
    return Pipeline([
        ("preprocess", make_preprocess()),
        ("clf", LogisticRegression(max_iter=1000, class_weight=class_weight)),
    ])


def build_rf_pipeline(
    n_estimators=200,
    max_depth=None,
    class_weight=None,
    random_state=42,
):
    return Pipeline([
        ("preprocess", make_preprocess()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


# ===================================
# 5. Choice of hyperparameters (TP-style)
# ===================================

def cv_knn_choose_k(X_train, y_train, K_max=20, n_splits=5, random_state=42):
    """Cherche le meilleur K comme dans le TP, avec StratifiedKFold."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = np.zeros((K_max, n_splits))

    for k in range(1, K_max + 1):
        pipe = build_knn_pipeline(n_neighbors=k)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        accs[k-1, :] = scores
        print(f"K = {k:2d}: acc = {scores.mean():.4f} ± {scores.std():.4f}")

    # on retourne le meilleur K + le tableau complet pour tracer dans le notebook
    best_k = np.argmax(accs.mean(axis=1)) + 1
    return best_k, accs


def cv_rf_choose_max_depth(X_train, y_train, depths=(5, 10, 20, 40, None), n_splits=5, random_state=42, class_weight=None):
    """Calque l'idée du TP: scanner max_depth et garder le meilleur."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

    for d in depths:
        pipe = build_rf_pipeline(max_depth=d, class_weight=class_weight)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        results[d] = scores
        print(f"max_depth = {d}: acc = {scores.mean():.4f} ± {scores.std():.4f}")

    # profondeur avec la meilleure moyenne
    best_depth = max(results, key=lambda d: results[d].mean())
    return best_depth, results


# =============================================
# 6. Permutation importance (copié du TP)
# =============================================

def permutation_importance(pipe, X, y, n_repeats=20, random_state=None, return_sorted=True):
    """
    Importance par permutation (inspirée du TP).
    - pipe : modèle déjà entraîné avec .predict(X)
    - X : numpy array (n_samples, n_features)
    - y : labels
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples, n_features = X.shape
    base_pred = np.ravel(pipe.predict(X))
    base_accuracy = float(np.mean(base_pred == y))

    importance_means = np.zeros(n_features, dtype=float)
    importance_stds = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        accs = np.empty(n_repeats, dtype=float)
        for r in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(n_samples)
            X_perm[:, j] = X_perm[perm_idx, j]
            pred_perm = np.ravel(pipe.predict(X_perm))
            accs[r] = float(np.mean(pred_perm == y))
        importance_means[j] = base_accuracy - accs.mean()
        importance_stds[j] = accs.std(ddof=0)

    result = {
        "feature_idx": np.arange(n_features),
        "base_accuracy": base_accuracy,
        "importance_mean": importance_means,
        "importance_std": importance_stds,
    }

    if return_sorted:
        order = np.argsort(result["importance_mean"])[::-1]
        for k in ["feature_idx", "importance_mean", "importance_std"]:
            result[k] = result[k][order]

    return result


# ==========================================
# 7. Workflows "haut niveau" pour les notebooks
# ==========================================

def workflow_spam(path_data: str):
    """Workflow complet pour le dataset SPAM (dataset plutôt équilibré)."""
    df = load_spam(path_data)
    X, y = make_X_y_spam(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    # 1) LogReg baseline
    log_pipe = build_logreg_pipeline(class_weight=None)
    log_pipe.fit(X_train, y_train)
    y_pred_log = log_pipe.predict(X_test)
    y_proba_log = log_pipe.predict_proba(X_test)[:, 1]
    metrics_log = compute_metrics(y_test, y_pred_log, y_proba_log)

    # 2) KNN avec choix du meilleur K via CV
    best_k, accs = cv_knn_choose_k(X_train, y_train, K_max=20)
    knn_pipe = build_knn_pipeline(n_neighbors=best_k)
    knn_pipe.fit(X_train, y_train)
    y_pred_knn = knn_pipe.predict(X_test)
    metrics_knn = compute_metrics(y_test, y_pred_knn)

    # 3) Random Forest
    best_depth, rf_results = cv_rf_choose_max_depth(X_train, y_train, class_weight=None)
    rf_pipe = build_rf_pipeline(max_depth=best_depth, class_weight=None)
    rf_pipe.fit(X_train, y_train)
    y_pred_rf = rf_pipe.predict(X_test)
    y_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]
    metrics_rf = compute_metrics(y_test, y_pred_rf, y_proba_rf)

    return {
        "logreg": metrics_log,
        "knn": metrics_knn,
        "rf": metrics_rf,
        "best_k": best_k,
        "best_depth": best_depth,
    }


def workflow_diabetes(path_data: str, target_col: str = "Diabetes_binary"):
    """Workflow complet pour le dataset Diabetes (dataset très déséquilibré)."""
    df = load_diabetes(path_data)
    X, y = make_X_y_diabetes(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    # On gère le déséquilibre avec class_weight="balanced"
    class_weight = "balanced"

    # 1) LogReg
    log_pipe = build_logreg_pipeline(class_weight=class_weight)
    log_pipe.fit(X_train, y_train)
    y_pred_log = log_pipe.predict(X_test)
    y_proba_log = log_pipe.predict_proba(X_test)[:, 1]
    metrics_log = compute_metrics(y_test, y_pred_log, y_proba_log)

    # 2) Random Forest
    best_depth, rf_results = cv_rf_choose_max_depth(
        X_train, y_train,
        class_weight=class_weight
    )
    rf_pipe = build_rf_pipeline(max_depth=best_depth, class_weight=class_weight)
    rf_pipe.fit(X_train, y_train)
    y_pred_rf = rf_pipe.predict(X_test)
    y_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]
    metrics_rf = compute_metrics(y_test, y_pred_rf, y_proba_rf)

    # Ici dans le notebook tu pourras aussi regarder recall de la classe 1, etc.
    return {
        "logreg": metrics_log,
        "rf": metrics_rf,
        "best_depth": best_depth,
    }