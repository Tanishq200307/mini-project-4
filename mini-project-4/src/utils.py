# src/utils.py
import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_logits, all_y, all_x = [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
        all_x.append(x.detach().cpu())
    return torch.cat(all_logits), torch.cat(all_y), torch.cat(all_x)


def softmax_probs(logits: torch.Tensor):
    return torch.softmax(logits, dim=1)


def compute_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return (y_true == y_pred).mean()


def make_cost_matrix():
    """
    Default: cost 1 for any mistake.
    Then override specific business costs:
      High=5, Medium=3, Low=1  (you can change these)
    """
    C = np.ones((10, 10), dtype=np.float32)
    np.fill_diagonal(C, 0.0)

    HIGH = 5.0
    MED = 3.0
    LOW = 1.0

    # Bag -> Sneaker (low)
    C[8, 7] = LOW

    # Shirt -> T-shirt (high)
    C[6, 0] = HIGH

    # Coat -> Pullover (high)
    C[4, 2] = HIGH

    # Sandal -> Sneaker (medium)
    C[5, 7] = MED

    # Ankle boot -> Sneaker (medium)
    C[9, 7] = MED

    return C


def cost_weighted_accuracy(y_true, y_pred, cost_matrix):
    """
    Convert costs into a normalized "cost-weighted accuracy" in [0,1].
    Lower cost is better.

    Score = 1 - (total_cost / worst_possible_cost)
    where worst_possible_cost assumes every example is misclassified
    with the maximum cost in its true row.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total_cost = 0.0
    worst_cost = 0.0

    for t, p in zip(y_true, y_pred):
        total_cost += float(cost_matrix[t, p])
        worst_cost += float(cost_matrix[t].max())

    if worst_cost == 0:
        return 1.0
    return 1.0 - (total_cost / worst_cost)


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=list(range(10)))


def per_class_accuracy(cm: np.ndarray):
    # cm rows = true, cols = pred
    row_sums = cm.sum(axis=1)
    acc = np.zeros(10, dtype=np.float32)
    for i in range(10):
        acc[i] = (cm[i, i] / row_sums[i]) if row_sums[i] > 0 else 0.0
    return acc
