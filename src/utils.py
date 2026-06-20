import yaml
import random
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

import torch


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif abs(self.best_score - current_score) < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0
        return self.early_stop


def load_config(cfg_path):
    with open(cfg_path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset_ids(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]


def calculate_metrics(preds, targets):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    preds = preds.ravel()
    targets = targets.ravel()

    cm = confusion_matrix(targets, preds)
    iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    precision = precision_score(targets, preds, average=None, zero_division=0)
    recall = recall_score(targets, preds, average=None, zero_division=0)
    f1 = f1_score(targets, preds, average=None, zero_division=0)
    accuracy = accuracy_score(targets, preds)

    class_metrics = {}
    for i in range(4):
        class_metrics[f'class_{i}'] = {
            'iou': iou[i],
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i]
        }

    return {
        'class_metrics': class_metrics,
        'mean_iou': np.mean(iou),
        'mean_precision': np.mean(precision),
        'mean_recall': np.mean(recall),
        'mean_f1_score': np.mean(f1),
        'accuracy': accuracy
    }
