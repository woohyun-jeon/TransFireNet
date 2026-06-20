import os
from tqdm import tqdm
import numpy as np
import warnings
import rasterio
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from datasets import get_siamese_datasets
from models import TransFireNet
from utils import set_seed, load_config, load_dataset_ids, EarlyStopping, calculate_metrics
from losses import calculate_class_weights, FocalLoss


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, patience, min_delta):
    scaler = GradScaler('cuda')
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    best_model_state = None
    best_val_metrics = None

    for epoch in tqdm(range(num_epochs), desc='Training'):
        model.train()

        for pre_fire, post_fire, labels, _ in dataloaders['train']:
            pre_fire, post_fire, labels = pre_fire.to(device), post_fire.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(pre_fire, post_fire)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_metrics = evaluate_model(model, dataloaders['train'], criterion, device)
        val_metrics = evaluate_model(model, dataloaders['val'], criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}: ")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train IoU: {train_metrics['mean_iou']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val IoU: {val_metrics['mean_iou']:.4f}")

        scheduler.step()

        if best_val_metrics is None or val_metrics['mean_iou'] > best_val_metrics['mean_iou']:
            best_val_metrics = val_metrics
            best_model_state = model.state_dict().copy()

        if early_stopping(val_metrics['mean_iou']):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return best_model_state, best_val_metrics


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for pre_fire, post_fire, labels, _ in dataloader:
            pre_fire, post_fire, labels = pre_fire.to(device), post_fire.to(device), labels.to(device)
            outputs = model(pre_fire, post_fire)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * pre_fire.size(0)
            all_preds.append(torch.argmax(outputs, dim=1))
            all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(dataloader.dataset)

    return metrics


def save_predictions(model, dataloader, device, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for pre_fire, post_fire, _, filenames in dataloader:
            pre_fire, post_fire = pre_fire.to(device), post_fire.to(device)
            outputs = model(pre_fire, post_fire)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            for pred, filename in zip(predictions, filenames):
                save_path = os.path.join(save_dir, filename)
                pred = pred.astype(np.uint8)
                with rasterio.open(save_path, 'w', driver='GTiff', height=pred.shape[0], width=pred.shape[1], count=1, dtype='uint8') as dst:
                    dst.write(pred, 1)


def main():
    config_path = 'configs.yaml'
    cfgs = load_config(config_path)
    set_seed(seed=cfgs['params']['seed'])
    device = torch.device(f"cuda:{cfgs['params']['gpu_idx']}" if torch.cuda.is_available() else 'cpu')

    train_ids = load_dataset_ids(os.path.join(cfgs['path']['data_path'], 'train.txt'))
    val_ids = load_dataset_ids(os.path.join(cfgs['path']['data_path'], 'valid.txt'))
    test_ids = load_dataset_ids(os.path.join(cfgs['path']['data_path'], 'test.txt'))

    train_dataset, val_dataset, test_dataset = get_siamese_datasets(
        cfgs['path']['data_path'], train_ids, val_ids, test_ids
    )

    train_loader = DataLoader(train_dataset, batch_size=cfgs['params']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfgs['params']['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=cfgs['params']['batch_size'], shuffle=False, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    model = TransFireNet(in_channels=3, num_classes=4, pretrained=True).to(device)

    class_weights = calculate_class_weights(train_loader.dataset)
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2)

    optimizer = optim.Adam(model.parameters(), lr=cfgs['params']['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfgs['params']['num_epochs'])

    best_model_state, best_val_metrics = train_model(
        model, dataloaders, criterion, optimizer, scheduler, device,
        num_epochs=cfgs['params']['num_epochs'],
        patience=cfgs['params']['early_stopping_patience'],
        min_delta=cfgs['params']['early_stopping_min_delta']
    )

    model.load_state_dict(best_model_state)
    train_metrics = evaluate_model(model, train_loader, criterion, device)
    val_metrics = best_val_metrics
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    results = {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}

    os.makedirs(cfgs['path']['out_path'], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfgs['path']['out_path'], 'TransFireNet.pth'))
    save_predictions(model, test_loader, device, os.path.join(cfgs['path']['out_path'], 'TransFireNet_predictions'))

    print("\nFinal Results Summary:")
    with open(os.path.join(cfgs['path']['out_path'], 'result_summary.txt'), 'w') as f:
        for split in ['train', 'val', 'test']:
            metrics = results[split]
            print(f"  {split.capitalize()} Metrics:")
            f.write(f"  {split.capitalize()} Metrics:\n")
            for metric in ['mean_iou', 'mean_precision', 'mean_recall', 'mean_f1_score', 'accuracy', 'loss']:
                print(f"    {metric}: {metrics[metric]:.4f}")
                f.write(f"    {metric}: {metrics[metric]:.4f}\n")
            print()
            f.write("\n")


if __name__ == "__main__":
    main()
