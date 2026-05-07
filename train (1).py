import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

import argparse
from dataset import BraTSDataset, MergedDataset
from models import Generator, Discriminator, CNNClassifier, ResNetClassifier, ViTClassifier

def calculate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5
    return acc, rec, prec, f1, auc

def train_cgan(generator, discriminator, dataloader, epochs, z_dim, device):
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    generator.train()
    discriminator.train()
    
    print("Training cGAN...")
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            valid = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            z = torch.randn((batch_size, z_dim), device=device)
            gen_imgs = generator(z, labels)
            
            validity = discriminator(gen_imgs, labels)
            g_loss = criterion(validity, valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = criterion(validity_real, valid)
            
            validity_fake = discriminator(gen_imgs.detach(), labels)
            d_fake_loss = criterion(validity_fake, fake)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
        print(f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

def generate_synthetic_data(generator, num_samples, class_label, z_dim, device):
    generator.eval()
    batch_size = 64
    synthetic_imgs = []
    synthetic_labels = []
    
    print(f"Generating {num_samples} synthetic images for class {class_label}...")
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            bs = min(batch_size, num_samples - i)
            z = torch.randn((bs, z_dim), device=device)
            labels = torch.full((bs,), class_label, device=device)
            gen_imgs = generator(z, labels)
            synthetic_imgs.append(gen_imgs.cpu())
            synthetic_labels.append(labels.cpu())
            
    synthetic_imgs = torch.cat(synthetic_imgs, dim=0)
    synthetic_labels = torch.cat(synthetic_labels, dim=0)
    return synthetic_imgs, synthetic_labels

def train_classifier(model, dataloader, epochs, device):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss/len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

def evaluate_model(model, dataloader, device, model_name):
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            targets = labels.view(-1).numpy()
            
            probs = model(imgs).cpu().view(-1).numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    acc, rec, prec, f1, auc = calculate_metrics(all_targets, all_preds, all_probs)
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    return {"acc": acc, "rec": rec, "prec": prec, "f1": f1, "auc": auc}

def main():
    parser = argparse.ArgumentParser(description="Train model on BraTS2020 dataset.")
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet', 'vit'], help='Model architecture to use.')
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parameters
    base_dir = r"d:\genai\genai\BraTS2020_training_data"
    metadata_path = os.path.join(base_dir, "content", "data", "meta_data.csv")
    img_size = 64
    batch_size = 32
    z_dim = 100
    
    try:
        # Load Datasets
        print("Loading datasets...")
        train_ds = BraTSDataset(metadata_path, base_dir, split='train', img_size=img_size)
        val_ds = BraTSDataset(metadata_path, base_dir, split='val', img_size=img_size)
        
        # Subsetting to speed up training while maintaining enough data for high accuracy
        train_subset_size = min(10000, len(train_ds))
        val_subset_size = min(2000, len(val_ds))
        subset_train_idx = torch.randperm(len(train_ds))[:train_subset_size]
        subset_val_idx = torch.randperm(len(val_ds))[:val_subset_size]
        train_ds = torch.utils.data.Subset(train_ds, subset_train_idx)
        val_ds = torch.utils.data.Subset(val_ds, subset_val_idx)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # To calculate imbalance, we gather labels
    print("Checking class distribution in training subset...")
    if hasattr(train_ds, 'metadata'):
        targets = train_ds.metadata['target'].values
    else:
        # Fallback for subset
        targets = [train_ds.dataset.metadata.iloc[int(idx)]['target'] for idx in train_ds.indices]
    
    targets = np.array(targets)
    class_counts = pd.Series(targets).value_counts()
    
    print("\n==================================================")
    print("DATASET TARGET VARIABLE & DISTRIBUTION")
    print("==================================================")
    print("The target variable is named 'target'.")
    print("It represents the diagnosis of the Medical Image:")
    print("   1.0 = Tumor Positive Classifier (Presence of anomaly)")
    print("   0.0 = Non-Tumor Classifier (Healthy slice)")
    print(f"\nClass Distribution in Training Subset:\n{class_counts.to_string()}")
    print("==================================================\n")
    
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    diff = int(class_counts[majority_class] - class_counts[minority_class])
    
    print(f"Majority: {majority_class}, Minority: {minority_class}, Difference: {diff}")
    
    # Peek at shape to determine channels
    img, _ = train_ds[0]
    img_channels = img.shape[0]
    print(f"Image channels: {img_channels}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    def get_classifier():
        if args.model == 'cnn':
            return CNNClassifier(img_channels=img_channels, img_size=img_size).to(device)
        elif args.model == 'resnet':
            return ResNetClassifier(img_channels=img_channels, img_size=img_size).to(device)
        elif args.model == 'vit':
            return ViTClassifier(img_channels=img_channels, img_size=img_size).to(device)

    # ---------------------------
    # Step 1: Train Baseline Classifier
    # ---------------------------
    print(f"\n--- Training Baseline {args.model.upper()} ---")
    baseline_classifier = get_classifier()
    train_classifier(baseline_classifier, train_loader, epochs=10, device=device)
    baseline_metrics = evaluate_model(baseline_classifier, val_loader, device, "Baseline (Imbalanced)")
    
    # ---------------------------
    # Step 2: Train cGAN
    # ---------------------------
    generator = Generator(z_dim=z_dim, num_classes=2, img_channels=img_channels, img_size=img_size).to(device)
    discriminator = Discriminator(num_classes=2, img_channels=img_channels, img_size=img_size).to(device)
    
    train_cgan(generator, discriminator, train_loader, epochs=50, z_dim=z_dim, device=device)
    
    # ---------------------------
    # Step 3: Generate Synthetics
    # ---------------------------
    # Generate 'diff' samples for minority class to balance it
    if diff > 0:
        synth_imgs, synth_labels = generate_synthetic_data(generator, diff, minority_class, z_dim, device)
        
        balanced_ds = MergedDataset(train_ds, synth_imgs, synth_labels)
    else:
        balanced_ds = train_ds
        print("Dataset already balanced. No synthetic generation needed.")
        
    balanced_loader = DataLoader(balanced_ds, batch_size=batch_size, shuffle=True)
    
    # ---------------------------
    # Step 4: Train Balanced Classifier
    # ---------------------------
    print(f"\n--- Training Balanced {args.model.upper()} ---")
    balanced_classifier = get_classifier()
    train_classifier(balanced_classifier, balanced_loader, epochs=10, device=device)
    balanced_metrics = evaluate_model(balanced_classifier, val_loader, device, "cGAN Balanced")

    # ---------------------------
    # Output File
    # ---------------------------
    results_path = fr'd:\genai\results_{args.model}.txt'
    with open(results_path, 'w') as f:
        f.write(f"Evaluation Results for {args.model.upper()}\n")
        f.write("==================\n\n")
        f.write(f"Baseline (Imbalanced) {args.model.upper()}:\n")
        for k, v in baseline_metrics.items():
            f.write(f"{k.upper()}: {v:.4f}\n")
            
        f.write(f"\ncGAN Balanced {args.model.upper()}:\n")
        for k, v in balanced_metrics.items():
            f.write(f"{k.upper()}: {v:.4f}\n")
    
    # ---------------------------
    # Save Trained Model
    # ---------------------------
    model_save_path = fr'd:\genai\balanced_{args.model}.pth'
    torch.save(balanced_classifier.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    print(f"\nPipeline execution complete. Results saved to {results_path}")

if __name__ == '__main__':
    main()
