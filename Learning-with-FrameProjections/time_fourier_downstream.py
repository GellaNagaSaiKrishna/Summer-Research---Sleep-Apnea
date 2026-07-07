import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score
import os

# Import the architecture and dataset classes from your pre-training script
from time_fourier import ResNet1D, SleepEDF_Evaluation_Dataset, CHECKPOINT_DIR, DEVICE, LATENT_DIM, NUM_CLASSES

def evaluate_model():
    print(f"Executing downstream pipeline on device: {DEVICE}")

    # 1. Initialize the Encoder Backbone
    # backbone=True returns latent features instead of classifier output
    eval_encoder = ResNet1D(
        in_channels=1, base_filters=32, kernel_size=5, stride=1, groups=1, 
        n_block=3, n_classes=LATENT_DIM, downsample_gap=2, increasefilter_gap=4, 
        use_do=False, backbone=True, output_dim=LATENT_DIM
    ).to(DEVICE)
    
    # 2. Load the Pretrained Weights
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_time_encoder_fourier.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Cannot find {checkpoint_path}. Did the pre-training script finish successfully?")
        
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Remove keys that cause size mismatch (the pretraining dense layers) if they exist
    keys_to_remove = ['dense.weight', 'dense.bias']
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
    
    # Load filtered state dictionary; strict=False ignores missing keys
    eval_encoder.load_state_dict(state_dict, strict=False)
    
    # FREEZE the encoder weights
    print("\n🔒 Loaded Pre-trained Encoder weights and freezing parameters...")
    for param in eval_encoder.parameters():
        param.requires_grad = False
        
    eval_encoder.eval() 
    
    # 3. Define the New Classification Head
    classifier_head = nn.Linear(LATENT_DIM, NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(classifier_head.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Data Loaders
    train_loader = DataLoader(SleepEDF_Evaluation_Dataset(split="train"), batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(SleepEDF_Evaluation_Dataset(split="val"), batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(SleepEDF_Evaluation_Dataset(split="test"), batch_size=128, shuffle=False, num_workers=2)

    def evaluate_pipeline(loader, is_test=False):
        classifier_head.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for signals, labels in loader:
                _, features = eval_encoder(signals.to(DEVICE))
                preds = torch.argmax(classifier_head(features), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.numpy())
                
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        
        if is_test:
            kappa = cohen_kappa_score(all_targets, all_preds)
            weighted_f1 = f1_score(all_targets, all_preds, average='weighted')
            CLASS_NAMES = ["Wake (W)", "N1 Stage", "N2 Stage", "N3 Stage", "REM"]
            print("\n📊 DETAILED PERFORMANCE BREAKDOWN (PRE-TRAINED BACKBONE + 1-LAYER LINEAR HEAD):")
            print(f"   ➡️ Test Accuracy:    {acc*100:.2f}%")
            print(f"   ➡️ Cohen's Kappa:     {kappa:.4f}")
            print(f"   ➡️ Macro F1-Score:    {macro_f1:.4f}")
            print(f"   ➡️ Weighted F1-Score: {weighted_f1:.4f}")
            print("   ➡️ Stage-Specific F1-Scores:")
            for name, score in zip(CLASS_NAMES, f1_score(all_targets, all_preds, average=None)):
                print(f"       • {name.ljust(12)}: {score:.4f}")
                
        return acc, macro_f1

    # 5. Training the Classifier
    print("\n🏋️ Training Supervised 1-Layer Linear Head...")
    EVAL_EPOCHS = 30
    best_val_f1 = 0.0
    
    for epoch in range(EVAL_EPOCHS):
        classifier_head.train()
        total_loss = 0
        for signals, labels in train_loader:
            optimizer.zero_grad()
            # Extract features without computing gradients for the encoder
            with torch.no_grad():
                _, features = eval_encoder(signals.to(DEVICE))
                
            loss = criterion(classifier_head(features), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validate at the end of the epoch
        val_acc, val_f1 = evaluate_pipeline(val_loader)
        avg_loss = total_loss / len(train_loader)
        print(f"Eval Epoch [{epoch+1:02d}/{EVAL_EPOCHS}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")
        
        # Save best classifier head
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(classifier_head.state_dict(), os.path.join(CHECKPOINT_DIR, "linear_classifier_head_fourier.pth"))

    # 6. Final Deployment Testing on Unseen Participants
    print("\n🔒 Final Deployment Testing on Unseen Participants...")
    classifier_head.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "linear_classifier_head_fourier.pth"), map_location=DEVICE))
    evaluate_pipeline(test_loader, is_test=True)

if __name__ == "__main__":
    evaluate_model()