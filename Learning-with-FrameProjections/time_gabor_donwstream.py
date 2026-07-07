import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
# Ensure these imports match your file structure
from time_gabor import ResNet1D, SleepEDF_Evaluation_Dataset, CHECKPOINT_DIR, DEVICE, LATENT_DIM, NUM_CLASSES

def evaluate_model():
    # 1. Initialize the Encoder Backbone
    # backbone=True returns latent features instead of classifier output
    encoder = ResNet1D(1, 32, 5, 1, 1, 3, NUM_CLASSES, backbone=True, output_dim=LATENT_DIM).to(DEVICE)
    
    # 2. Load the Pretrained Weights
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_time_encoder.pth")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Remove keys that cause size mismatch (the pretraining classifier layers)
    keys_to_remove = ['dense.weight', 'dense.bias']
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
    
    # Load filtered state dictionary; strict=False ignores missing keys (the ones we deleted)
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval() 
    
    # 3. Define the New Classification Head
    classifier = nn.Linear(LATENT_DIM, NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Data Loaders
    train_loader = DataLoader(SleepEDF_Evaluation_Dataset(split="train"), batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(SleepEDF_Evaluation_Dataset(split="test"), batch_size=128, shuffle=False, num_workers=2)

    # 5. Training the Classifier
    print("Training downstream classifier...")
    for epoch in range(30):
        classifier.train()
        for signals, labels in train_loader:
            optimizer.zero_grad()
            # Only use encoder for feature extraction; gradients are not computed for it
            with torch.no_grad():
                _, feat = encoder(signals.to(DEVICE))
            loss = criterion(classifier(feat), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/30 complete.")

    # 6. Evaluation on Test Set
    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, labels in test_loader:
            _, feat = encoder(signals.to(DEVICE))
            preds = torch.argmax(classifier(feat), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Print standard performance metrics for sleep staging
    print(classification_report(all_labels, all_preds, target_names=["W", "S1", "S2", "S3/4", "R"]))

if __name__ == "__main__":
    evaluate_model()