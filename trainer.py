"""
Helper training loops for Machine Learning and Deep Learning models.
Supports class-weighted cross-entropy loss to handle imbalance.
"""
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from config import DEVICE, BATCH_SIZE, NUM_EPOCHS, PATIENCE, LR, MODELS_DIR


def train_sklearn_model(model, X_train, y_train, X_test):
    """Train any scikit-learn model and output test predictions."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, model


def get_class_weights(y_train):
    """Compute class weights inversely proportional to class frequencies."""
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    weights = total / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_pytorch_model(model, train_data, val_data, model_name="pytorch_model"):
    """
    General training loop for PyTorch models.
    train_data: TensorDataset (features, labels) or (features, lengths, labels)
    val_data: TensorDataset (features, labels) or (features, lengths, labels)
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Extract labels to compute class weights for imbalance
    y_train = train_data.tensors[-1].numpy()
    class_weights = get_class_weights(y_train).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Unpack depending on dataset features (2 inputs or 3 inputs)
            if len(batch) == 3:
                x_batch, lens_batch, y_batch = batch
                x_batch, lens_batch, y_batch = x_batch.to(DEVICE), lens_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(x_batch, lens_batch)
            else:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(x_batch)
                
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x_batch, lens_batch, y_batch = batch
                    x_batch, lens_batch, y_batch = x_batch.to(DEVICE), lens_batch.to(DEVICE), y_batch.to(DEVICE)
                    outputs = model(x_batch, lens_batch)
                else:
                    x_batch, y_batch = batch
                    x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                    outputs = model(x_batch)
                    
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        # Early stopping and model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_weights = model.state_dict().copy()
            torch.save(best_weights, MODELS_DIR / f"{model_name}_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch+1}!")
                break
                
    # Load best weights
    model.load_state_dict(best_weights)
    return model


def predict_pytorch_model(model, test_data):
    """Generate predictions using trained PyTorch model."""
    model.eval()
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    all_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x_batch, lens_batch, _ = batch
                outputs = model(x_batch.to(DEVICE), lens_batch.to(DEVICE))
            else:
                x_batch, _ = batch
                outputs = model(x_batch.to(DEVICE))
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            
    return np.array(all_preds)