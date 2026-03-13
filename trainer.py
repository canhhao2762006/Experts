import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    DEVICE, BATCH_SIZE, MAX_EPOCHS,
    USE_FOCAL_LOSS, FOCAL_GAMMA, LABEL_SMOOTHING,
    WEIGHT_DECAY, LR, USE_ONECYCLE, MAX_LR,
    USE_AMP, USE_EMA_WEIGHTS, EMA_DECAY, EARLY_STOPPING_PATIENCE,
)
from utils import FocalLoss, ModelEMA
from model import CNNBiLSTMTransformer


class TorchSequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def compute_class_weights(y: np.ndarray):
    counts = np.bincount(y, minlength=3).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def train_model(x_train, y_train, x_valid, y_valid, input_size: int):
    train_loader = DataLoader(
        TorchSequenceDataset(x_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
    )
    valid_loader = DataLoader(
        TorchSequenceDataset(x_valid, y_valid),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    )

    model = CNNBiLSTMTransformer(input_size=input_size).to(DEVICE)
    class_weights = compute_class_weights(y_train)

    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = None
    if USE_ONECYCLE:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=MAX_LR,
            steps_per_epoch=max(1, len(train_loader)), epochs=MAX_EPOCHS,
        )

    # ── FIX: Updated PyTorch AMP API ──
    scaler_amp = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA_WEIGHTS else None

    best_val_loss = float("inf")
    best_state = None
    patience_left = EARLY_STOPPING_PATIENCE

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

            train_loss += loss.item() * len(xb)
            train_count += len(xb)

        train_loss /= max(train_count, 1)

        # ── FIX: Apply EMA BEFORE validation & save best state with EMA weights ──
        model.eval()
        if ema is not None:
            ema.apply_to(model)

        valid_loss = 0.0
        valid_count = 0

        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                valid_loss += loss.item() * len(xb)
                valid_count += len(xb)

        valid_loss /= max(valid_count, 1)

        # ── FIX: Save best state WHILE EMA is applied (= EMA weights) ──
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = EARLY_STOPPING_PATIENCE
        else:
            patience_left -= 1

        # Restore original weights for continued training
        if ema is not None:
            ema.restore(model)

        print(f"epoch={epoch+1}/{MAX_EPOCHS} train_loss={train_loss:.5f} valid_loss={valid_loss:.5f}")

        if patience_left <= 0:
            print("Early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_proba(model, x, batch_size: int = 1024):
    loader = DataLoader(
        TorchSequenceDataset(x, np.zeros(len(x), dtype=np.int64)),
        batch_size=batch_size, shuffle=False,
    )

    model.eval()
    probs_all = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(probs_all, axis=0)


def evaluate(model, x, y):
    probs = predict_proba(model, x)
    y_pred = probs.argmax(axis=1)
    return {
        "report_text": classification_report(y, y_pred, digits=4),
        "report": classification_report(y, y_pred, digits=4, output_dict=True),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }
