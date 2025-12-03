import os
import re
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from models import create_model


# --------------------- 设备 ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# --------------------- 数据集 ---------------------
class TextDataset(Dataset):
    def __init__(self, csv_file, vocab=None, max_length=50):
        self.data = pd.read_csv(csv_file)
        self.texts = self.data["Text"].astype(str).apply(self.preprocess)
        self.labels = self.data["Label"].values

        if vocab is None:
            self.vocab = self.build_vocab(self.texts)
        else:
            self.vocab = vocab

        self.max_length = max_length

    @staticmethod
    def preprocess(text):
        text = text.lower()
        return re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)

    @staticmethod
    def build_vocab(texts):
        counter = Counter()
        for t in texts:
            counter.update(t.split())

        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word in counter:
            vocab[word] = len(vocab)

        return vocab

    def encode(self, text):
        tokens = text.split()[:self.max_length]
        ids = [self.vocab.get(t, 1) for t in tokens]
        ids += [0] * (self.max_length - len(ids))
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return (
            torch.tensor(self.encode(self.texts[i]), dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.float),
        )


# --------------------- 工具函数 ---------------------
def calc_loss(model, batch_x, batch_y):
    out = model(batch_x)
    return nn.BCELoss()(out, batch_y)


def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = (model(x) > 0.5).long()
            correct += (pred == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


# --------------------- 单模型绘图 ---------------------
def plot_single_model(result_dict, save_dir, model_name):
    os.makedirs(save_dir, exist_ok=True)

    # ---- Loss ----
    plt.figure(figsize=(8, 5))
    plt.plot(result_dict["train_loss"], label="Train Loss")
    plt.plot(result_dict["val_loss"], "--", label="Val Loss")
    plt.legend()
    plt.title(f"{model_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.savefig(os.path.join(save_dir, "loss.pdf"))
    plt.close()

    # ---- Accuracy ----
    plt.figure(figsize=(8, 5))
    plt.plot(result_dict["train_acc"], label="Train Accuracy")
    plt.plot(result_dict["val_acc"], "--", label="Val Accuracy")
    plt.legend()
    plt.title(f"{model_name} Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_dir, "accuracy.png"))
    plt.savefig(os.path.join(save_dir, "accuracy.pdf"))
    plt.close()


# --------------------- 主训练 ---------------------
def main():
    BATCH = 64
    EPOCHS = 40
    LR = 1e-3
    MAX_LEN = 50

    model_list = ["MLP", "RNN", "LSTM", "CNN"]

    print("Loading dataset...")
    train_ds = TextDataset("data/train.csv", max_length=MAX_LEN)
    val_ds = TextDataset("data/validation.csv", vocab=train_ds.vocab, max_length=MAX_LEN)
    test_ds = TextDataset("data/test.csv", vocab=train_ds.vocab, max_length=MAX_LEN)

    train_loader = DataLoader(train_ds, BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH)
    test_loader = DataLoader(test_ds, BATCH)

    os.makedirs("results", exist_ok=True)

    for model_type in model_list:
        print("\n" + "=" * 45)
        print(f"Training {model_type}")
        print("=" * 45)

        save_dir = f"results/{model_type}"
        os.makedirs(save_dir, exist_ok=True)

        log_file = open(os.path.join(save_dir, "training.log"), "w")

        model = create_model(model_type, len(train_ds.vocab)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

        # ---- 学习率衰减器 ----
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )

        train_loss_list, val_loss_list = [], []
        train_acc_list, val_acc_list = [], []

        start_time = time.time()

        # ---- Early Stopping 初始化 ----
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0
        best_model_path = os.path.join(save_dir, "best_model.pth")

        for epoch in range(EPOCHS):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                loss = calc_loss(model, x, y)
                optimizer.zero_grad()
                loss.backward()

                # ---- 梯度裁剪 ----
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

                optimizer.step()

            # ---- 评估 ----
            model.eval()
            with torch.no_grad():
                train_loss = sum(calc_loss(model, x.to(device), y.to(device)).item()
                                 for x, y in train_loader) / len(train_loader)
                val_loss = sum(calc_loss(model, x.to(device), y.to(device)).item()
                               for x, y in val_loader) / len(val_loader)

            train_acc = accuracy(model, train_loader)
            val_acc = accuracy(model, val_loader)

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)

            # ---- 更新学习率 ----
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            msg = (f"[{model_type}] Epoch {epoch + 1}/{EPOCHS} | "
                   f"Train Acc={train_acc:.3f} | Val Acc={val_acc:.3f} | "
                   f"LR={current_lr:.6f}")
            print(msg)
            log_file.write(msg + "\n")

            # ---- Early Stopping ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    log_file.write(f"Early stopping at epoch {epoch + 1}\n")
                    break

        # ---- 测试集准确率 ----
        test_acc = accuracy(model, test_loader)
        msg = f"Test Accuracy: {test_acc:.3f}"
        print(msg)
        log_file.write(msg + "\n")

        duration_min = (time.time() - start_time) / 60
        msg = f"Training Time: {duration_min:.2f} min"
        print(msg)
        log_file.write(msg + "\n")

        log_file.close()

        # 保存模型
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

        # 绘制曲线
        plot_single_model(
            {
                "train_loss": train_loss_list,
                "val_loss": val_loss_list,
                "train_acc": train_acc_list,
                "val_acc": val_acc_list,
                "test_acc": test_acc,
            },
            save_dir,
            model_type
        )


if __name__ == "__main__":
    main()
