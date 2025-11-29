import torch
import time
from DataLoader_FT import train_loader, val_loader, test_loader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import tiktoken
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == target_batch).sum().item()
            num_examples += target_batch.size(0)
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model):
    logits = model(input_batch)[:, -1, :]
    loss = nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            loss = calc_loss_batch(input_batch, target_batch, model)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        start_time = time.time()
        eval_num=0
        temp_train_loss,temp_val_loss=0,0
        model.train()
        for input_batch, target_batch in train_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model)
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1
            train_loss, val_loss = evaluate_model(
                model, train_loader, val_loader, device, eval_iter
            )
            eval_num+=1
            temp_train_loss+=train_loss
            temp_val_loss+=val_loss
                    
        train_losses.append(temp_train_loss/eval_num)
        val_losses.append(temp_val_loss/eval_num)
        train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        end_time = time.time()
        print(
            f"Epoch {epoch + 1:>4}/{num_epochs} | Train Loss: {temp_train_loss/eval_num:.4f} | Val Loss: {temp_val_loss/eval_num:.4f} "
            f"| Train ACC: {train_acc:.4f} | Val ACC: {val_acc:.4f} | Time: {end_time - start_time:.2f}s"
        )
    return train_losses, val_losses, train_accs, val_accs, examples_seen


def classify_review(text, model, tokenizer, device, max_length=1024, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    input_ids = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
        print(logits)
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"

def plot_values(x, train_values, val_values, label="metric"):

    # 创建 results 文件夹
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 绘图
    plt.figure(figsize=(8,5))
    plt.plot(x.numpy(), train_values, label=f"Train {label}", marker='o')
    plt.plot(x.numpy(), val_values, label=f"Val {label}", marker='o')
    plt.xticks(range(1,len(x)+1,40))
    plt.xlabel("Epochs")
    plt.ylabel(label.capitalize())
    plt.title(f"{label.capitalize()} over Epochs")
    plt.legend()
    plt.grid(False)

    # 保存图片
    save_path = os.path.join(results_dir, f"{label}_over_epochs.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"{label} figure saved to: {save_path}")


def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = tiktoken.get_encoding("gpt2")
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves you"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.1,
        "qkv_bias": True,
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_heads": 12, "n_layers": 12}
    }
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.to(device)
    
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(INPUT_PROMPT, tokenizer).to(device),
        max_new_tokens=25,
        context_size=BASE_CONFIG["context_length"],
        top_k=50,
        temperature=1.5,
    )

    # 冻结全部参数，仅微调
    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)
    num_classes = 2
    model.out_head = nn.Linear(
        in_features=BASE_CONFIG["emb_dim"], out_features=num_classes
    ).to(device)

    # 解冻最后一层 Transformer + LayerNorm
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.001)
    num_epochs = 200

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq=50, eval_iter=5
    )

    epochs_tensor = torch.arange(1, num_epochs+1)
    # 绘制 Loss
    plot_values(epochs_tensor, train_losses, val_losses, label="loss")
    # 绘制 Accuracy
    plot_values(epochs_tensor, train_accs, val_accs, label="accuracy")

    example1="That would be great. We'll be at the Guild."#not spam
    example2="Free 1st week entry 2 TEXTPOD 4 a chance 2 win 40GB iPod or £250 cash every wk."#spam
    example3="Alright, we're all set here, text the man"#not
    example4="My friends use to call the same."#not
    example5="Thanks for your ringtone order, reference number X49. Your mobile will be charged 4.50. Should your tone not arrive please call customer services 09065989182. From: [colour=red]text[/colour]TXTstar"#spam
    result1=classify_review(example1, model, tokenizer, device)
    result2=classify_review(example2, model, tokenizer, device)
    result3=classify_review(example3, model, tokenizer, device)
    result4=classify_review(example4, model, tokenizer, device)
    result5=classify_review(example5, model, tokenizer, device)
    print(f"{example1} is {result1}")
    print(f"{example2} is {result2}")
    print(f"{example3} is {result3}")
    print(f"{example4} is {result4}")
    print(f"{example5} is {result5}")
    
    #torch.save(model.state_dict(), "spam_classifier.pth")#only spam
#    torch.save(model.state_dict(), "1.pth")#onlu not spam
    torch.save(model.state_dict(), "best.pth")
    print("Model saved to spam_classifier.pth")


    model.eval()  # 切换到评估模式
    # 2. 遍历测试集，收集预测结果和真实标签
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for input_batch, target_batch in test_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch)[:, -1, :]
            predictions = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)[:, 1]  # spam 的概率

            y_true.extend(target_batch.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
    # 3. 计算指标
    accuracy = sum([p==t for p,t in zip(y_pred, y_true)]) / len(y_true)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    print("\n=== Test Set Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"AUC: {auc:.4f}")



if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from GPTModel import GPTModel, generate, text_to_token_ids, token_ids_to_text
    from GPTDownload import download_and_load_gpt2
    from LoadWeight import load_weights_into_gpt
    
    main()
