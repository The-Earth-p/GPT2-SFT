import torch
import os
import sys
from Model import classify_review
import tiktoken

# ====== 配置 ======
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#MODEL_PATH = "spam_classifier.pth"
MODEL_PATH = "best.pth"
CHOOSE_MODEL = "gpt2-small (124M)"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_heads": 12, "n_layers": 12}
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
num_classes = 2  # 分类任务类别数

# ====== 主程序 ======
def main():
    print(f"Using device: {device}")

    # 加载 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 下载并加载 GPT-2 权重
    #model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    #settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    # 创建模型结构
    model = GPTModel(BASE_CONFIG)
    #load_weights_into_gpt(model, params)

    # 添加分类头
    model.out_head = torch.nn.Linear(BASE_CONFIG["emb_dim"], num_classes)

    # 加载训练好的权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # ===== 单条文本预测 =====
    sample_text = input("请输入要预测的文本: ")
    prediction = classify_review(sample_text, model, tokenizer, device)
    print(f"预测结果: {prediction}")


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from GPTModel import GPTModel, generate, text_to_token_ids, token_ids_to_text
    from GPTDownload import download_and_load_gpt2
    from LoadWeight import load_weights_into_gpt
    main()
