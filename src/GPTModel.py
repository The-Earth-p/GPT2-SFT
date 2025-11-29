from LayerNorm import LayerNorm
from DataLoader import create_dataloader_v1
from TransformerBlock import TransformerBlock
import tiktoken
import torch.nn as nn
import torch


GPT_CONFIG_124M={
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "n_layers": 12,
}

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        #Embedding层的前一个参数表示可能出现的不同token数量，而不是具体的长度，这和Linear不一样
        self.tok_emb=nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb=nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb=nn.Dropout(cfg["drop_rate"])

        self.trf_blocks=nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm=LayerNorm(cfg["emb_dim"])
        self.out_head=nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)

    def forward(self,in_idx):
        batch_size,seq_len=in_idx.shape
        tok_embeds=self.tok_emb(in_idx)
        pos_embeds=self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        x=tok_embeds+pos_embeds
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        x=self.final_norm(x)
        logits=self.out_head(x)
        return logits

def text_to_token_ids(text,tokenizer):
    encoded=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)
def token_ids_to_text(token_ids,tokenizer):
    flat=token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate_text_simple(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-context_size:]
        with torch.no_grad():
            logits=model(idx_cond)

        logits=logits[:,-1,:]
        probas=torch.softmax(logits,dim=-1)
        idx_next=torch.argmax(probas,dim=-1,keepdim=True)
        idx=torch.cat((idx,idx_next),dim=1)
    return idx

#计算单个批次的损失
def calc_loss_batch(input_batch,target_batch,model,device):
    input_batch=input_batch.to(device)
    target_batch=target_batch.to(device)
    logits=model(input_batch)
    loss=nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
    return loss

#计算一整个数据加载器的损失，也就是验证集
def calc_loss_loader(data_loader,model,device,num_batches=None):
    total_loss=0
    if len(data_loader)==0:
        return float("nan")
    elif num_batches is None:
        num_batches=len(data_loader)
    else:
        num_batches=min(num_batches,len(data_loader))
    for i, (input_batch,target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss=calc_loss_batch(input_batch,target_batch,model,device)
            total_loss+=loss
        else:
            break
    return total_loss/num_batches

def evaluate_model(model,train_loader,val_loader,device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss=calc_loss_loader(
            train_loader,model,device,num_batches=eval_iter
        )
        val_loss=calc_loss_loader(
            val_loader,model,device,num_batches=eval_iter
        )
    model.train()
    return train_loss,val_loss

def generate_and_print_sample(model,tokenizer,device,start_context):
    model.eval()
    context_size=model.pos_emb.weight.shape[0]
    encoded=text_to_token_ids(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids=generate_text_simple(model=model,idx=encoded,max_new_tokens=50,context_size=context_size)
    decoded_text=token_ids_to_text(token_ids,tokenizer)
    print(decoded_text.replace("\n"," "))
    model.train()

def generate(model,idx,max_new_tokens,context_size,temperature=0.0,top_k=None,eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-context_size:]
        with torch.no_grad():
            logits=model(idx_cond)
        # 这个logits包含的是逐词增加作为输入的多种情况，比如有4个词，最后会输出第一个词的预测、第一个+第二个词的预测等等
        logits=logits[:,-1,:]
        if top_k is not None:
            top_logits,_=torch.topk(logits,top_k)
            min_val=top_logits[:,-1]
            logits=torch.where(
                logits<min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature>0.0:
            logits=logits/temperature
            probs=torch.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
        else:
            idx_next=torch.argmax(logits,dim=-1,keepdim=True)
        if idx_next==eos_id:
            break
        idx=torch.cat((idx,idx_next),dim=1)
    return idx



def train_model_simple(model,train_loader,val_loader,optimizer,device,num_epochs,
                       eval_freq,eval_iter,start_context,tokenizer):
    train_losses,val_losses,track_tokens_seen=[],[],[]
    tokens_seen,global_step=0,-1

    for epoch in range(num_epochs):
        model.train()
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss=calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()
            tokens_seen+=input_batch.numel()
            global_step+=1
            if global_step%eval_freq==0:
                train_loss,val_loss=evaluate_model(
                    model,train_loader,val_loader,device,eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} | Global step {global_step} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f}")
        generate_and_print_sample(
            model,tokenizer,device,start_context
        )
    return train_losses,val_losses,track_tokens_seen


if __name__=="__main__":
    torch.manual_seed(0)
    device = torch.device('cpu')
    model=GPTModel(GPT_CONFIG_124M)
    model.to(device)
    file_path="the-verdict.txt"
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0004,weight_decay=0.1)
    num_epochs=10
    tokenizer=tiktoken.get_encoding("gpt2")
    with open(file_path,"r",encoding="utf-8") as file:
        text_data=file.read()
    train_ratio=0.9
    train_size=int(train_ratio*len(text_data))
    train_data=text_data[:train_size]
    val_data=text_data[train_size:]

    train_loader=create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],#保证样本正好不重合
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader=create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],#保证样本正好不重合
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    #训练
    # train_losses,val_losses,tokens_seen=train_model_simple(
    #     model,train_loader,val_loader,optimizer,device,num_epochs=num_epochs,
    # eval_freq=5,eval_iter=5,start_context="Every effort moves you",tokenizer=tokenizer
    # )

    token_ids=generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you",tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )
    print(f"Output: {token_ids_to_text(token_ids,tokenizer)}")
