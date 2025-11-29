import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


tokenizer=tiktoken.get_encoding("gpt2")
tokenizer.encode("<|endoftext|>",allowed_special={"<|endoftext|>"})

class SpamDataset(Dataset):
    def __init__(self,csv_file,tokenizer,max_length=None,pad_token_id=50256):
        self.data=pd.read_csv(csv_file)
        self.encoded_texts=[
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        if max_length is None:
            self.max_length=self._longest_encoded_length()
        else:
            #编码太长了给他截了
            self.max_length=max_length
            self.encoded_texts=[
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]
        #把所有句子编码长度都填充到最大长度
        self.encoded_texts=[
            encoded_text+[pad_token_id]*(self.max_length-len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    #返回一个样本，DataLoader会自动调用这个方法来遍历每个元素
    def __getitem__(self,index):
        encoded=self.encoded_texts[index]
        label=self.data.iloc[index]["Label"]
        return (torch.tensor(encoded,dtype=torch.long),
                torch.tensor(label,dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length=0
        for encoded_text in self.encoded_texts:
            encoded_length=len(encoded_text)
            if max_length<encoded_length:
                max_length=encoded_length
        return max_length

train_dataset=SpamDataset(csv_file="train.csv",max_length=None,tokenizer=tokenizer)
test_dataset=SpamDataset(csv_file="test.csv",max_length=None,tokenizer=tokenizer)
val_dataset=SpamDataset(csv_file="validation.csv",max_length=None,tokenizer=tokenizer)
num_workers=0
batch_size=256
torch.manual_seed(123)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,
                        num_workers=num_workers,drop_last=True)
val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,
                      num_workers=num_workers,drop_last=False)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,
                       num_workers=num_workers,drop_last=False)

for input_batch,target_batch in train_loader:
    pass
