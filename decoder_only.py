import torch
import torch.nn as nn
import math
from datasets import load_dataset
from transformers import GPT2Tokenizer
import os
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention and add-norm
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward network and add-norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, dim_ff, num_layers, max_len=5000):
        super(DecoderTransformer, self).__init__()
        self.d_model=d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, dim_ff) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.output_layer(x)



# Load a small subset of the IMDB dataset (e.g., first 500 samples)
dataset = load_dataset("imdb", split="train[:100]")

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to the eos token for compatibility
vocab_size = tokenizer.vocab_size

def preprocess_sentence(sentence, tokenizer):
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
    return torch.tensor(tokens)

# Define Dataset class for the subset
class IMDBTextDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = [preprocess_sentence(text, tokenizer) for text in data["text"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][:-1], self.data[idx][1:]

# Create the subset dataset and DataLoader
train_data = IMDBTextDataset(dataset, tokenizer)

def collate_batch(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=tokenizer.pad_token_id)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=tokenizer.pad_token_id)

    return src_batch, tgt_batch
 

train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=False, collate_fn=collate_batch)


# Initialize model, loss function, and optimizer

d_model = 512
n_heads = 4
dim_ff = 2048
num_layers = 2
max_len = 100


model = DecoderTransformer(vocab_size, d_model, n_heads, dim_ff, num_layers, max_len)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


checkpoint_dir="/home/mcw/shivam2/tenstorrent/checkpoints"

# Training function
def train_small_data(model, data_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for src, tgt in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")
         # Save model checkpoint at the end of each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

# Train on small subset
train_small_data(model, train_loader, criterion, optimizer, num_epochs=10)


