import torch
import torch.nn as nn
import math
from datasets import load_dataset
from transformers import GPT2Tokenizer
import os
from tqdm import tqdm

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
    

from transformers import GPT2Tokenizer
import torch

d_model = 512
n_heads = 4
dim_ff = 2048
num_layers = 2
max_len = 100


# Load the tokenizer from the local directory
tokenizer = GPT2Tokenizer.from_pretrained("/home/mcw/shivam2/tenstorrent/gpt2_tokenizer")
tokenizer.pad_token = tokenizer.eos_token  # Ensure the pad token is set if needed
vocab_size = tokenizer.vocab_size
# Initialize the model with the same architecture
model = DecoderTransformer(vocab_size, d_model, n_heads, dim_ff, num_layers, max_len)

# Load the model state
model.load_state_dict(torch.load("/home/mcw/shivam2/tenstorrent/checkpoints/model_epoch_7.pt"))


def inference_with_prompt(model, prompt, max_len, tokenizer):
    # Tokenize the prompt
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    generated = [tokenizer.bos_token_id] + prompt_tokens  # Start with BOS + prompt tokens

    model.eval()
    with torch.no_grad():
        for _ in range(max_len - len(generated)):
            inputs = torch.tensor(generated).unsqueeze(0)  # Shape (1, sequence length)
            logits = model(inputs)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()  # Get the predicted next token
            generated.append(next_token)

            # Stop if EOS token is generated
            if next_token == tokenizer.eos_token_id:
                break

    # Decode the tokens back to a string
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated_text


# Define a prompt
prompt = "spider man is "

# Generate text with the prompt
output_sentence = inference_with_prompt(model, prompt, max_len=50, tokenizer=tokenizer)
print("Generated Sentence:", output_sentence)

