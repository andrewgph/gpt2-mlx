import json
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path


import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


@dataclass
class ModelArgs:
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int
    n_positions: int
    embd_pdrop: int
    resid_pdrop: int
    attn_pdrop: int


class NewGELU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3.0))))


class CausalSelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.n_embd % args.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(args.n_embd, 3 * args.n_embd)
        
        # output projection
        self.c_proj = nn.Linear(args.n_embd, args.n_embd)
        
        # regularization
        self.attn_dropout = nn.Dropout(args.attn_pdrop)
        self.resid_dropout = nn.Dropout(args.resid_pdrop)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.mask = mx.tril(mx.ones((args.n_positions, args.n_positions))).reshape(1, 1, args.n_positions, args.n_positions)
        self.mask = (self.mask - 1) * float(np.finfo(np.float32).max)
        
        self.n_head = args.n_head
        self.n_embd = args.n_embd

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(3, axis=2)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(k.shape[-1]))
        att = att + self.mask[:,:,:T,:T]
        att = mx.softmax(att, axis=-1)
        att = self.attn_dropout(att)
    
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v 
        # re-assemble all head outputs side by side
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C) 

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class Block(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.ln_1 = nn.LayerNorm(args.n_embd)
        self.attn = CausalSelfAttention(args)
        self.ln_2 = nn.LayerNorm(args.n_embd)
        
        # MLP
        self.c_fc = nn.Linear(args.n_embd, 4 * args.n_embd)
        self.c_proj = nn.Linear(4 * args.n_embd, args.n_embd)
        self.act = NewGELU()
        self.dropout = nn.Dropout(args.resid_pdrop)

    def __call__(self, x: mx.array) -> mx.array:
        x1 = self.ln_1(x)
        x1 = self.attn(x1)
        x = x + x1

        x2 = self.ln_2(x)
        x2 = self.c_fc(x2)
        x2 = self.act(x2)
        x2 = self.c_proj(x2)
        x2 = self.dropout(x2)
        x = x + x2

        return x


class GPT2(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        
        assert args.vocab_size is not None
        assert args.n_positions is not None

        self.n_positions = args.n_positions

        self.wte = nn.Embedding(args.vocab_size, args.n_embd)
        self.wpe = nn.Embedding(args.n_positions, args.n_embd)
        self.drop = nn.Dropout(args.embd_pdrop)
        self.h = [Block(args) for _ in range(args.n_layer)]
        self.ln_f = nn.LayerNorm(args.n_embd)

        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, T = x.shape

        assert T <= self.n_positions, f"sequence of length {T} is larger than number of positions {self.n_positions}"
        pos = mx.arange(0, T, dtype=mx.int64).reshape(1, T)

        # token embeddings of shape (B, T, n_embd)
        tok_emb = self.wte(x)
        # position embeddings of shape (1, T, n_embd)
        pos_emb = self.wpe(pos) 
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.h:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, x, temperature=1.0):
        assert temperature >= 0, "temperature must be greater or equal to 0"
        while True:
            # if the sequence context is growing too long we must crop it at n_positions
            x = x if x.shape[1] <= self.n_positions else x[:, -self.n_positions:]
            
            # forward the model to get the logits for the index in the sequence
            logits = self(x)

            # Only care about logits for the last position to generate the next token
            logits = logits[:, -1, :]

            if temperature == 0:
                x_next = mx.argmax(logits, axis=-1)
            else:
                # Sample using gumbel-max trick
                x_next = mx.random.categorical(logits * (1 / temperature))

            # append sampled index to the running sequence and continue
            x = mx.concatenate([x, x_next.reshape(x.shape[0], 1)], axis=1)

            yield x_next


def validate_config(config):
    assert config["n_positions"] == 1024, "expected number of positions to be 1024"

    # Get the fields of the ModelArgs dataclass
    model_args_fields = ModelArgs.__annotations__.keys()
    # Delete keys in config that are not in ModelArgs
    keys_to_delete = [key for key in config if key not in model_args_fields]
    for key in keys_to_delete:
        del config[key]

    return config


def load_model(model_path):
    model_path = Path(model_path)
    weights = mx.load(str(model_path / "mlx_weights.npz"))

    with open(model_path / "config.json", "r") as f:
        config = validate_config(json.loads(f.read()))
    model = GPT2(ModelArgs(**config))
    
    # Check that weights file contains all expected layer weights
    # Ignore the attention masks as they are defined manually
    expected_keys = set([x[0] for x in tree_flatten(model.parameters()) if not x[0].endswith("attn.mask")])
    actual_keys = set(weights.keys())
    assert expected_keys == actual_keys, "expected keys to match"
    
    model.update(tree_unflatten(list(weights.items())))
    return model