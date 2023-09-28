import torch
import torch.nn as nn

"""
This version uses abreviation of variable names rather than descriptive long ones.
+ D: d_model
+ T: Sequence length (aka max_seq_length)
+ PositionalEncoding: PositionalEmbedding
"""
    
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.sqrt_d = d_model**-0.5 # sqrt(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x): return self.embedding(x) * self.sqrt_d # (N,T) -> (N,T,D)

class PositionalEncoding(nn.Module):
    def __init__(self, D: int, T: int, dropout: float) -> None:
        super().__init__()
        # self.D, self.T = D, T # Not needed
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(T, D) # Create a matrix of shape (T, D)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1) # (T, 1)
        Dindx = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D)) # (D / 2)
        pe[:, 0::2] = torch.sin(position * Dindx) # sin(position * (10000 ** (2i / D))
        pe[:, 1::2] = torch.cos(position * Dindx) # cos(position * (10000 ** (2i / D))
        pe = pe.unsqueeze(0) #  (T, D)->(1, T, D) Add a batch dim, to be able to add PE to entire batch
        self.register_buffer('pe', pe) # Register the PE as a buffer, as part of the model

    def forward(self, x): # x (Batch, seq_len, d_model)
        # x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # Not sure why he registered the buffer here.
        x = x + self.pe # or `x + (self.pe[:, :x.shape[1], :])` or `x + self.pe[:x.size(0), :]` (N,T,D)
        return self.dropout(x)

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, D: int, H: int, dropout: float) -> None:
        super().__init__()
        assert D % H == 0, "D is not divisible by H" # Make sure D is divisible by H
        self.d_k = D // H # Dimension of vector seen by each head
        self.D, self.H = D, H # d_model, heads
        self.sqrtDk = self.d_k**-0.5 # math.sqrt(self.d_k)
        self.Wqkv = nn.parameter.Parameter(torch.randn(D, 3*D)) # torch: in_proj.T 
        self.Wout = nn.parameter.Parameter(torch.randn(D, D))   # Torch: out_proj.T
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, z, mask): # z will be used to extract Q,K,V  (N, T, D)
        """ The mask in the encoder is used to mask the padded tokens, so it's must not an option """
        N, T, D = z.shape
        Q,K,V = torch.chunk(z@self.Wqkv, 3, dim=-1)
        # split/rearange dims to fit Heads   # z.shape[0],z.shape[1] are N, T
        Q,K,V = [a.reshape(N, T,self.H,self.d_k).swapaxes(1,2) for a in (Q,K,V)]
        attn = F.softmax(Q@K.swapaxes(-1,-2)*self.sqrtDk + mask, dim=-1)
        attn = self.dropout(attn)
        return (attn@V).swapaxes(1,2).reshape(N, T,self.D) @ self.Wout, attn.mean(1)

class MultiHeadCrossAttentionBlock(nn.Module):
    def __init__(self, D: int, H: int, dropout: float) -> None:
        super().__init__()
        assert D % H == 0, "D is not divisible by H" # Make sure D is divisible by H
        self.d_k = D // H # Dimension of vector seen by each head
        self.D = D # d_model
        self.H = H # heads
        self.sqrtDk = self.d_k**-0.5 # math.sqrt(self.d_k)
        self.Wqkv = nn.parameter.Parameter(torch.randn(D, 3*D)) # torch: in_proj.T 
        self.Wout = nn.parameter.Parameter(torch.randn(D, D))   # Torch: out_proj.T
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Zq, Zkv, mask=None): # Zq will be used to extract Q while Zkv K & V  all have shape(N, T, D) 
        Wq, Wk, Wv = torch.chunk(self.Wqkv, 3, dim=-1)
        Q, K,V = Zq@Wq, Zkv@Wk, Zkv@Wv
        # split/rearange dims to fit Heads   # z.shape[0],z.shape[1] are N, T
        Q,K,V = [a.reshape(Zq.shape[0],Zq.shape[1],self.H,self.d_k).swapaxes(1,2) for a in (Q,K,V)]
        attn = F.softmax(Q@K.swapaxes(-1,-2)*self.sqrtDk + mask, dim=-1)
        attn = self.dropout(attn)
        return (attn@V).swapaxes(1,2).reshape(Zq.shape[0],Zkv.shape[1],self.D) @ self.Wout, attn.mean(1)

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model) )
    def forward(self, x): return self.net(x) # (N,T,D) --> (N, T, d_ff) --> (N,T,D)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))


class EncoderBlock(nn.Module):
    def __init__(self, D: int, H: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = Residual(PreNorm(D, MultiHeadSelfAttentionBlock(D, H, dropout)))
        self.ff = Residual(PreNorm(D, FeedForwardBlock(D, dropout)))

    def forward(self, x, mask):
        return self.ff(self.attn(x, mask))

class DecoderBlock(nn.Module):
    def __init__(self, D: int, H: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = Residual(PreNorm(D, MultiHeadSelfAttentionBlock(D, H, dropout)))
        self.cross_attn = Residual(PreNorm(D, MultiHeadCrossAttentionBlock(D, H, dropout)))
        self.ff = Residual(PreNorm(D, FeedForwardBlock(D, dropout)))

    def forward(self, x, e, mask, cross_mask):
        return self.ff(self.cross_attn(self.attn(x, mask), e, cross_mask))

class Transformer(nn.Module):
    def __init__(self, D: int, H: int, d_ff: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_embedding = InputEmbeddings(D, 256)
        self.positional_encoding = PositionalEncoding(D, 1024, dropout)
        self.encoder = nn.Sequential(*[EncoderBlock(D, H, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.Sequential(*[DecoderBlock(D, H, d_ff, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(D, 256)

    def forward(self, x, e, mask, cross_mask):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        e = self.input_embedding(e)
        e = self.positional_encoding(e)
        x = self.encoder(x, mask)
        x = self.decoder(x, e, mask, cross_mask)
        return self.output_layer(x)

def get_model(D: int, H: int, d_ff: int, num_layers: int, dropout: float) -> nn.Module:
    return Transformer(D, H, d_ff, num_layers, dropout)

if __name__ == '__main__':
    from torchsummaryX import summary
    model = get_model(512, 8, 2048, 6, 0.1)
    summary(model, (256, 1024), device='cpu')
    # print(model)
