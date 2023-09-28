import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.sqrt_d = d_model**-0.5 # sqrt(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x): return self.embedding(x) * self.sqrt_d # (N,T) -> (N,T,D)

class PositionalEncoding(nn.Module):
    def __init__(self, D: int, T: int, dropout: float) -> None:
        super().__init__()
        # self.D, self.T = D, T # Note needed
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(T, D) # Create a matrix of shape (T, D)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1) # (T, 1)
        Dindx = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D)) # (D / 2)
        pe[:, 0::2] = torch.sin(position * Dindx) # sin(position * (10000 ** (2i / D))
        pe[:, 1::2] = torch.cos(position * Dindx) # cos(position * (10000 ** (2i / D))
        pe = pe.unsqueeze(0) #  (T, D)->(1, T, D) Add a batch dim, to be able to add PE to entire batch
        self.register_buffer('pe', pe) # Register the PE as a buffer, as part of the model

    def forward(self, x): # x (N, T, D) -> (N, T, D)
        x = x + self.pe # you can use x + (self.pe[:, :x.shape[1], :]) or x + self.pe[:x.size(0), :] (N,T,D)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadSelfAttentionBlock(nn.Module):
    """
    Issues: 
    + Manual Parameter Initialization: The weight matrices are initialized with a standard normal distribution. Depending on the model, a more specific initialization strategy might be beneficial.
    + No Bias Terms for Linear Operations: The parameter matrices self.Wqkv and self.Wout don't include bias terms. Depending on your specific use case, omitting these might impact the model's expressiveness.

    def forward(self, z, mask=True): 
        assert len(z.shape) == 3 and z.shape[-1] == self.D f"Input tensor must be 3D andLast dimension must be {self.D}"
        N, T, D = z.shape

    """
    def __init__(self, D: int, H: int, dropout: float=0.1) -> None:
        super().__init__()
        assert D % H == 0, "D is not divisible by H" # Make sure D is divisible by H
        self.d_k = D // H # Dimension of vector seen by each head
        self.D = D # d_model
        self.H = H # heads
        self.sqrtDk = self.d_k**-0.5 # math.sqrt(self.d_k)
        self.Wqkv = nn.parameter.Parameter(torch.randn(D, 3*D)) # torch: in_proj.T 
        self.Wout = nn.parameter.Parameter(torch.randn(D, D))   # Torch: out_proj.T
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, z, mask=None): # z will be used to extract Q,K,V  (N, T, D)
        Q,K,V = torch.chunk(z@self.Wqkv, 3, dim=-1)
        # split/rearange dims to fit Heads   # z.shape[0],z.shape[1] are N, T
        Q,K,V = [a.reshape(z.shape[0],z.shape[1],self.H,self.d_k).swapaxes(1,2) for a in (Q,K,V)]
        attn = F.softmax(Q@K.swapaxes(-1,-2)*self.sqrtDk + mask, dim=-1) if mask else F.softmax(Q@K.swapaxes(-1,-2) * self.sqrtDk, dim=-1)
        if self.dropout: attn = self.dropout(attn)
        return (attn@V).swapaxes(1,2).reshape(z.shape[0],z.shape[1],self.D) @ self.Wout, attn.mean(1)

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, D: int, H: int, dropout: float, block_size: int) -> None:
        super().__init__()
        if D % H != 0:
            raise ValueError("D must be divisible by H")
        
        self.d_k = D // H
        self.D = D
        self.H = H
        self.sqrtDk = self.d_k ** -0.5
        self.Wqkv = nn.Parameter(torch.randn(D, 3 * D))
        nn.init.normal_(self.Wqkv, std=0.01)  # Specific initialization, if needed
        self.Wout = nn.Parameter(torch.randn(D, D))
        nn.init.normal_(self.Wout, std=0.01)  # Specific initialization, if needed
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, z, mask=True):  
        if len(z.shape) != 3 or z.shape[-1] != self.D:
            raise ValueError("Input tensor must be 3D and last dimension must match D")

        N, T, D = z.shape
        Q, K, V = torch.chunk(z @ self.Wqkv, 3, dim=-1)
        Q, K, V = [a.reshape(N, T, self.H, self.d_k).permute(0, 2, 1, 3) for a in (Q, K, V)]

        if mask:
            attn = Q @ K.permute(0, 1, 3, 2) * self.sqrtDk
            attn.masked_fill_(self.tril[:T, :T] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(Q @ K.permute(0, 1, 3, 2) * self.sqrtDk, dim=-1)

        if self.dropout.p > 0:
            attn = self.dropout(attn)

        return (attn @ V).permute(0, 2, 1, 3).reshape(N, T, self.D) @ self.Wout


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
        attn = F.softmax(Q@K.swapaxes(-1,-2)*self.sqrtDk + mask, dim=-1) if mask else F.softmax(Q@K.swapaxes(-1,-2)*self.sqrtDk, dim=-1)
        if self.dropout: attn = self.dropout(attn)
        return (attn@V).swapaxes(1,2).reshape(Zq.shape[0],Zkv.shape[1],self.D) @ self.Wout, attn.mean(1)

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model) )
    def forward(self, x): return self.net(x) # (N,T,D) --> (N, T, d_ff) --> (N,T,D)

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    def forward(self, x) -> None: return torch.log_softmax(self.proj(x), dim = -1) # (N,T,D) --> (N, T, vocab_size)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (N,T,D)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (N,T,D)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout) # why the hell use two different PE, stuppid!!!
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer