# Layers.py

class LayerNorm(nn.Module):
  def __init__(self, dim, eps=1e-5, elementwise_affine=True): # we should support device
    super(LayerNorm, self).__init__()
    self.eps = eps
    self.elementwise_affine = elementwise_affine
    if elementwise_affine: self.weight, self.bias = nn.Parameter(torch.ones(dim)), nn.Parameter(torch.zeros(dim))
    else:  self.weight, self.bias = None, None

  def forward(self, x):
    xmean, xvar = x.mean(-1, keepdim=True), x.var(-1, keepdim=True, unbiased=False)
    x_hat = (x - xmean) / torch.sqrt(xvar + self.eps)
    if self.weight is not None: x_hat = self.weight * x_hat + self.bias
    return x_hat

class MultiHeadCrossAttentionBlock(nn.Module):
    """ You can with this if you don't like the previous, to reduce the number of matmul"""
    def __init__(self, D: int, h: int, dropout: float) -> None:
        super().__init__()
        assert D % h == 0, "D is not divisible by h" # Make sure D is divisible by h
        self.d_k = D // h # Dimension of vector seen by each head
        self.sqrtDk = sqrt(self.d_k)
        self.Wkv = nn.parameter.Parameter(torch.randn(D, 2*D))
        self.Wq = nn.parameter.Parameter(torch.randn(D, D))
        self.Wout = nn.parameter.Parameter(torch.randn(D, D))   # Torch: out_proj.T
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,Zq,Zkv,mask): # Zq the decoder attention,Zkv the encoder attention
        K,V = torch.chunk(Zkv@self.Wkv, 2, dim=-1)
        Q = Zq@self.Wq
        Q,K,V = [a.reshape(N,T,heads,self.d_k).swapaxes(1,2) for a in (Q,K,V)] # split/rearange dims to fit h heads
        attn = F.softmax(Q@K.swapaxes(-1,-2)/self.sqrtDk + mask, dim=-1)
        if self.dropout: attn = self.dropout(attn)
        return (attn@V).swapaxes(1,2).reshape(N,T,d) @ self.Wout, attn.mean(1)
