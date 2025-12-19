import torch


class multihead_self_attention(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, attn_pdrop:float=0.0):
        super(multihead_self_attention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads
        self.attn_pdrop = attn_pdrop

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(attn_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q=self.W_q(x) # (B,T,d_model)
        k=self.W_k(x)
        v=self.W_v(x)   
        q=q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1,2) # (B,heads,T,d_k)
        k=k.view(batch, seq_len, self.num_heads, self.d_k).transpose(1,2) # (B,heads,T,d_k)
        v=v.view(batch, seq_len, self.num_heads, self.d_k).transpose(1,2)
        score=(q @ k.transpose(-2,-1))/ (self.d_k ** 0.5) # (B,heads,T,T)
        mask = torch.triu(
            torch.ones(seq_len, seq_len),
            diagonal=1,
        ).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
        score = score.masked_fill(mask, float("-inf"))
        attn=torch.nn.functional.softmax(score, dim=-1) # (B,heads,T,T)
        attn = self.dropout(attn)
        out=attn @ v # (B,heads,T,d_k)
        out=out.transpose(1,2).contiguous().view(batch, seq_len, self.d_model) # (B,T,d_model)
        out=self.W_o(out) # (B,T,d_model)
        return out

        

