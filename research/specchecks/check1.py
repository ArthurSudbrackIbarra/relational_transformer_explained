import math, torch, torch.nn.functional as F
from torch import nn
# 1. RMSNorm default eps
n = nn.RMSNorm(4)
x32 = torch.tensor([[1e-4, 0., 0., 0.]])
print("rmsnorm fp32", n(x32))
nb = nn.RMSNorm(4).to(torch.bfloat16)
print("rmsnorm bf16", nb(x32.bfloat16()))
z = torch.zeros(1,4)
print("rmsnorm(0)", n(z), nb(z.bfloat16()))
# 2. huber default delta
a = torch.tensor([3.0]); b = torch.tensor([0.0])
print("huber(3,0)", F.huber_loss(a,b), "expected delta=1:", 1*(3-0.5))
# 3. flex attention: fully masked row and scale
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
B,H,S,D = 1,2,8,4
torch.manual_seed(0)
q = torch.randn(B,H,S,D); k = torch.randn(B,H,S,D); v = torch.randn(B,H,S,D)
mask = torch.zeros(B,S,S, dtype=torch.bool)
mask[0,0,:3] = True; mask[0,1,1] = True  # row 2..7 fully masked
def mod(b,h,qi,ki): return mask[b,qi,ki]
bm = create_block_mask(mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device="cpu")
try:
    out = flex_attention(q,k,v,block_mask=bm)
    s = (q @ k.transpose(-1,-2)) / math.sqrt(D)
    s = s.masked_fill(~mask[:,None], float("-inf"))
    ref = torch.nan_to_num(torch.softmax(s,-1)) @ v
    print("flex row0 vs ref (1/sqrt(D))", (out[0,:,0]-ref[0,:,0]).abs().max().item())
    print("flex row1 equals v[1]:", (out[0,:,1]-v[0,:,1]).abs().max().item())
    print("flex fully masked rows abs sum:", out[0,:,2:].abs().sum().item())
except Exception as e:
    print("flex error", repr(e)[:300])
# 4. bf16 rounding demo
for val in [1.2345, -0.5, 1.5, 0.3333, 2.718]:
    print(val, torch.tensor(val).bfloat16().item())
# 5. BCE with logits
print("bce(0 logit, 1)", F.binary_cross_entropy_with_logits(torch.tensor([0.]), torch.tensor([1.])).item(), math.log(2))
