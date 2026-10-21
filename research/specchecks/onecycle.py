import torch
p=[torch.nn.Parameter(torch.zeros(1))]
opt=torch.optim.AdamW(p,lr=1e-3,weight_decay=0.1,betas=(0.9,0.999),eps=1e-8)
s=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=1e-3,total_steps=50001,pct_start=0.2,anneal_strategy="linear")
marks={0,1000,5000,9999,10000,20000,30000,40000,50000}
for st in range(50001):
    if st in marks: print(st, f"{opt.param_groups[0]['lr']:.3e}", opt.param_groups[0]['betas'])
    opt.step(); 
    if st<50000: s.step()
