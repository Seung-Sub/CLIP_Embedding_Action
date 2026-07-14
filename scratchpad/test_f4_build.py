import sys, yaml, torch
sys.path.insert(0, "src")
from models.f4 import build_f4_from_cfg
cfg = yaml.safe_load(open("configs/phase2_libero_c1.yaml"))
f4cfg = cfg["module"]["f4"]
# C1 substrate dims: siglip2-large-256 -> latent 1024, 256 patch, text 1024
f4 = build_f4_from_cfg(f4cfg, dense_dim=1024, text_dim=1024, latent_dim=1024,
                       n_base_tokens=5, action_dim=7, n_chunk=16, n_patch=256)
print("BUILD OK")
print("K", f4.K, "bneck", f4.bneck, "zf_dim", f4.zf_dim)
print("alpha", float(f4.alpha), "beta", float(f4.beta))
print("gate=tanh:", "tanh" in open("src/models/f4.py").read())
print("params(M)", sum(p.numel() for p in f4.parameters())/1e6)
# forward smoke: encode + flow + fine
B=4
dF=torch.randn(B,256,1024); txt=torch.randn(B,1024)
zf_tgt=f4.encode(dF,txt); print("encode zf shape", zf_tgt.shape, "init norm(a=0)", float(zf_tgt.norm()))
base=torch.randn(B,5*1024)
zf,lfm=f4.flow_fm_and_sample(base, zf_tgt); print("flow zf", zf.shape, "lfm", float(lfm))
zg=torch.randn(B,1024); zc=torch.randn(B,1024)
r=f4.fine_action(zg,zf,zc); print("fine_action", r.shape, "init contrib(b=0)", float(r.abs().max()))
