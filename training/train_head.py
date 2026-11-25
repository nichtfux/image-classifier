"""Learn appended outputs while preserving every old network parameter exactly.

Cache the fixed backbone's features once. All real and synthetic training rows
then train a 29-output head, optionally with appended nonlinear features. The
original features and 270 outputs remain fixed and cannot use the new features.
No raw drawings are displayed or sent anywhere.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from categories import CATEGORIES
from train import Net, extend_from_checkpoint

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE.parent / "data", HERE / "out"


def features(net, x, include_fc=True):
    x = F.max_pool2d(F.relu(net.b1(net.c1(x))), 2)
    x = F.max_pool2d(F.relu(net.b2(net.c2(x))), 2)
    x = F.max_pool2d(F.relu(net.b3(net.c3(x))), 2)
    x = x.flatten(1)
    return F.relu(net.f1(x)) if include_fc else x


class ExtraHead(nn.Module):
    """New features for custom labels; the old feature/output path stays fixed."""
    def __init__(self, inputs, old_fc, extra):
        super().__init__()
        self.extra = nn.Linear(inputs, extra)
        self.output = nn.Linear(old_fc + extra, 29)

    def forward(self, x, old_features):
        return self.output(torch.cat([old_features, F.relu(self.extra(x))], 1))


def assemble(initial, labels, head, widths, size=28):
    """Embed the frozen model plus appended features into the same export schema."""
    old = initial["state_dict"]
    old_fc, old_classes = old["f1.weight"].shape[0], len(initial["categories"])
    extra = head.extra.out_features if isinstance(head, ExtraHead) else 0
    expanded = Net(widths, old_fc + extra, len(labels), size).to(old["f1.weight"].device)
    if not extra:
        extend_from_checkpoint(expanded, initial, labels)
    else:
        state = expanded.state_dict()
        for key, value in old.items():
            if not key.startswith(("f1.", "f2.")):
                state[key].copy_(value)
        state["f1.weight"][:old_fc].copy_(old["f1.weight"])
        state["f1.bias"][:old_fc].copy_(old["f1.bias"])
        state["f1.weight"][old_fc:].copy_(head.extra.weight)
        state["f1.bias"][old_fc:].copy_(head.extra.bias)
        state["f2.weight"].zero_()
        state["f2.weight"][:old_classes,:old_fc].copy_(old["f2.weight"])
        state["f2.bias"][:old_classes].copy_(old["f2.bias"])
        expanded.load_state_dict(state)
    output = head.output if extra else head
    with torch.no_grad():
        expanded.f2.weight[old_classes:].copy_(output.weight)
        expanded.f2.bias[old_classes:].copy_(output.bias)
    for key,value in old.items():
        actual=expanded.state_dict()[key]
        if key.startswith("f1."): actual=actual[:old_fc]
        elif key=="f2.weight": actual=actual[:old_classes,:old_fc]
        elif key=="f2.bias": actual=actual[:old_classes]
        assert torch.equal(actual,value),f"old parameter changed: {key}"
    assert not expanded.f2.weight[:old_classes,old_fc:].count_nonzero()
    return expanded


def cache_features(net, corpus, baseline, dev, batch, conv=False):
    cache = corpus / ("fixed-conv-features-wide270_28" if conv else "fixed-features-wide270_28")
    cache.mkdir(exist_ok=True)
    key = {"baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
           "source": {s: {"bytes": (corpus/f"{s}-x.npy").stat().st_size,
                           "mtime_ns": (corpus/f"{s}-x.npy").stat().st_mtime_ns}
                      for s in ("train", "validation")},
           "features": net.f1.in_features if conv else net.f1.out_features,
           "storage": "float16", "compute": "bfloat16 on CUDA"}
    meta = cache / "cache.json"
    if meta.exists() and json.loads(meta.read_text()) == key and all((cache/f"{s}.npy").exists() for s in ("train","validation")):
        print("reusing fixed feature cache", flush=True)
        return cache
    net.eval()
    t0, last = time.time(), time.time()
    for split in ("train", "validation"):
        source = np.load(corpus/f"{split}-x.npy", mmap_mode="r")
        dest = np.lib.format.open_memmap(cache/f"{split}.part.npy", mode="w+", dtype=np.float16, shape=(len(source),key["features"]))
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev=="cuda"):
            for i in range(0,len(source),batch):
                x = torch.from_numpy(np.array(source[i:i+batch])).to(dev).float().div_(255).unsqueeze(1).contiguous(memory_format=torch.channels_last)
                f = features(net,x,include_fc=not conv).to(torch.float16).cpu().numpy()
                assert np.isfinite(f).all()
                dest[i:i+len(f)] = f
                if time.time()-last >= 45:
                    print(f"cache {split} {min(i+batch,len(source))/len(source):.1%} ({time.time()-t0:.0f}s)", flush=True)
                    last=time.time()
        dest.flush(); del dest, source, x, f
        (cache/f"{split}.part.npy").replace(cache/f"{split}.npy")
    meta.write_text(json.dumps(key,indent=1),encoding="utf-8",newline="\n")
    print(f"fixed feature cache complete ({time.time()-t0:.0f}s)", flush=True)
    return cache


def validation(head, x, y, oldw, oldb, batch=8192, old_projection=None):
    counts = {"real": [0,0,0], "synthetic": [0,0,0]}
    false = 0
    with torch.no_grad():
        for i in range(0,len(x),batch):
            xb, yb = x[i:i+batch].float(), y[i:i+batch]
            hidden = F.relu(F.linear(xb,*old_projection)) if old_projection else xb
            logits = F.linear(hidden,oldw,oldb)
            if head is not None:
                logits = torch.cat([logits,head(xb,hidden) if isinstance(head,ExtraHead) else head(hidden)],1)
            top = logits.topk(3,1).indices
            for name, mask in (("real",yb<270),("synthetic",yb>=270)):
                counts[name][0] += int(mask.sum())
                counts[name][1] += int(((top[:,0]==yb)&mask).sum())
                counts[name][2] += int(((top==yb[:,None]).any(1)&mask).sum())
            false += int(((yb<270)&(top[:,0]>=270)).sum())
    result = {n: {"drawings": c[0], "top1":c[1]/max(1,c[0]), "top3":c[2]/max(1,c[0])} for n,c in counts.items()}
    result["false_alarm"] = false/counts["real"][0]
    result["overall"] = sum(c[1] for c in counts.values())/len(y)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--name",default="wide299_28")
    ap.add_argument("--corpus",default="expanded299-28")
    ap.add_argument("--baseline",default=str(OUT/"wide270_28.pt"))
    ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--lr",type=float,default=0.003)
    ap.add_argument("--bs",type=int,default=8192)
    ap.add_argument("--feature-bs",type=int,default=2048)
    ap.add_argument("--chunk",type=int,default=250000)
    ap.add_argument("--extra-hidden",type=int,default=0,help="append new features while retaining all old feature/output values")
    a=ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(6); torch.backends.cudnn.benchmark=True
    dev="cuda" if torch.cuda.is_available() else "cpu"
    start=time.time()
    corpus, baseline=DATA/a.corpus,Path(a.baseline)
    labels=json.loads((corpus/"splits.json").read_text())["categories"]
    assert labels==CATEGORIES and len(labels)==299
    initial=torch.load(baseline,map_location=dev,weights_only=True)
    assert initial["categories"]==labels[:270]
    net=Net([24,48,96],256,270,28).to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(initial["state_dict"]); net.eval().requires_grad_(False)
    cache=cache_features(net,corpus,baseline,dev,a.feature_bs,conv=a.extra_hidden>0)
    xtr=np.load(cache/"train.npy",mmap_mode="r")
    ytr=np.load(corpus/"train-y.npy",mmap_mode="r")
    xv=torch.from_numpy(np.array(np.load(cache/"validation.npy",mmap_mode="r"))).to(dev)
    yv=torch.from_numpy(np.array(np.load(corpus/"validation-y.npy",mmap_mode="r"))).to(dev)
    oldw,oldb=net.f2.weight.detach(),net.f2.bias.detach()
    old_projection=(net.f1.weight.detach(),net.f1.bias.detach()) if a.extra_hidden else None
    base=validation(None,xv,yv,oldw,oldb,old_projection=old_projection)
    print("cached baseline validation",json.dumps(base),flush=True)
    head=(ExtraHead(net.f1.in_features,256,a.extra_hidden) if a.extra_hidden else nn.Linear(256,29)).to(dev)
    output=head.output if isinstance(head,ExtraHead) else head
    with torch.no_grad():
        output.weight.normal_(0,0.01); output.bias.fill_(-3)
    opt=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=1e-4)
    steps=a.epochs*sum((min(a.chunk,len(xtr)-i)+a.bs-1)//a.bs for i in range(0,len(xtr),a.chunk))
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,a.lr,total_steps=steps)
    rng=torch.Generator().manual_seed(0)
    best,best_state,best_epoch=-1,None,0
    history=[]
    last=time.time()
    for ep in range(a.epochs):
        ranges=[(i,min(i+a.chunk,len(xtr))) for i in range(0,len(xtr),a.chunk)]
        ranges=[ranges[i] for i in torch.randperm(len(ranges),generator=rng).tolist()]
        total_loss,seen=0.,0
        for ci,(lo,hi) in enumerate(ranges):
            xc=torch.from_numpy(np.array(xtr[lo:hi])).to(dev)
            yc=torch.from_numpy(np.array(ytr[lo:hi])).to(dev)
            order=torch.randperm(len(xc),device=dev)
            for i in range(0,len(xc),a.bs):
                j=order[i:i+a.bs]
                x,y=xc[j].float(),yc[j]
                hidden=F.relu(F.linear(x,*old_projection)) if old_projection else x
                new_logits=head(x,hidden) if isinstance(head,ExtraHead) else head(hidden)
                logits=torch.cat([F.linear(hidden,oldw,oldb),new_logits],1)
                loss=F.cross_entropy(logits,y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                total_loss+=loss.item()*len(j); seen+=len(j)
            del xc,yc,order,x,y,hidden,new_logits,logits,loss
            if time.time()-last>=45:
                print(f"head epoch {ep+1}/{a.epochs} {ci+1}/{len(ranges)} chunks ({time.time()-start:.0f}s)",flush=True)
                last=time.time()
        metrics=validation(head,xv,yv,oldw,oldb,old_projection=old_projection)
        acceptable=(metrics["real"]["top1"]>=base["real"]["top1"]-0.005 and
                    metrics["real"]["top3"]>=base["real"]["top3"]-0.005 and metrics["false_alarm"]<=0.01)
        history.append({"epoch":ep+1,"loss":total_loss/seen,"metrics":metrics,"guard":acceptable})
        print("head validation",json.dumps(history[-1]),flush=True)
        if acceptable and metrics["overall"]>best:
            best,best_epoch=metrics["overall"],ep+1
            best_state=copy.deepcopy(head.state_dict())
    assert best_state is not None,"No new-head checkpoint met retention/false-alarm guards"
    head.load_state_dict(best_state)
    expanded=assemble(initial,labels,head,[24,48,96])
    path=OUT/(a.name+"-head.pt")
    torch.save({"state_dict":expanded.cpu().state_dict(),"categories":labels,"args":vars(a),
                "total_epochs":a.epochs,"training_mode":"frozen backbone and old output rows",
                "selected_head_epoch":best_epoch},path)
    head_info={"train_seconds":time.time()-start,"baseline":str(baseline),"best_epoch":best_epoch,
               "epochs":a.epochs,"baseline_validation":base,"history":history,"old_parameters_exact":True,
               "extra_hidden":a.extra_hidden,
               "mode":f"frozen old network/output path; {a.extra_hidden} appended nonlinear features; no extra image augmentation"}
    path.with_suffix(".json").write_text(json.dumps(head_info,indent=1),encoding="utf-8",newline="\n")
    print("Evaluating the full exported model, not cached features",flush=True)
    subprocess.run([sys.executable,"-u",str(HERE/"train.py"),"--corpus",a.corpus,"--name",a.name,
                    "--size","28","--widths","24,48,96","--fc",str(256+a.extra_hidden),"--epochs","0",
                    "--resume",str(path),"--clean","--chunk",str(a.chunk)],check=True)
    result_path=OUT/(a.name+".json")
    result=json.loads(result_path.read_text())
    result.update({"training_mode":head_info["mode"],"initialized_from":str(baseline),
                   "epochs":a.epochs,"best_epoch":best_epoch,"selected_epoch":best_epoch,
                   "head_training":head_info,"old_parameters_exact":True})
    # Per-output-channel quantization must preserve all old weights exactly too.
    with np.load(OUT/"wide270_28.npz") as oldq, np.load(OUT/(a.name+".npz")) as newq:
        for i in range(5):
            for prefix in ("w","s","b"):
                old,new=oldq[f"{prefix}{i}"],newq[f"{prefix}{i}"]
                if i==3:new=new[:256]
                if i==4:
                    new=new[:270,:256] if prefix=="w" else new[:270]
                assert np.array_equal(old,new),("old int8 weights changed",prefix,i)
        assert not newq["w4"][:270,256:].any(),"new features leaked into old outputs"
    result["old_quantized_weights_exact"]=True
    result_path.write_text(json.dumps(result,indent=1),encoding="utf-8",newline="\n")
    print("Old floating and quantized parameters preserved exactly.",flush=True)


if __name__=="__main__":
    main()
