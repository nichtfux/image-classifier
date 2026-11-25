"""Cosine-margin GACL, Yang et al., Eq. 5/14 and early soft binning Eq. 18.

q is a linear rescaling of the embedding L2 norm, bounded to [.35,.8].
The target cosine logit is reduced by q; negative logits are unchanged.
G(q)=1/q+q/.8**2; lambda=.2 exceeds the paper's .1515 sufficient bound.
No quality annotations, external data, confidence-to-tier mapping or colour table
are used as quality targets. Scores still require checking against human ratings.
"""
import torch
import torch.nn.functional as F
from train import Net


class QualityNet(Net):
    def __init__(self, widths=(48,96,192), fc=384, size=40, in_channels=4):
        super().__init__(widths, fc, 299, size=size, in_channels=in_channels)
        self.register_buffer('norm_scale', torch.tensor(100.))
        self.cosine = False
        self.quantized_head = False

    def features(self, x):
        for i in (1,2,3):
            x = F.max_pool2d(F.relu(getattr(self, f'b{i}')(getattr(self, f'c{i}')(x))), 2)
        return F.relu(self.f1(x.flatten(1)))

    def forward(self, x, details=False):
        h = self.features(x)
        norm = h.float().norm(dim=1).clamp_min(1e-8)
        if self.cosine:
            weight = self.f2.weight.float() if self.quantized_head else 64*F.normalize(self.f2.weight.float(),dim=1)
            z = F.linear(F.normalize(h.float(),dim=1), weight, self.f2.bias.float())
        else:
            z = self.f2(h)
        return (z, norm) if details else z


def loss(logits, norm, labels, scale, soft_bin=False):
    q = (norm / scale).clamp(.35, .8)
    if soft_bin:
        # Five fixed bucket representatives with four midpoint cut points.
        values = torch.linspace(.35,.8,5,device=q.device)
        cuts = (values[:-1]+values[1:])*.5
        weights = torch.arange(1,6,device=q.device)
        bias = torch.cat([torch.zeros(1,device=q.device), -cuts.cumsum(0)])
        q = (((q[:,None]*weights+bias)/.008).softmax(1)*values).sum(1)
    margin = torch.zeros_like(logits).scatter_(1, labels[:,None], q[:,None])
    ce = F.cross_entropy(logits-margin, labels)
    reg = (1/q + q/.8**2).mean()
    return ce + .2*reg, {'ce':float(ce.detach()), 'q':float(q.mean().detach())}


def widen_from(state, device='cuda'):
    """Function-preserving Net2Wider initialization, then tiny symmetry breaking."""
    net = QualityNet().to(device).to(memory_format=torch.channels_last)
    result = net.state_dict()
    maps = [torch.arange(n,device=device)%old for n,old in [(48,24),(96,48),(192,96),(384,288)]]
    for i, outmap in enumerate(maps[:3],1):
        w=state[f'c{i}.weight'].to(device)[outmap]
        if i>1:w=w[:,maps[i-2]]/2
        result[f'c{i}.weight'].copy_(w)
        for key in ['weight','bias','running_mean','running_var']:
            result[f'b{i}.{key}'].copy_(state[f'b{i}.{key}'].to(device)[outmap])
        result[f'b{i}.num_batches_tracked'].copy_(state[f'b{i}.num_batches_tracked'])
    result['f1.weight'].copy_(state['f1.weight'].to(device)[maps[3]].reshape(384,96,25)[:,maps[2]].reshape(384,-1)/2)
    result['f1.bias'].copy_(state['f1.bias'].to(device)[maps[3]])
    counts=torch.bincount(maps[3],minlength=288)
    result['f2.weight'].copy_(state['f2.weight'].to(device)[:,maps[3]]/counts[maps[3]])
    result['f2.bias'].copy_(state['f2.bias'].to(device))
    net.load_state_dict(result)
    return net


def quantize(net):
    """Fold cosine class weights once; serving normalizes the embedding only."""
    import copy
    from train_colour import quantise_colour
    folded=copy.deepcopy(net).eval()
    if net.cosine and not net.quantized_head:
        with torch.no_grad():
            folded.f2.weight.copy_(64*F.normalize(net.f2.weight,dim=1))
    return quantise_colour(folded)


def from_quant(q, device='cuda', cosine=True, norm_scale=100.):
    from train import dequant
    widths=[q[i][0].shape[0] for i in range(3)];fc=q[3][0].shape[0]
    grid=round((q[3][0].shape[1]/widths[-1])**.5)
    size=28 if grid==3 else 8*grid
    net=QualityNet(widths,fc,size,in_channels=q[0][0].shape[1]).to(device).to(memory_format=torch.channels_last)
    plain=dequant(q)
    with torch.no_grad():
        for i,(w,b) in enumerate(plain[:3],1):
            conv=getattr(net,f'c{i}');conv.weight.copy_(torch.from_numpy(w).to(device))
            conv.bias=torch.nn.Parameter(torch.from_numpy(b).float().to(device))
            setattr(net,f'b{i}',torch.nn.Identity())
        for target,(w,b) in zip((net.f1,net.f2),plain[3:]):
            target.weight.copy_(torch.from_numpy(w).to(device));target.bias.copy_(torch.from_numpy(b).to(device))
    net.cosine=cosine;net.quantized_head=cosine;net.norm_scale.fill_(norm_scale)
    return net.eval().requires_grad_(False)


def reference(layers, x, normalize=False, details=False):
    """Independent float64 exported-weight reference, including feature norm."""
    params=[(torch.from_numpy(w),torch.from_numpy(b)) for w,b in layers]
    for w,b in params[:3]:x=F.max_pool2d(F.relu(F.conv2d(x,w,b,padding=1)),2)
    h=F.relu(F.linear(x.flatten(1),*params[3]));norm=h.norm(dim=1)
    if normalize:h=F.normalize(h,dim=1)
    z=F.linear(h,*params[4])
    return (z,norm) if details else z
