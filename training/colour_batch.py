"""Compiled training sampler for colour.py's distribution; no runtime class table.

Evaluation uses colour.colourise itself with a fixed Generator seed. This sampler
uses independently seeded Numba RNG streams to recolour every training pass.
"""
import numpy as np
from numba import njit
from categories import CATEGORIES
from colour import _PARSED, P_BLACK, P_RANDOM, S_SINGLE, S_OUTLINE, BIG

KIND=np.zeros((299,2),np.uint8)
CDF=np.zeros((299,2,10),np.float64)
SEQ=np.zeros((299,2,10),np.uint8)
COUNT=np.zeros((299,2),np.int64)
for ci,name in enumerate(CATEGORIES):
    for part,(kind,value) in enumerate(_PARSED[name]):
        if kind=='cycle':
            KIND[ci,part]=1;COUNT[ci,part]=len(value);SEQ[ci,part,:len(value)]=value
        else:
            CDF[ci,part]=np.cumsum(value);CDF[ci,part,-1]=1.


@njit(cache=True)
def _pick(ci,part,kind,count,cdf):
    if kind[ci,part]:return int(np.random.random()*count[ci,part])
    r=np.random.random()
    for j in range(10):
        if r<cdf[ci,part,j]:return j
    return 9


@njit(cache=True)
def _sample(labels,offset,length,seed,mode,kind,cdf,seq,count,p_black,p_random,s_single,s_outline,big_share):
    """mode: 0 colourise mixture; 1 independent random inks; 2 all black."""
    np.random.seed(seed)
    out=np.zeros(len(length),np.uint8)
    if mode==2:return out
    for row in range(len(labels)):
        lo,hi=offset[row],offset[row+1]
        if mode==1:
            for j in range(lo,hi):out[j]=int(np.random.random()*10)
            continue
        r=np.random.random()
        if r<p_black:continue
        total=0.;best=lo
        for j in range(lo,hi):
            total+=length[j]
            if length[j]>length[best]:best=j
        if r<p_black+p_random:
            a,b=int(np.random.random()*10),int(np.random.random()*10)
            single=np.random.random()<.5
            for j in range(lo,hi):out[j]=a if single or j==best or length[j]/total>=big_share else b
            continue
        ci=labels[row];main,accent=_pick(ci,0,kind,count,cdf),_pick(ci,1,kind,count,cdf);style=np.random.random()
        for j in range(lo,hi):
            si=j-lo
            m=seq[ci,0,(main+si)%count[ci,0]] if kind[ci,0] else main
            a=seq[ci,1,(accent+si)%count[ci,1]] if kind[ci,1] else accent
            big=j==best or length[j]/total>=big_share
            if style<s_single:out[j]=m
            elif style<s_single+s_outline:out[j]=0 if big else a
            else:out[j]=m if big or np.random.random()<.25 else a
    return out


def sample(labels,offset,length,seed,mode=0):
    # Pass policy arrays explicitly: a cached JIT must not bake in stale priors.
    return _sample(labels,offset,length,seed,mode,KIND,CDF,SEQ,COUNT,P_BLACK,P_RANDOM,S_SINGLE,S_OUTLINE,BIG)


@njit(cache=True)
def _fills(labels,offset,length,colours,seed,kind,cdf,seq,count,p_own,p_black):
    np.random.seed(seed)
    out=np.zeros(len(labels),np.uint8)
    for row in range(len(labels)):
        lo,hi=offset[row],offset[row+1];best=lo;all_black=True
        for j in range(lo,hi):
            if length[j]>length[best]:best=j
            if colours[j]!=0:all_black=False
        own=colours[best]
        if own!=0 and np.random.random()<p_own:out[row]=own
        elif all_black and np.random.random()<p_black:out[row]=0
        else:
            ci=labels[row];v=_pick(ci,0,kind,count,cdf)
            out[row]=seq[ci,0,v] if kind[ci,0] else v
    return out


def sample_fills(labels,offset,length,colours,seed):
    from fill import P_OWN,P_BLACK_FILL
    return _fills(labels,offset,length,colours,seed,KIND,CDF,SEQ,COUNT,P_OWN,P_BLACK_FILL)
