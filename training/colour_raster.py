"""Ink plus nearest-stroke ownership, with stable later-stroke numerical ties."""
import math
import numpy as np
from numba import njit
from fast_raster import render_flat

TIE_EPS = 1e-12


def flatten_owned(strokes):
    counts = [len(s[0]) for s in strokes]
    offsets = np.r_[np.int64(0), np.cumsum(counts,dtype=np.int64)]
    assert offsets[-1] > 0 and len(counts) <= 65535
    xs = np.concatenate([np.asarray(s[0],np.float64) for s in strokes])
    ys = np.concatenate([np.asarray(s[1],np.float64) for s in strokes])
    return xs,ys,offsets


@njit(cache=True)
def owned_flat(xs,ys,offsets,size=28,box=22.,radius=1.):
    minx,miny,maxx,maxy=np.min(xs),np.min(ys),np.max(xs),np.max(ys)
    extent=max(maxx-minx,maxy-miny)
    scale=box/extent if extent>1e-6 else 1.
    cx,cy=(minx+maxx)*.5,(miny+maxy)*.5
    half,reach=size/2,radius+.5
    dmin=np.full((size,size),np.inf,np.float64)
    owner=np.zeros((size,size),np.uint16)
    lengths=np.ones(len(offsets)-1,np.float64)
    for si in range(len(offsets)-1):
        lo,hi=offsets[si],offsets[si+1]
        if lo==hi:continue
        for j in range(lo,hi-1):
            lengths[si]+=math.hypot(xs[j+1]-xs[j],ys[j+1]-ys[j])
        for ai in range(lo,max(lo+1,hi-1)):
            bi=min(ai+1,hi-1)
            ax,ay=(xs[ai]-cx)*scale+half,(ys[ai]-cy)*scale+half
            bx,by=(xs[bi]-cx)*scale+half,(ys[bi]-cy)*scale+half
            dx,dy=bx-ax,by-ay
            ll=dx*dx+dy*dy
            x0,x1=max(0,math.ceil(min(ax,bx)-reach-.5)),min(size-1,math.floor(max(ax,bx)+reach-.5))
            y0,y1=max(0,math.ceil(min(ay,by)-reach-.5)),min(size-1,math.floor(max(ay,by)+reach-.5))
            for py in range(y0,y1+1):
                for px in range(x0,x1+1):
                    fx,fy=px+.5,py+.5
                    t=0.
                    if ll>1e-12:t=min(1.,max(0.,((fx-ax)*dx+(fy-ay)*dy)/ll))
                    ex,ey=fx-(ax+t*dx),fy-(ay+t*dy)
                    d2=ex*ex+ey*ey
                    prev=dmin[py,px]
                    if d2<=prev+TIE_EPS:owner[py,px]=si
                    if d2<prev:dmin[py,px]=d2
    ink=np.zeros((size,size),np.uint8)
    for y in range(size):
        for x in range(size):
            if dmin[y,x]<reach*reach:
                ink[y,x]=np.uint8(np.rint(min(1.,reach-math.sqrt(dmin[y,x]))*255.))
    return ink,owner,lengths


def render(strokes,size=28):
    return owned_flat(*flatten_owned(strokes),size,22.*size/28,size/28)


def reference(strokes,size=28):
    """Independent NumPy float64 reference (not the compiled dataset builder)."""
    xs,ys,offsets=flatten_owned(strokes)
    mn=np.array([xs.min(),ys.min()]);mx=np.array([xs.max(),ys.max()])
    extent=max(mx-mn);scale=22.*size/28/extent if extent>1e-6 else 1.
    points=np.stack([xs,ys],1);points=(points-(mn+mx)*.5)*scale+size/2
    yy,xx=np.mgrid[:size,:size];px=xx+.5;py=yy+.5
    distance=np.full((size,size),np.inf);owner=np.zeros((size,size),np.uint16)
    for si,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
        if lo==hi:continue
        best=np.full((size,size),np.inf)
        for ai in range(lo,max(lo+1,hi-1)):
            a,b=points[ai],points[min(ai+1,hi-1)];dx,dy=b-a;ll=dx*dx+dy*dy
            t=np.clip(((px-a[0])*dx+(py-a[1])*dy)/ll,0,1) if ll>1e-12 else 0.
            ex,ey=px-(a[0]+t*dx),py-(a[1]+t*dy)
            best=np.minimum(best,ex*ex+ey*ey)
        owner[best<=distance+TIE_EPS]=si
        distance=np.minimum(distance,best)
    ink=np.clip(size/28+.5-np.sqrt(distance),0,1)
    owner[ink==0]=0
    return ink,owner
