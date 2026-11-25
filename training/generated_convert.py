"""Generated bitmap -> legal thin-line game strokes -> exact 40px owner raster.

Display brush width controls scan-stroke spacing; serving still uses thin lines.
No inference-only fill flag or direct bitmap shortcut is used for training.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.measure import approximate_polygon
from colour_raster import render
from colour import PALETTE,NET,planes

VERSION='generated-strokes-v4'
RGB=np.asarray([p.rgb for p in PALETTE],np.float32)


def palette_indices(patch):
    """Match ink hue through white-paper dilution and darker paint shades.

    Raw RGB distance mapped pastel blue to pink. Compare each palette ink's
    white/black mixing rays instead; neutral pixels still stay black or grey.
    """
    p=patch.astype(np.float32)/255.;base=RGB/255.
    q=1-p;white=1-base
    a=np.clip((q[:,:,None,:]*white[None,None]).sum(3)/(white*white).sum(1),0,1)
    white_error=((p[:,:,None,:]-(1-a[:,:,:,None]*white[None,None]))**2).sum(3)
    b=np.clip((p[:,:,None,:]*base[None,None]).sum(3)/(base*base).sum(1),0,1)
    shade_error=((p[:,:,None,:]-b[:,:,:,None]*base[None,None])**2).sum(3)
    owner=np.minimum(white_error,shade_error).argmin(2)
    chroma=patch.max(2)-patch.min(2)
    relative=chroma/np.maximum(1,np.minimum(patch.max(2),255-patch.min(2)))
    neutral=relative<.15
    neutral_distance=((patch[:,:,None,:]-RGB[None,None,:2])**2).sum(3)
    owner[neutral]=neutral_distance.argmin(2)[neutral]
    return owner


def validate_geometry(strokes):
    if not 1<=len(strokes)<=128 or sum(len(s[0]) for s in strokes)>4096:
        raise ValueError('too complex for game stroke payload')
    if any(not 1<=len(s[0])<=2048 or len(s[0])!=len(s[1]) for s in strokes):
        raise ValueError('too many points in one game stroke')
    xs=np.concatenate([s[0] for s in strokes]);ys=np.concatenate([s[1] for s in strokes])
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():raise ValueError('non-finite game coordinates')
    span=max(float(xs.max()-xs.min()),float(ys.max()-ys.min()),1e-9)
    length=sum(float(np.abs(np.diff(s[0])).sum()+np.abs(np.diff(s[1])).sum()) for s in strokes)
    if length>200*span:raise ValueError('excess repeated ink for game work limit')
    return length/span


def trace(mask):
    points=set(zip(*np.nonzero(skeletonize(mask))))
    def neighbors(p):
        y,x=p;out=[]
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                if not (dx or dy):continue
                q=(y+dy,x+dx)
                if q not in points:continue
                if dx and dy and ((y,x+dx) in points or (y+dy,x) in points):continue
                out.append(q)
        return out
    graph={p:neighbors(p) for p in points};used=set();visited=set();result=[]
    # One continuous pen walk per connected skeleton, retracing branches when
    # necessary. Splitting at every junction made simple shaded shapes exceed
    # the game's stroke cap. Retracing preserves the same thin-line coverage.
    for start in sorted(points):
        if start in visited:continue
        stack=[start];path=[start];visited.add(start)
        while stack:
            cur=stack[-1]
            nxt=next((q for q in graph[cur] if tuple(sorted((cur,q))) not in used),None)
            if nxt is None:
                stack.pop()
                if stack:path.append(stack[-1])
            else:
                used.add(tuple(sorted((cur,nxt))));visited.add(nxt)
                stack.append(nxt);path.append(nxt)
        result.append(np.asarray(path,float)[:,::-1])
    return result


def scans(mask,spacing,rng):
    # Axis changes make genuine alternate stroke geometries, not raster dilation.
    angle=float(rng.uniform(0,180));original_shape=mask.shape
    rotated=ndi.rotate(mask,angle,reshape=True,order=0,prefilter=False)
    c,s=np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
    matrix=np.asarray([[c,-s],[s,c]])
    source_center=np.asarray([(original_shape[1]-1)/2,(original_shape[0]-1)/2])
    rotated_center=np.asarray([(rotated.shape[1]-1)/2,(rotated.shape[0]-1)/2])
    mask=rotated
    horizontal=bool(rng.integers(2));m=mask if horizontal else mask.T
    result=[];phase=rng.uniform(0,max(1,spacing))
    for fy in np.arange(phase,m.shape[0],spacing):
        y=int(fy);row=m[y];runs=np.diff(np.r_[False,row,False].astype(np.int8))
        for lo,hi in zip(np.flatnonzero(runs==1),np.flatnonzero(runs==-1)):
            if hi-lo<2:continue
            pts=np.asarray([[lo+.5,fy],[hi-.5,fy]],float)
            if not horizontal:pts=pts[:,::-1]
            # Hand-paint waviness, rather than geometrically perfect fill rows.
            if hi-lo>12:
                mid=pts.mean(0)+rng.normal(0,.3,2);pts=np.vstack([pts[0],mid,pts[-1]])
            if len(result)%2:pts=pts[::-1]
            # Join consecutive paint sweeps only when the connecting segment
            # stays inside the same colour region. Never bridge a white hole.
            if result:
                line=np.linspace(result[-1][-1],pts[0],max(2,int(np.linalg.norm(result[-1][-1]-pts[0])*2)+1))
                px=np.rint(line[:,0]).astype(int).clip(0,mask.shape[1]-1)
                py=np.rint(line[:,1]).astype(int).clip(0,mask.shape[0]-1)
                if mask[py,px].all():result[-1]=np.vstack([result[-1],pts]);continue
            result.append(pts)
    return [(p-rotated_center)@matrix.T+source_center for p in result]


def convert(image,style,seed):
    rng=np.random.default_rng(seed)
    rgb=np.asarray(image.convert('RGB'),np.float32)
    border=np.concatenate([rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]])
    bg=np.median(border,axis=0)
    if float(bg.mean())<220 or float(np.percentile(np.linalg.norm(border-bg,axis=1),90))>45:
        raise ValueError('background is not plain light paper')
    mask=np.linalg.norm(rgb-bg,axis=2)>35
    labels,n=ndi.label(mask)
    if not n:raise ValueError('blank')
    counts=np.bincount(labels.ravel());keep=np.flatnonzero(counts>=max(4,mask.sum()*.00025));keep=keep[keep!=0]
    mask=np.isin(labels,keep)
    ys,xs=np.nonzero(mask)
    if len(xs)<12:raise ValueError('too little foreground')
    lo=(int(xs.min()),int(ys.min()));hi=(int(xs.max())+1,int(ys.max())+1)
    crop=image.convert('RGB').crop((*lo,*hi));w,h=crop.size
    scale=112/max(w,h);ww,hh=max(2,round(w*scale)),max(2,round(h*scale))
    patch=np.asarray(crop.resize((ww,hh),Image.Resampling.LANCZOS),np.float32)
    patch=ndi.median_filter(patch,size=(3,3,1))
    fg=Image.fromarray(mask[lo[1]:hi[1],lo[0]:hi[0]].astype(np.uint8)*255).resize((ww,hh),Image.Resampling.NEAREST)
    fg=np.asarray(fg)>127
    owner=palette_indices(patch)
    if style in ['marker','outline']:owner[:]=0
    strokes=[];colours=[];widths=[]
    filled=style in ['flat','painted','kids']
    for ci in list(range(1,10))+[0]: # black detail strokes drawn last
        region=(owner==ci)&fg
        cc,n=ndi.label(region)
        for component in range(1,n+1):
            part=cc==component;yy,xx=np.nonzero(part)
            if len(xx)<5:continue
            density=len(xx)/((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1))
            brush=float(rng.uniform(8,56)) if filled else float(rng.uniform(3,12))
            do_paint=filled and len(xx)>28 and density>.28
            if do_paint:
                # Human fills include overlapping passes as well as visible
                # stripes. Serving ignores display width, so the actual point
                # paths must cover solid areas rather than relying on width.
                kind=rng.choice(3,p=[.6,.3,.1])
                spacing=rng.uniform(*[(.7,2.5),(3.,7.),(8.,12.)][kind])
                paths=scans(part,max(1.,spacing),rng)
                if rng.random()<.35:paths+=trace(part)
            else:paths=trace(part)
            for path in paths:
                if len(path)>2:path=approximate_polygon(path,tolerance=float(rng.uniform(.25,.8)))
                if not len(path):continue
                path=(path+4)*2
                for start in range(0,max(1,len(path)-1),2047):
                    piece=path[start:start+2048]
                    strokes.append([piece[:,0].tolist(),piece[:,1].tolist()]);colours.append(ci);widths.append(brush)
    if not strokes:raise ValueError('no strokes after conversion')
    # Payload limits stricter than the game endpoint, without silently clipping geometry.
    work_ratio=validate_geometry(strokes)
    ink,pixel_owner,length=render(strokes,40)
    x=np.rint(planes(ink.astype(np.float32)/255,pixel_owner,colours)*255).astype(np.uint8)
    if int((x[0]>100).sum())<8:raise ValueError('too little converted ink')
    return {'x':x,'strokes':strokes,'colours':colours,'widths':widths,'version':VERSION,
            'source_bbox':[list(lo),list(hi)],'stroke_count':len(strokes),'point_count':sum(len(s[0]) for s in strokes),
            'work_length_to_span':work_ratio}


def preview(x,size=384):
    v=x.astype(np.float32)/255
    rgb=np.clip(1-v[0,:,:,None]+v[1:].transpose(1,2,0),0,1)
    return Image.fromarray(np.rint(rgb*255).astype(np.uint8)).resize((size,size),Image.Resampling.NEAREST)
