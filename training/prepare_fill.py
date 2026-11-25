"""Cache only deterministic enclosed regions; colour/lazy fills still change each epoch.

Packed bits cost 98/200 bytes per drawing at 28/40px. Bounded read-only ink chunks
avoid four-plane corpus copies and Windows copy-on-write commit reservations.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from numba import njit,prange,set_num_threads
from fill import INK_T,STEPS


@njit(cache=True,parallel=True)
def regions(images,steps,threshold):
    n,h,w=images.shape;out=np.zeros((n,h,w),np.bool_)
    for b in prange(n):
        ink=images[b]>threshold;wall=np.zeros((h,w),np.bool_)
        for y in range(h):
            for x in range(w):
                if ink[y,x]:
                    for yy in range(max(0,y-1),min(h,y+2)):
                        for xx in range(max(0,x-1),min(w,x+2)):wall[yy,xx]=True
        dist=np.full((h,w),-1,np.int16);qy=np.empty(h*w,np.int16);qx=np.empty(h*w,np.int16);head=0;tail=0
        for y in range(h):
            for x in range(w):
                if (y==0 or x==0 or y==h-1 or x==w-1) and not wall[y,x]:
                    dist[y,x]=0;qy[tail]=y;qx[tail]=x;tail+=1
        while head<tail:
            y,x=qy[head],qx[head];head+=1
            if dist[y,x]>=steps:continue
            for yy in range(max(0,y-1),min(h,y+2)):
                for xx in range(max(0,x-1),min(w,x+2)):
                    if dist[yy,xx]<0 and not wall[yy,xx]:
                        dist[yy,xx]=dist[y,x]+1;qy[tail]=yy;qx[tail]=xx;tail+=1
        for y in range(h):
            for x in range(w):
                if ink[y,x] or dist[y,x]>=0:continue
                give_back=False
                if wall[y,x]:
                    for yy in range(max(0,y-1),min(h,y+2)):
                        for xx in range(max(0,x-1),min(w,x+2)):
                            if dist[yy,xx]>=0:give_back=True
                out[b,y,x]=not give_back
    return out


def prepare(folder,threads=4):
    import hashlib,json
    from colour_data import ColourData
    set_num_threads(threads)
    for split in ('validation','test','train'):
        data=ColourData(split,folder);path=data.root/f'{split}-regions.npy';marker=path.with_suffix('.json')
        signature={'size':data.size,'drawings':len(data.y),'fill_sha':hashlib.sha256(Path(__file__).with_name('fill.py').read_bytes()).hexdigest()}
        if marker.exists() and json.loads(marker.read_text())==signature:continue
        dst=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(len(data.y),(data.size**2+7)//8));t=time.time()
        for lo in range(0,len(data.y),100000):
            hi=min(lo+100000,len(data.y));r=regions(np.asarray(data.x[lo:hi]),max(STEPS,round(STEPS*data.size/28)),INK_T*255)
            dst[lo:hi]=np.packbits(r.reshape(len(r),-1),axis=1)
            if lo%2000000==0:print(folder,split,hi,len(data.y),round(time.time()-t),'s',flush=True)
        dst.flush();del dst
        marker.write_text(json.dumps(signature),encoding='utf-8',newline='\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--folder',default='colour299-28');ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args();prepare(a.folder,a.threads)
