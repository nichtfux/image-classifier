"""Exploratory colour-adapter control after visual QA; never a selected headline.

Same preserved ink/crop/padding as external_eval, but continuous RGB instead of
nearest-palette RGB. This is outside the ten discrete training inks; no labels
or class priors are used. DomainNet source archives remain read-only.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import io,json,zipfile,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
from external_eval import ROOT,OUT,load_net,metrics,write_json

ARCHIVES={}
def convert(job):
    i,archive,member=job
    try:
        if archive not in ARCHIVES:ARCHIVES[archive]=zipfile.ZipFile(archive)
        image=Image.open(io.BytesIO(ARCHIVES[archive].read(member))).convert('RGBA');image.thumbnail((1600,1600))
        a=np.asarray(image,dtype=np.float32)/255;rgb=a[:,:,:3];alpha=a[:,:,3]
        if np.mean(alpha<.99)>.01:mask=alpha>.10
        else:
            border=np.concatenate((rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]));bg=np.median(border,axis=0)
            mask=np.maximum(np.maximum(abs(rgb[:,:,0]-bg[0]),abs(rgb[:,:,1]-bg[1])),abs(rgb[:,:,2]-bg[2]))>.10
        ys,xs=np.nonzero(mask)
        if not len(xs):return i,None
        rgb=rgb[ys.min():ys.max()+1,xs.min():xs.max()+1];mask=mask[ys.min():ys.max()+1,xs.min():xs.max()+1]
        h,w=mask.shape;shape=(max(1,round(w*256/max(h,w))),max(1,round(h*256/max(h,w))))
        alpha=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize(shape,Image.Resampling.BOX),np.float32)/255
        rgb=np.asarray(Image.fromarray((rgb*255).astype(np.uint8)).resize(shape,Image.Resampling.BILINEAR),np.float32)/255
        rgb[rgb.max(2)<.16]=0 # black ink network convention; not a class prior
        source=np.concatenate((alpha[None],rgb.transpose(2,0,1)*alpha[None]),0)
        fit=4*(24*40/28)/max(source.shape[1:]);shape=(max(1,round(source.shape[2]*fit)),max(1,round(source.shape[1]*fit)))
        output=[]
        for p in source:
            canvas=Image.new('F',(160,160));canvas.paste(Image.fromarray(p).resize(shape,Image.Resampling.BOX),((160-shape[0])//2,(160-shape[1])//2))
            output.append(np.asarray(canvas.resize((40,40),Image.Resampling.BOX),np.float32))
        return i,np.stack(output).astype(np.float16)
    except Exception:return i,None

def main():
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    result={}
    for domain in ('clipart','painting'):
        folder=OUT/domain;rows=(ROOT/f'data/external/domainnet/{domain}_test.txt').read_text().splitlines()
        path=folder/'continuous-rgb40.npy'
        if not path.exists():
            arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(len(rows),4,40,40))
            jobs=[(i,str(ROOT/f'data/external/domainnet/{domain}.zip'),row.rsplit(' ',1)[0]) for i,row in enumerate(rows)]
            with ProcessPoolExecutor(max_workers=4) as pool:
                for n,(i,x) in enumerate(pool.map(convert,jobs,chunksize=20),1):
                    arr[i]=0 if x is None else x
                    if n%5000==0:print('rgb control',domain,n,flush=True)
            arr.flush()
        while not (folder/'complete.json').exists():time.sleep(1)
        x=np.load(path,mmap_mode='r');preserve=np.load(folder/'preserve40.npy',mmap_mode='r')
        diff=0.
        for lo in range(0,len(x),512):diff=max(diff,float(np.max(abs(x[lo:lo+512,0]-preserve[lo:lo+512,0]))))
        assert diff==0,'RGB control changed ink geometry'
        truth=np.load(folder/'truth.npy');valid=np.load(folder/'valid.npy');top=np.zeros((len(x),3),np.int16)
        net=load_net('colour299_40','cuda')
        with torch.inference_mode():
            for lo in range(0,len(x),256):
                p=net(torch.from_numpy(np.array(x[lo:lo+256],np.float32)).cuda().contiguous(memory_format=torch.channels_last))
                top[lo:lo+256]=p.topk(3,1).indices.cpu().numpy()
        top[~valid]=-1
        result[domain]={'ink_pixel_difference':diff,**metrics(truth,top,valid)}
        print('RGB SCORE',domain,result[domain]['all'],flush=True)
        write_json(ROOT/'notes/external-rgb-control.json',result)
        del net

if __name__=='__main__':main()
