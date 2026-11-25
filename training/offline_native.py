"""Offline Luau CLI native/VM ratio and parity; no Roblox/Studio connection.

CLI native timings are measurements on this PC, not fresh Roblox server timings.
Calibrate the ratio with the last measured server native baseline (6.87 ms).
"""
import argparse,json,re,shutil,subprocess,math
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
EXE=ROOT/'training/tools/luau-0.746/luau.exe'

SCRIPT=r'''
local R=require("./DrawRecognizer")
local vectors=require("./TestVectors")
local images={}
local maxLogit,maxRaster,maxNorm,topMismatch=0,0,0,0
local complexStrokes,complexColours,complexPoints=nil,nil,0
for vi,v in vectors do
    local strokes={}
    for _,flat in v.strokes do
        local stroke={}
        for i=1,#flat,2 do table.insert(stroke,vector.create(flat[i],flat[i+1],0)) end
        table.insert(strokes,stroke)
    end
    local points=0;for _,st in strokes do points+=#st end
    if points>complexPoints then complexStrokes,complexColours,complexPoints=strokes,v.colours,points end
    local image=R.rasterize(strokes,v.colours)
    if not image then error("missing image") end
    local sums=table.create(4,0)
    for c=1,R.inputChannels do
        for y=0,R.size-1 do for x=0,R.size-1 do
            sums[c]+=image[(c-1)*R.planeSize+(y+1)*R.paddedSize+x+2]
        end end
        if v.imageSums then maxRaster=math.max(maxRaster,math.abs(sums[c]-v.imageSums[c])) end
    end
    local z,norm=R.logits(image)
    for i=1,#z do maxLogit=math.max(maxLogit,math.abs(z[i]-v.logits[i])) end
    if v.featureNorm then maxNorm=math.max(maxNorm,math.abs(norm-v.featureNorm)) end
    if vi==1 then
        local _,_,_,syncImage,syncNorm=R.recognize(strokes,3,v.colours)
        assert(#syncImage==#image,"preview layout changed")
        local job=R.beginRecognize(strokes,3,v.colours)
        while not job:step(.75) do end
        assert(job.result.top[1].label==R.labels[(function()
            local best=1;for i=2,#z do if z[i]>z[best] then best=i end end;return best
        end)()],"sliced recognition disagrees")
        if v.featureNorm then
            assert(math.abs(syncNorm-norm)<1e-6,"sync quality norm mismatch")
            assert(math.abs(job.result.qualityNorm-norm)<1e-6,"sliced quality norm mismatch")
        end
    end
    local best,ref=1,1
    for i=1,#z do if z[i]>z[best] then best=i end;if v.logits[i]>v.logits[ref] then ref=i end end
    if best~=ref then topMismatch+=1 end
    table.insert(images,image)
end
print("PARITY",#images,maxLogit,maxRaster,maxNorm,topMismatch)
-- Identical timing inputs for both architectures, independent of vector count.
images={}
for mode=1,3 do
    local image=table.create(4*R.planeSize,0)
    for y=0,R.size-1 do for x=0,R.size-1 do
        local ink=if (x-20)^2+(y-20)^2<220 then 1 else 0
        local i=(y+1)*R.paddedSize+x+2
        image[i]=ink
        for c=1,3 do
            image[c*R.planeSize+i]=if mode==1 then 0 elseif mode==2 then ink*({.89,.19,.19})[c] else ink*((x*13+y*7+c*3)%29)/29
        end
    end end
    table.insert(images,image)
end
-- Alternate representative black, policy and random-coloured drawings.
for i=1,5 do R.forward(images[(i-1)%#images+1]) end
local times={}
for i=1,60 do
    local t=os.clock();R.forward(images[(i-1)%#images+1]);times[i]=(os.clock()-t)*1000
end
table.sort(times)
print("FORWARD",times[30],times[57],times[60])
local active,slice,slices,maxProbabilityError=0,0,0,0
local jobs,expected={},{}
for i=1,8 do
    local image=images[(i-1)%#images+1]
    jobs[i]=R.beginForward(image);expected[i]=R.forward(image)
end
local remaining=8
while remaining>0 do
    for i,job in jobs do
        if job.state~="completed" then
            if job:step(.75) then remaining-=1 end
        end
    end
end
for i,job in jobs do
    active+=job.stats.activeMs;slice=math.max(slice,job.stats.maxSliceMs);slices+=job.stats.slices
    for j=1,#expected[i] do maxProbabilityError=math.max(maxProbabilityError,math.abs(expected[i][j]-job.result[j])) end
end
print("BURST8",active,slice,slices)
print("SLICE_PARITY",maxProbabilityError)
assert(maxProbabilityError<1e-6,"interleaved jobs disagree with synchronous forward")
-- A private, independently scripted painted circle/stem: same input on both nets.
local strokes,colours={},{}
local outline={}
for i=0,40 do local a=i*math.pi/20;table.insert(outline,vector.create(50+30*math.cos(a),55+30*math.sin(a),0)) end
table.insert(strokes,outline);table.insert(colours,4)
for y=28,82,3 do
    local dx=math.sqrt(math.max(0,30^2-(y-55)^2))
    table.insert(strokes,{vector.create(50-dx,y,0),vector.create(50+dx,y,0)});table.insert(colours,4)
end
table.insert(strokes,{vector.create(50,25,0),vector.create(54,10,0)});table.insert(colours,3)
active,slice,slices=0,0,0
for i=1,8 do
    local job=R.beginRecognize(strokes,3,if i%3==0 then nil else colours)
    while not job:step(.75) do end
    active+=job.stats.activeMs;slice=math.max(slice,job.stats.maxSliceMs);slices+=job.stats.slices
end
print("RECOGNIZE8",active,slice,slices)
local complexJob=R.beginRecognize(complexStrokes,3,complexColours)
while not complexJob:step(.75) do end
print("COMPLEX_RECOGNIZE",complexJob.stats.activeMs,complexJob.stats.maxSliceMs,complexJob.stats.slices,complexPoints,#complexStrokes)
'''


def prepare(model,tag):
    dest=ROOT/'training/native_bench'/tag;dest.mkdir(parents=True,exist_ok=True)
    for name in ['Weights','TestVectors','DrawRecognizer','DrawRasterizer','DrawJob']:
        src=model/(name+'.luau')
        if tag=='baseline-cli' and name in ['DrawRecognizer','DrawRasterizer','DrawJob']:
            src=ROOT/'luau'/(name+'.luau')
        if not src.exists():src=ROOT/'luau'/(name+'.luau')
        s=src.read_text(encoding='utf-8')
        s=s.replace('require(script.Weights)','require("./Weights")').replace('require(script.DrawRasterizer.DrawJob)','require("./DrawJob")')
        s=s.replace('require(script.DrawRasterizer)','require("./DrawRasterizer")').replace('require(script.DrawJob)','require("./DrawJob")')
        s=s.replace('Vector3.new','vector.create').replace('Vector3.zero','vector.create(0,0,0)').replace('Vector3','vector')
        (dest/(name+'.luau')).write_text(s,encoding='utf-8',newline='\n')
    (dest/'bench.luau').write_text(SCRIPT,encoding='utf-8',newline='\n')
    return dest


def run(model,tag):
    folder=prepare(model,tag);result={}
    for mode,args in [('native',['--codegen']),('interpreted',[])]:
        proc=subprocess.run([str(EXE),'-O2',*args,str(folder/'bench.luau')],capture_output=True,text=True,timeout=180)
        if proc.returncode:raise RuntimeError(proc.stdout+'\n'+proc.stderr)
        rows={}
        for line in proc.stdout.splitlines():
            words=line.split()
            if words and words[0] in ['PARITY','FORWARD','BURST8','SLICE_PARITY','RECOGNIZE8','COMPLEX_RECOGNIZE']:rows[words[0]]=[float(x) for x in words[1:]]
        assert set(rows)=={'PARITY','FORWARD','BURST8','SLICE_PARITY','RECOGNIZE8','COMPLEX_RECOGNIZE'},'Incomplete native/VM check output'
        assert all(math.isfinite(v) for values in rows.values() for v in values),'Non-finite runtime result'
        _,logit,raster,norm,top_mismatches=rows['PARITY']
        assert logit<3e-4 and raster<1e-6 and norm<2e-4 and top_mismatches==0,'Python/Luau parity failed'
        assert rows['SLICE_PARITY'][0]<1e-6,'Interleaved jobs differ from synchronous forward'
        result[mode]=rows;print(tag,mode,rows,flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--tag',default='candidate')
    a=ap.parse_args()
    base=run(ROOT/'luau/models/colour299_40','baseline-cli')
    candidate=run(Path(a.candidate).resolve(),a.tag+'-cli')
    ratio=candidate['native']['FORWARD'][0]/base['native']['FORWARD'][0]
    cost=6.87*ratio
    queued=candidate['native']['BURST8'][0]*6.87/base['native']['FORWARD'][0]
    recognized=candidate['native']['RECOGNIZE8'][0]*6.87/base['native']['FORWARD'][0]
    result={'baseline':base,'candidate':candidate,'native_ratio':ratio,'calibrated_server_forward_ms':cost,
        'eight_guess_work_ms':cost*8,'eight_guess_minimum_wall_seconds_at_2ms_60hz':cost*8/120,
        'calibrated_eight_queued_active_ms':queued,
        'eight_queued_wall_seconds_at_2ms_60hz':queued/120,
        'queued_guesses_per_second_at_2ms_60hz':8*120/queued,
        'calibrated_eight_recognize_active_ms':recognized,
        'eight_recognize_wall_seconds_at_2ms_60hz':recognized/120,
        'caveat':'Offline Luau 0.746 CLI measurements under concurrent GPU training. Ratio calibrated against the prior Studio server median 6.87 ms. This is not a fresh Roblox server measurement; queue/raster/network overhead excluded from calibrated forward cost.'}
    out=ROOT/'notes'/f'{a.tag}-native.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print('RESULT',result['calibrated_server_forward_ms'],result['eight_guess_minimum_wall_seconds_at_2ms_60hz'],flush=True)


if __name__=='__main__':main()
