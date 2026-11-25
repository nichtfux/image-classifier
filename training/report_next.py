"""Write the offline candidate report and a metrics-only figure (no dataset art)."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parent.parent;NOTES=ROOT/'notes';OUT=ROOT/'training/out'
NAME='gacl299_40_wide';BASE='colour299_40'


def pct(v):return f'{100*v:.2f}%'
def pair(row):return f"{100*row['top1']:.2f} / {100*row['top3']:.2f}"
def top(row):return ', '.join(f"{r['label']} {100*r['prob']:.1f}%" for r in row['top3'])


def paired_ci(y,old,new):
    a=old[:,0]==y;b=new[:,0]==y;d=b.astype(float)-a
    delta=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(len(d)))
    improved=int((b&~a).sum());regressed=int((a&~b).sum())
    p=float(binomtest(improved,improved+regressed,.5).pvalue) if improved+regressed else 1.
    return {'n':len(d),'delta_top1':delta,'approx_95ci':[delta-1.96*se,delta+1.96*se],
            'improved':improved,'regressed':regressed,'mcnemar_exact_p':p}


def main():
    report=json.loads((NOTES/'training-round-next.json').read_text());b=report[BASE];n=report[NAME]
    native=json.loads((NOTES/f'{NAME}-native.json').read_text())
    selected=json.loads((OUT/f'{NAME}-selection.json').read_text())
    domains=[('tu','TU Berlin'),('qd_unrecognized','QD unrecognised'),('clipart','DomainNet clipart'),('painting','DomainNet painting'),('sketch','DomainNet sketch')]
    modes=[('black','Black'),('policy','Colourise'),('random','Random colours'),('filled','Filled'),('painted','Brush / scribble')]
    lines=['# Next recognition training round — 2026-11-10','',
        f"Candidate: **{NAME}**, selected {selected['selected']['phase']}. **Not installed.** The live model remains **{BASE}**. No Studio calls, place writes, saves or publishing were made.",'',
        f"Black-drawing acceptance against both the current model and r4: **{'PASS' if n.get('black_acceptance_guard') else 'FAIL'}** (real/custom top-1 and top-3, at most 0.2 percentage points down). Selection was completed on permitted validation before running candidate external tests.",'',
        '## Quick, Draw! and original custom tests','',
        'Top-1 / top-3, percent. Same pinned test IDs; paired colour/fill seeds and 512-drawing batches. Filled and brush tests are synthetic transformations, not real coloured-player accuracy. The baseline was recomputed on exactly these inputs; rounding and batching can produce small differences from earlier reports.','',
        '| Input | Installed, all 299 | Candidate, all 299 | Installed, 270 real | Candidate, 270 real |',
        '|---|---:|---:|---:|---:|']
    for mode,label in modes:
        a=b['quickdraw'][mode];c=n['quickdraw'][mode]
        lines.append(f"| {label} | {pair(a['all'])} | {pair(c['all'])} | {pair(a['real'])} | {pair(c['real'])} |")
    lines += ['', '## Custom-class generalisation','',
        'The original synthetic test is a very optimistic upper bound. The new procedural test uses disjoint seeds and held-out view=2 branches; some branches share geometry with view=1. Only TU Berlin pistol/revolver and grenade supply real human examples of our custom labels. Nothing here establishes real-player accuracy for all 29 labels.','',
        '| Custom test | N | Installed top-1 / top-3 | Candidate top-1 / top-3 |','|---|---:|---:|---:|']
    for mode,label in [('black','Original synthetic, black'),('filled','Original synthetic, filled')]:
        a=b['quickdraw'][mode]['synthetic'];c=n['quickdraw'][mode]['synthetic']
        lines.append(f"| {label} | {a['drawings']:,} | {pair(a)} | {pair(c)} |")
    a=b['diverse_synthetic']['all'];c=n['diverse_synthetic']['all']
    lines.append(f"| New procedural views (upper bound) | {a['n']:,} | {pair(a)} | {pair(c)} |")
    for key in ['custom']:
        a=b['external']['tu']['modes']['line'][key];c=n['external']['tu']['modes']['line'][key]
        lines.append(f"| TU Berlin custom labels | {a['n']:,} | {pair(a)} | {pair(c)} |")
    for name in ['pistol','grenade']:
        a=next(r for r in b['external']['tu']['modes']['line']['classes'] if r['label']==name)
        c=next(r for r in n['external']['tu']['modes']['line']['classes'] if r['label']==name)
        lines.append(f"| TU Berlin {name} | {a['n']} | {pair(a)} | {pair(c)} |")
    lines += ['', '## Outside sets — testing only','',
        'Same frozen label mappings and adapters as [the external evaluation](external-evaluation.md); unmapped names are listed in [the mapping report](external-label-mapping.md). TU uses unique images (80 repeated images excluded). Quick, Draw! recognised=false rows were excluded from all training and are disjoint from the pinned train/validation/test IDs; their prompts are noisy labels. DomainNet results are raster-domain stress tests, not game accuracy.','',
        '| Set / primary view | Mapped N | Installed top-1 / top-3 | Candidate top-1 / top-3 |','|---|---:|---:|---:|']
    for domain,label in domains:
        mode=n['external'][domain]['primary_mode'];a=b['external'][domain]['modes'][mode];c=n['external'][domain]['modes'][mode]
        lines.append(f"| {label}, {mode} | {a['all']['n']:,} | {pair(a['all'])} | {pair(c['all'])} |")
    lines += ['', 'Raster conversion: estimate foreground from transparency/border background; crop the foreground bounding box, preserve aspect and padding. TU/sketch primary views skeletonise and redraw soft thin strokes at the trained width. Clipart/painting primary views preserve coverage and quantise colour to the ten inks. Both line and preserved-coverage controls are in the JSON. Thinning destroys shading and small details; foreground extraction and background/colour quantisation introduce additional biases. All adapters stayed fixed. DomainNet licence remains unconfirmed, so its images remain offline test-only. No outside-set images or failed-QD rows were used for training, model selection, or shown in the game.','',
        '## False alarms into custom classes','',
        'Top-1 custom output on mapped real (first 270) ground-truth classes. Denominators exclude true custom and unmapped labels. Percent of real drawings, not percent of mistakes.','',
        '| Set | Installed any custom | Candidate any custom | Installed black hole | Candidate black hole | Candidate worst destinations (count) |','|---|---:|---:|---:|---:|---|']
    alarms=[]
    for mode,label in [('black','QD black'),('filled','QD filled'),('painted','QD brush / scribble')]:
        a=b['quickdraw'][mode];c=n['quickdraw'][mode];worst=', '.join(f"{r['label']} ({r['count']})" for r in c['false_alarms']['worst'][:3])
        lines.append(f"| {label} | {pct(a['false_alarms']['rate'])} | {pct(c['false_alarms']['rate'])} | {pct(a['black_hole_false_alarms']['rate'])} | {pct(c['black_hole_false_alarms']['rate'])} | {worst} |")
    for domain,label in domains:
        mode=n['external'][domain]['primary_mode'];a=b['external'][domain]['modes'][mode];c=n['external'][domain]['modes'][mode]
        worst=', '.join(f'{name} ({count})' for name,count in list(c['real_to_custom']['destinations'].items())[:3])
        lines.append(f"| {label} | {pct(a['real_to_custom']['rate'])} | {pct(c['real_to_custom']['rate'])} | {pct(a['black_hole_false_alarms']['rate'])} | {pct(c['black_hole_false_alarms']['rate'])} | {worst} |")
    lines += ['', '## Retained stroke regressions','',
        'Independent authored developer gestures, never trained on or used for selection. Replayed offline through the same 40px stroke/owner rasteriser; no Studio interaction this turn. Probabilities are uncalibrated and should not be compared as quality scores.','',
        '| Drawing | Installed top-3 | Candidate top-3 |','|---|---|---|']
    old={r['name']:r for r in b['scripted_probes']}
    for row in n['scripted_probes']:lines.append(f"| {row['name']} | {top(old[row['name']])} | {top(row)} |")
    lines += ['', '## GACL quality check','',
        'Implemented the cosine-margin variant of [Yang et al. GACL](https://arxiv.org/html/2507.20548v1): unit class weights/features, scale 64, target logit minus q, q = clipped linear rescaling of feature L2 norm in [0.35, 0.8], regulariser 0.2 × (1/q + q/0.8²). Early fixed soft bins are used for the first quarter of the first GACL epoch. The raw pre-normalisation feature norm is available at inference; no human rating was used as a training target.','',
        '170 exact-name real classes, 80 unique TU and 80 held-out recognised-QD drawings per class (13,600 each). Class weighting and the QD sample match the earlier quality sanity check.','',
        '| Norm check | Installed raw feature norm (control) | GACL candidate |','|---|---:|---:|']
    qb=b['quality'];qn=n['quality']
    for label,key in [('Mean TU norm','tu_mean_norm'),('Mean QD norm','qd_mean_norm'),('Within-class TU > QD AUC','macro_tu_over_qd_auc'),('Classes with higher TU mean','tu_higher_classes')]:
        lines.append(f"| {label} | {qb[key]:.4f} | {qn[key]:.4f} |")
    for dataset in ['tu','quickdraw']:
        lines.append(f"| Within-class norm / ink Spearman, {dataset} | {qb['within_class_norm_ink_spearman'][dataset]:.3f} | {qn['within_class_norm_ink_spearman'][dataset]:.3f} |")
    if qn['tu_mean_norm']<qn['qd_mean_norm']:
        lines += ['', '**The requested TU-higher quality sanity check fails.** The GACL norm favours QD on average here. It must not be presented as a validated artistic-quality score.']
    lines += ['', '**This does not validate an artistic-quality score.** Even the old, non-GACL model ranks TU above QD and correlates with ink coverage. Dataset identity, stroke density, recognisability, and TU raster conversion remain confounds. The author’s Rough / Good / Detailed ratings remain the ground truth for later within-class quality checks and tier fitting. No tier, threshold, detail effect or game change was built.','',
        '## Native cost and queue budget','',
        '**No fresh Roblox server measurement was possible or attempted:** this turn explicitly forbade Studio. Timings below are measured with the official Luau 0.746 CLI on this PC while training ran. The forward ratio is calibrated against the prior Studio server median of 6.87 ms for colour299_40; calibrated values are estimates.','',
        '| Offline measurement | Installed architecture | Candidate |','|---|---:|---:|']
    for mode,label in [('native','Native'),('interpreted','Interpreted')]:
        x=native['baseline'][mode];y=native['candidate'][mode]
        lines.append(f"| {label} forward median / p95, ms | {x['FORWARD'][0]:.2f} / {x['FORWARD'][1]:.2f} | {y['FORWARD'][0]:.2f} / {y['FORWARD'][1]:.2f} |")
        lines.append(f"| {label} worst 0.75 ms job slice, ms | {x['RECOGNIZE8'][1]:.2f} | {y['RECOGNIZE8'][1]:.2f} |")
    lines += ['', f"Estimated server forward: **{native['calibrated_server_forward_ms']:.2f} ms**, under the ~100 ms budget. Eight scripted filled/coloured recognitions including rasterisation/checkpoints are estimated at **{native['calibrated_eight_recognize_active_ms']:.1f} ms active work**, or **{native['eight_recognize_wall_seconds_at_2ms_60hz']:.2f} s at 2 ms × 60 Heartbeats/s**, plus remote/scheduler overhead. Forward-only cooperative throughput is about **{native['queued_guesses_per_second_at_2ms_60hz']:.1f} guesses/s**. Lower server frame rates or bursts beyond eight extend this time; a full 32-job burst could exceed the existing six-second deadline. A fresh native server/burst measurement is required before any swap.",'',
        'The wider 40px network uses 48/96/192 convolution channels and 384 dense features, versus 24/48/96 and 288. It requires ~37.90M multiply-accumulates versus ~10.45M. Spending the entire 100 ms on each guess would leave only about 1.2 guesses/s at this queue budget; this size leaves headroom for bursts and later quality checks.','',
        '## Training and compatibility','',
        'Training used only recognised=true Quick, Draw! and our own procedural generation: 26,263,888 recognised real training drawings, 783,000 original procedural drawings, and 87,000 additional procedural drawings. The new drawings are oversampled to bring custom-class exposure nearer the real-class exposure. The first 299 labels and all old split IDs stay unchanged. The fallback model/artifacts remain intact.','',
        'Function-preserving channel widening starts from colour299_40, followed by a half-pass classic warm-up and GACL refinement. Colour is resampled per training pass. Augmentations cover solid/interrupted fills, parallel and crossed scribble gaps, overpainting, thin/broad strokes, soft brush edges, small coordinate changes and some directly painted silhouettes. New custom programs add simpler icons, different viewpoints and more detailed variants, including minimal top-fuse bombs, revolvers, different grenade bodies and black holes with a dark core plus a disk/curved arms. New variants are procedural and have not yet been rated by the author. The real-class loss also penalises black-hole confusion.','',
        'Input stays four 40×40 planes (ink, ink×r/g/b), with the same nearest-stroke ownership, palette, crop and padding. Colour remains optional. The dev runtime adds final-feature normalisation and exposes raw qualityNorm; the first four recognise returns and preview layout stay unchanged (40, padded 42, planeSize 1764, inputChannels 4). Generated DrawModelInfo still contains only public raster/palette data. No DrawTest/DrawCollect/layout edits were made.','',
        'Dev artifacts: `luau/models/gacl299_40_wide/`; selected checkpoint and selection record in `training/out/`. Reproduce with `training/pipeline_next.py`. Weights/TestVectors/DrawModelInfo are written by `training/export.py`; `candidate_runtime.py` generates the uninstalled compatible runtime. Python/Luau float64-logit, colour-plane, feature-norm, synchronous/sliced and interleaved-job checks are ready. Fresh Studio Verify, actual server/round-trip timing, and the author’s live/rated drawings are **not checked**.','',
        'Quick, Draw! training attribution remains Google, CC BY 4.0. TU is test-only CC BY 4.0; DomainNet remains offline test-only with unconfirmed licence. Dataset or generated training drawings were not shown in the game.','',
        'Full per-class scores, alternate raster views, false-alarm counts, probes and quality controls: [results JSON](training-round-next.json). Timing/parity details: [native JSON](gacl299_40_wide-native.json).']
    # Practical regressions and paired evidence, rather than just an aggregate win.
    oldclasses=b['quickdraw']['black']['classes'];newclasses=n['quickdraw']['black']['classes']
    regressions=sorted([(z['top1']-a['top1'],a['label'],a['top1'],z['top1']) for a,z in zip(oldclasses[:270],newclasses[:270])])[:10]
    lines += ['', '## Largest real-class changes on black held-outs','', '| Class | Installed top-1 | Candidate top-1 | Change, pp |','|---|---:|---:|---:|']
    for delta,label,a,z in regressions:lines.append(f'| {label} | {pct(a)} | {pct(z)} | {delta*100:+.2f} |')
    # Primary paired differences, all computed after candidate selection.
    truth=np.load(ROOT/'data/colour299-40/test-y.npy',mmap_mode='r');paired={}
    for mode in ['black','filled','painted']:
        a=np.load(OUT/f'{BASE}-next-{mode}-predictions.npz')['top3'];z=np.load(OUT/f'{NAME}-next-{mode}-predictions.npz')['top3']
        use=truth<270;paired['QD '+mode]=paired_ci(truth[use],a[use],z[use])
    for domain,label in domains:
        mode=n['external'][domain]['primary_mode'];folder=ROOT/'data/external_eval'/domain
        truth=np.load(folder/'truth.npy');use=truth>=0
        if (folder/'include.npy').exists():use&=np.load(folder/'include.npy')
        a=np.load(OUT/f'{BASE}-next-{domain}-{mode}.npz')['top3'];z=np.load(OUT/f'{NAME}-next-{domain}-{mode}.npz')['top3']
        paired[label]=paired_ci(truth[use],a[use],z[use])
    (NOTES/'training-round-next-paired.json').write_text(json.dumps(paired,indent=2),encoding='utf-8',newline='\n')
    lines += ['', 'Paired top-1 differences and discordance counts are in [paired results](training-round-next-paired.json). Their image-level intervals describe these fixed tests, not uncertainty across future players.','']
    (NOTES/'training-round-next.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')

    fig,axes=plt.subplots(3,2,figsize=(16,14));oldcolor='#687788';newcolor='#e27629'
    def bars(ax,labels,old,new,title,ylim=None,fmt='%.1f'):
        x=np.arange(len(labels));w=.36
        p=ax.bar(x-w/2,np.array(old)*100,w,label='Installed colour299_40',color=oldcolor)
        q=ax.bar(x+w/2,np.array(new)*100,w,label='Candidate GACL wide',color=newcolor)
        ax.set_xticks(x,labels);ax.set_title(title,loc='left',fontweight='bold');ax.set_ylabel('Top-1 (%)')
        for bars_ in [p,q]:ax.bar_label(bars_,fmt=fmt,padding=3,fontsize=10)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
        if ylim:ax.set_ylim(*ylim)
    bars(axes[0,0],[label for _,label in modes],[b['quickdraw'][m]['real']['top1'] for m,_ in modes],[n['quickdraw'][m]['real']['top1'] for m,_ in modes],'A  Recognised Quick, Draw! — 270 real classes',(0,100))
    labels=['TU Berlin','QD failed','Clipart','Painting','Sketch']
    basev=[];newv=[]
    for domain,_ in domains:
        mode=n['external'][domain]['primary_mode'];basev.append(b['external'][domain]['modes'][mode]['all']['top1']);newv.append(n['external'][domain]['modes'][mode]['all']['top1'])
    bars(axes[0,1],labels,basev,newv,'B  Outside tests — fixed raster adapters',(0,100));axes[0,0].legend(fontsize=10,loc='lower left')
    bars(axes[1,0],['Original synth','New synth views','TU pistol + grenade'],[b['quickdraw']['black']['synthetic']['top1'],b['diverse_synthetic']['all']['top1'],b['external']['tu']['modes']['line']['custom']['top1']],
        [n['quickdraw']['black']['synthetic']['top1'],n['diverse_synthetic']['all']['top1'],n['external']['tu']['modes']['line']['custom']['top1']],'C  Custom classes — synth scores are upper bounds',(0,110))
    bhb=[];bhn=[]
    for domain,_ in domains:
        mode=n['external'][domain]['primary_mode'];bhb.append(b['external'][domain]['modes'][mode]['black_hole_false_alarms']['rate']);bhn.append(n['external'][domain]['modes'][mode]['black_hole_false_alarms']['rate'])
    bars(axes[1,1],labels,bhb,bhn,'D  Black-hole false alarms on real mapped classes',fmt='%.2f');axes[1,1].set_ylabel('Real drawings predicted black hole (%)')
    norms=np.load(OUT/f'{NAME}-quality-comparison.npz');ax=axes[2,0]
    for key,label,color in [('tu','TU Berlin careful sketches','#bd593e'),('quickdraw','Quick, Draw! held-outs','#377da1')]:
        v=np.sort(norms[key]);ax.plot(v,np.arange(1,len(v)+1)/len(v),label=label,color=color,linewidth=2)
    ax.set_title('E  GACL feature norm — human ratings still required',loc='left',fontweight='bold');ax.set_xlabel('Raw feature L2 norm');ax.set_ylabel('Cumulative share');ax.legend(fontsize=10);ax.grid(alpha=.2)
    ax.text(.02,.05,f"TU > QD within-class AUC: {qn['macro_tu_over_qd_auc']:.3f}\nOld raw-norm control: {qb['macro_tu_over_qd_auc']:.3f}\nDataset identity is not a quality label.",transform=ax.transAxes,fontsize=11)
    ax=axes[2,1];ax.axis('off')
    text=(f"F  Runtime budget and status\n\n40px · 4 colour planes · 299 labels\n48/96/192 conv · 384 features · int8\n\nNative CLI median: {native['candidate']['native']['FORWARD'][0]:.1f} ms\nCalibrated server forward: ~{native['calibrated_server_forward_ms']:.1f} ms (estimate)\n8 guesses at 2 ms / 60 Hz: ~{native['eight_recognize_wall_seconds_at_2ms_60hz']:.2f} s + remote overhead\nWorst native CLI slice: {native['candidate']['native']['RECOGNIZE8'][1]:.2f} ms\n\nNot installed. No Studio calls.\nFresh server timing and the author’s rated drawings pending.")
    ax.text(.02,.98,text,va='top',fontsize=13,linespacing=1.5)
    fig.suptitle('Recognition training candidate — paired offline comparison',fontsize=21,fontweight='bold',y=.995)
    fig.text(.5,.97,'Test-only outside sets never used for training or checkpoint selection. No player-quality tiers fitted.',ha='center',fontsize=12)
    fig.tight_layout(rect=(0,.01,1,.95),h_pad=3,w_pad=3)
    fig.savefig(NOTES/'training-round-next.png',dpi=160,facecolor='white');plt.close(fig)
    print(NOTES/'training-round-next.md');print(NOTES/'training-round-next.png')


if __name__=='__main__':main()
