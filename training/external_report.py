"""Render the completed offline benchmark, with duplicate-aware headline scores."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from external_eval import ROOT,OUT,CATEGORIES,metrics,write_json

NAMES={'tu':'TU Berlin','clipart':'DomainNet clipart','painting':'DomainNet painting','sketch':'DomainNet sketch','qd_unrecognized':'Quick, Draw! unrecognised'}
MODE={'tu':'line','clipart':'preserve','painting':'preserve','sketch':'line','qd_unrecognized':'native'}
MODELS=('colour299_40','wide299_28_r4')
def pct(v):return '—' if v is None else f'{v*100:.2f}%'

def main():
    mapping=json.loads((OUT/'label-mapping.json').read_text())
    results={}
    for domain in NAMES:
        folder=OUT/domain;truth=np.load(folder/'truth.npy');valid=np.load(folder/'valid.npy')
        keep=np.load(folder/'include.npy') if (folder/'include.npy').exists() else np.ones(len(truth),bool)
        modes=('native',) if domain=='qd_unrecognized' else ('line','preserve')
        results[domain]={'primary_mode':MODE[domain],'raw_rows':len(truth),'included_unique_rows':int(keep.sum()),'excluded_duplicates':int((~keep).sum()),'models':{}}
        for model in MODELS:
            results[domain]['models'][model]={}
            for mode in modes:
                p=np.load(folder/f'{model}-{mode}-predictions.npz')
                results[domain]['models'][model][mode]=metrics(truth[keep],p['top3'][keep],valid[keep])
    write_json(ROOT/'notes/external-evaluation.json',results)
    lines=['# Outside recognition tests, 2026-11-10','',
      '**Offline evaluation only. No weights were trained or modified, no Studio calls were made, and no dataset images were shown in the game.**','',
      '## Headline results','',
      'All guesses compete against all 299 outputs. Accuracy denominators contain only mapped source labels; unmapped examples are excluded from accuracy and tracked separately. TU Berlin uses 20,000 distinct PNGs: the supplied copy had 80 duplicate `train`-category images. DomainNet uses its complete supplied official test lists; no other archive members were read. Quick, Draw! failures are a fixed prefix sample of up to 300 unique failed drawings per real class, not the complete downloaded failure population.','',
      '| Set | Mapped examples | Main conversion | 40px first / top three | r4 28px first / top three |','|---|---:|---|---:|---:|']
    for d,r in results.items():
        a=r['models'][MODELS[0]][MODE[d]]['all'];b=r['models'][MODELS[1]][MODE[d]]['all']
        lines.append(f'| {NAMES[d]} | {a["n"]:,} | {MODE[d]} | {pct(a["top1"])} / {pct(a["top3"])} | {pct(b["top1"])} / {pct(b["top3"])} |')
    lines+=['','These compare complete trained candidates, including different resolution, colour channels and training histories; they do not isolate a causal colour or resolution gain. [Complete per-class/mode results](external-evaluation.json), [complete mappings including every unmapped label](external-label-mapping.md).','',
      '## What went into the network','',
      '- TU Berlin and DomainNet sketch use a black line view: foreground threshold from alpha or distance from border-median paper colour, tight bounding crop, a maximum 256px working grid, skeletonization, then model-width soft strokes fitted to the trained centre-line box (22/28 of the input). Padding and radius scale with 28/40px. Original stroke order, ownership and brush size are unavailable.',
      '- Clipart and paintings use preserved foreground/fills and per-pixel colour. The same automatic foreground extraction estimates paper; coverage and premultiplied colour are area-resampled with 4x supersampling and centred into the expected outside-ink box (24/28 of the input). Colours use the nearest of the ten actual game inks, with black network RGB zero. Class colour priors are never consulted.',
      '- Each raster set also has the alternate conversion. For filled art its line control uses Canny edges plus silhouette boundary, rather than collapsing solid bodies to their medial skeleton. This control is black and loses colour/fill. No per-image best-view selection is used.',
      '- Raster masks cannot recover vector stroke centres exactly. Cropping, paper segmentation, thinning, edge extraction, unknown brush width, palette quantization and absent stroke layering all introduce bias. Detailed paintings can include backgrounds and textures that the clean game canvas does not. These are tests of model-plus-adapter, not direct player-drawing accuracy. Float16 input caches add at most about 0.00025 component rounding; inference uses the dequantized int8 weights in float32.',
      '- DomainNet is especially adapter-limited: inspected examples include watermark grids, text around an apple, a pianist plus piano, dense pencil shading, a conductor/music scene labelled piano, and a teddy bear under the generic bear label. At 40px, thickened watermark/shading edges can swamp the object. The edge control does not isolate an object either. Its low scores must not be presented as clean filled-player-drawing accuracy. No label-informed object crops or source-image selection were used.',
      '- Failed/blank conversions count as misses for mapped examples; they are not silently removed. Source HF `train` and `test` shards were pooled only as an external evaluation set; neither was used to train our models.','',
      '## Conversion sensitivity','', '| Set | 40px line first / top three | 40px preserved first / top three |','|---|---:|---:|']
    for d,r in results.items():
        if d=='qd_unrecognized':continue
        a=r['models'][MODELS[0]]['line']['all'];b=r['models'][MODELS[0]]['preserve']['all']
        lines.append(f'| {NAMES[d]} | {pct(a["top1"])} / {pct(a["top3"])} | {pct(b["top1"])} / {pct(b["top3"])} |')
    rgb_path=ROOT/'notes/external-rgb-control.json'
    if rgb_path.exists():
        rgb=json.loads(rgb_path.read_text())
        lines+=['','### Exploratory colour control','',
          'Visual QA found that RGB nearest-palette matching can map dark red to brown. A separate post-QA control preserves continuous source RGB instead (near-black pixels use the black-network convention), with **identical ink geometry**. This was added after inspecting the adapter, is outside the discrete training palette, and was not used to choose headline results. No labels or class priors are consulted.','',
          '| Set | Continuous RGB first / top three |','|---|---:|']
        for d,r in rgb.items():lines.append(f'| {NAMES[d]} | {pct(r["all"]["top1"])} / {pct(r["all"]["top3"])} |')
        lines+=['','The sources also contain typography, watermarks, full-body poses and multi-object scenes; the automatic crop does not isolate the labelled object. [Colour control details](external-rgb-control.json).']
    lines+=['','## Worst classes and custom-output false alarms','',
      'False alarms below mean a mapped **real-class** example whose first answer is one of the 29 custom classes. Correct custom-class targets are excluded from that denominator. Unsupported labels are not scored as recognition errors; their forced routing is retained in JSON. All class sample counts are shown so tiny classes are not mistaken for reliable estimates.','']
    for d,r in results.items():
        m=r['models'][MODELS[0]][MODE[d]];baseline=r['models'][MODELS[1]][MODE[d]]
        fa=m['real_to_custom'];fb=baseline['real_to_custom']
        lines+=['### '+NAMES[d],'',f'40px real-to-custom: **{fa["n"]:,}/{fa["denominator"]:,} ({pct(fa["rate"])})**; r4: {fb["n"]:,}/{fb["denominator"]:,} ({pct(fb["rate"])}).',
            'Worst destinations: '+(', '.join(f'{name} {count:,} ({pct(count/fa["denominator"])})' for name,count in list(fa['destinations'].items())[:5]) or 'none')+'.','',
            '| Worst 40px labels | N | First | Top three |','|---|---:|---:|---:|']
        worst=sorted(m['classes'],key=lambda c:(c['top1'],-c['n']))[:8]
        lines += [f'| {c["label"]} | {c["n"]} | {pct(c["top1"])} | {pct(c["top3"])} |' for c in worst]
        lines += ['',f'Macro average by target class: {pct(m["macro_top1"])} / {pct(m["macro_top3"])}. Mapped conversion failures: {m["conversion_failures_mapped"]}.','']
        if m['custom']['n']:lines += [f'Actual mapped custom-class targets: {m["custom"]["n"]}; accuracy {pct(m["custom"]["top1"])} / {pct(m["custom"]["top3"])}. This is limited custom-vocabulary coverage, not a test of all 29.','']
    detail=json.loads((ROOT/'notes/external-detail-sanity.json').read_text())
    lines+=['## Confidence and detail sanity check','',
      f'On {detail["shared_classes"]} exact-name shared real classes, with {detail["tu_n"]:,} unique TU examples and {detail["quickdraw_n"]:,} recognised Quick, Draw! held-outs, the class-balanced mean **correct-class probability** is {pct(detail["macro_true_probability"]["tu"])} for TU versus {pct(detail["macro_true_probability"]["quickdraw"])} for Quick, Draw!. This is probability, not accuracy or a human quality rating.', '',
      '| Image-compatible proxy at common 40px input resolution | TU Berlin | Quick, Draw! |','|---|---:|---:|']
    for name in detail['macro_features']['tu']:
        lines.append(f'| {name} | {detail["macro_features"]["tu"][name]:.3f} | {detail["macro_features"]["quickdraw"][name]:.3f} |')
    lines+=['', 'TU has more junction clusters and ink coverage here, while recognition confidence is lower. This does **not** mean its art is worse. Familiarity with the training distribution, Google’s recognised-only selection, conversion differences and information lost at 40px confound the comparison. Skeleton length barely distinguishes the sets at this resolution. Connected components are not pen strokes; the image proxies cannot provide original stroke order/count, drawing duration or a legitimate last-30%-ink ablation. No GACL quality norm exists in the current classifier.','',
      'Do not fit effect tiers to these dataset identities. The author’s Rough/Good/Detailed ratings remain the required first ground truth; no quality score or gameplay behaviour was implemented.', '',
      '## Training exclusion and limitations','',
      '`training/prepare.py`, `expand_data.py` and `prepare_colour.py` all filter out `recognized: false`. The installed colour model inherits real-class paths from those filtered corpora. Every sampled failed drawing ID was also checked against the pinned train, validation and test IDs and had zero overlap. They are eligible test examples. Common or near-identical shapes can still occur under different IDs; this was not a 26-million-example geometry deduplication.',
      'The failure cohort is deliberately difficult and can include incomplete, misleading or wrongly drawn responses to a known prompt. It is not ordinary Quick, Draw! IID accuracy. Its prefix sampling and the original partial raw downloads are additional limits.',
      'Exact PNG duplicates were checked for TU Berlin. Cross-dataset near-duplicates, artist overlap, and DomainNet provenance overlap are not ruled out; do not pool the sets into one independence claim. DomainNet licence remains unconfirmed: offline local testing only, with no training, distribution or in-game display. TU Berlin attribution/licence follows the supplied HF card (CC BY 4.0); Quick, Draw! credit remains Google, CC BY 4.0. [Sources and restrictions](../data/external/SOURCES.md).','',
      'The confidence/detail check is separate and exploratory: [matched-class sanity check](external-detail-sanity.json). It does not fit effect tiers or replace the author’s Rough/Good/Detailed ratings. No gameplay quality score, GACL training or anti-scribble second pass was added.','']
    (ROOT/'notes/external-evaluation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    fig,ax=plt.subplots(figsize=(11,6.5));fig.subplots_adjust(left=.25,right=.98,top=.90,bottom=.15);y=np.arange(len(results));w=.33
    for j,model in enumerate(MODELS):
        values=[results[d]['models'][model][MODE[d]]['all'] for d in results]
        yy=y+(j-.5)*w
        ax.barh(yy,[v['top1']*100 for v in values],height=w*.85,label='40px colour/fill: first' if j==0 else 'r4 28px: first',color='#e97a36' if j==0 else '#697d94')
        ax.scatter([v['top3']*100 for v in values],yy,s=35,color='#a44216' if j==0 else '#253c56',marker='|',label='40px top three' if j==0 else 'r4 top three')
        for yy0,v in zip(yy,values):ax.text(max(v['top1'],v['top3'])*100+.8,yy0,f'{v["top1"]*100:.1f} / {v["top3"]*100:.1f}',va='center',fontsize=8)
    ax.set_yticks(y,[NAMES[d] for d in results]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Correct (%) — mapped labels, all 299 outputs compete');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    ax.set_title('Outside drawing tests: accuracy depends on the domain and raster conversion',loc='left',fontsize=13,pad=15)
    ax.legend(loc='lower right',fontsize=9,ncol=2)
    fig.text(.01,.025,'Offline testing only; not player accuracy. DomainNet has major adapter/background bias. No training.',fontsize=9,color='#555')
    fig.savefig(ROOT/'notes/external-evaluation.png',dpi=160);plt.close(fig)
    print('\n'.join(lines[:20]))

if __name__=='__main__':main()
