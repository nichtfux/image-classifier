"""Metrics and kept/rejected game-format samples for the local-model pilot."""
import json,math,time,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent;DATA=ROOT/'data/generated';NOTES=ROOT/'notes';OUT=ROOT/'training/out'
MODELS=['gacl299_40_continued','gacl299_40_genhard','gacl299_40_genkd']
LABELS=['No generated data','Generated, hard labels','Generated + SigLIP']
COLORS=['#526c8c','#d78032','#36876a']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def pair(r):return f"{100*r['top1']:.2f} / {100*r['top3']:.2f}"
def percent(x):return f'{x*100:.3f}%'


def record_students(names):
    records=[]
    for name in names:
        paths=[OUT/(name+suffix) for suffix in ['.pt','.npz','.json']]
        paths+=sorted((ROOT/'luau/models'/name).glob('*.luau'))
        records.append({'name':name,'installed_this_round':False,'files':[
            {'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in paths if p.exists()]})
    report={'training_provenance':'Google Quick, Draw! CC BY 4.0; own procedural drawings; generated-data branches additionally use local FLUX.1-schnell outputs and frozen SigLIP soft labels, source licences/revisions above.',
            'models':records}
    (DATA/'student-artifacts.json').write_text(json.dumps(report,indent=2),encoding='utf-8',newline='\n')
    source=DATA/'SOURCES.md';text=source.read_text(encoding='utf-8')
    if '## Student research artifacts' not in text:
        text+='\n## Student research artifacts\n\n[Student artifact hashes](student-artifacts.json) record the checkpoints, int8 arrays, metadata and dev-only Luau exports for the control and matched branches. These are uninstalled research artifacts. Quick, Draw! attribution remains required; training provenance is recorded separately from the foundation-model licences.\n'
        source.write_text(text,encoding='utf-8',newline='\n')


def main():
    scores=read(NOTES/'training-round-next.json');pilot=read(DATA/'pilot-generation-status.json')
    snap=read(DATA/'pilot-training-snapshot.json');training=read(DATA/'ablation-training.json')
    benchmark=read(DATA/'benchmark-resident.json');native=read(NOTES/'gacl299_40_genkd-native.json')
    process=read(DATA/'overnight-process.json')
    from itertools import islice
    with (DATA/'manifest.jsonl').open(encoding='utf-8') as f:
        pilot_rows=[json.loads(l) for l in islice(f,pilot['attempted']) if l.strip()]
    rate=pilot['kept_per_second'];target=29*5000+270*100
    live=read(DATA/'generation-status.json');live_rate=live.get('kept_per_second',0.)
    live_days=(target-live['kept'])/live_rate/86400 if live_rate>0 else None
    projected=(target-pilot['kept'])/rate/86400 if rate>0 else None
    zero=[r['label'] for r in pilot['classes'][270:] if r['attempted'] and not r['kept']]
    split_counts=Counter((r['word'],r.get('split')) for r in pilot_rows if r['keep'])
    missing_train=[r['label'] for r in pilot['classes'][270:] if not split_counts[(r['label'],'train')]]
    timing=native['candidate'];worst_native=max(v[1] for k,v in timing['native'].items() if k in ['BURST8','RECOGNIZE8','COMPLEX_RECOGNIZE'])
    style_lines=['## Pilot acceptance by style','','| Style | Attempted | Kept | Keep rate |','|---|---:|---:|---:|']
    for style in ['marker','outline','flat','painted','kids']:
        cohort=[r for r in pilot_rows if r['style']==style];kept=sum(r['keep'] for r in cohort)
        style_lines.append(f"| {style} | {len(cohort)} | {kept} | {percent(kept/len(cohort)) if cohort else 'not attempted'} |")
    reasons=Counter(r['reason'] for r in pilot_rows if not r['keep'])
    style_lines += ['', 'Rejected counts by reason: '+', '.join(f'{reason}: {count}' for reason,count in reasons.most_common())+'.','']
    matched=scores[MODELS[0]]
    outcome_lines=['## Outcome','','Nothing should be swapped into NN test from this pilot alone. Passing the aggregate black-accuracy guard does not make a model safe if real drawings are reassigned to custom outputs.','']
    for name,label in zip(MODELS[1:],LABELS[1:]):
        s=scores[name]
        weapon_delta=100*(s['external']['tu']['modes']['line']['custom']['top1']-matched['external']['tu']['modes']['line']['custom']['top1'])
        filled_delta=100*(s['quickdraw']['filled']['real']['top1']-matched['quickdraw']['filled']['real']['top1'])
        a=100*matched['quickdraw']['black']['false_alarms']['rate'];b=100*s['quickdraw']['black']['false_alarms']['rate']
        outcome_lines.append(f"- **{label}:** TU pistol/grenade {weapon_delta:+.2f} points; real filled test {filled_delta:+.2f} points versus the matched no-generated control. Real black QD drawings predicted as custom outputs change from **{a:.3f}% to {b:.3f}%**.")
    outcome_lines += ['', 'The broad-style control is a stronger starting point for the private drawing evaluation. The generated-data pilot tests a small, uneven bank, including heavily repeated rare-class prototypes. Broader generation continues, but more images alone are not a demonstrated cure: the next training protocol needs to constrain real-to-custom errors on permitted QD/own-generated validation and balance generated real negatives against custom positives. No further weights were tuned against the outside test results in this report.','']
    lines=['# FLUX / SigLIP generated-data pilot — 2026-11-10','',
        '**Research candidates only; nothing installed. No Studio calls.** The live colour299_40 model remains unchanged. The original bigger GACL/style control finished separately before these matched ablations.','',
        *outcome_lines,
        f"Pilot: **{pilot['attempted']:,} generated images, {pilot['kept']:,} kept ({100*pilot['kept']/max(1,pilot['attempted']):.1f}%)**. Frozen splits: **{len(snap['rows'])} train / {len(snap['validation_ids'])} validation / {len(snap['test_ids'])} test**. Split assignment hashes the class and complete prompt, independent of teacher acceptance: all seeds using the same prompt stay together. Model selection and training never use TU Berlin, DomainNet, QD recognised=false, or the author’s held-out drawings.",'',
        '## What ran locally','',
        '- Generator: Black Forest Labs FLUX.1-schnell, pinned official revision, Apache-2.0; original weights locally quantised to NF4/double quant with bfloat16 compute for this 12 GB GPU.',
        '- Teacher: Google SigLIP SO400M/384, pinned official revision, Apache-2.0, frozen zero-shot. It was not fine-tuned: retaining its broader pretraining avoids forcing it back toward Quick, Draw! style, and avoids using outside tests as training data.',
        '- Source revisions, canonical download URLs, original and locally derived hashes, component licences and software versions are in [SOURCES.md](../data/generated/SOURCES.md). The FLUX text encoders are packaged CLIP-L and T5-XXL; the VAE is also local.',
        '- The author accepted the official access conditions himself. Authentication used huggingface_hub normal token lookup; no token was printed, copied into the project or included in manifests. No paid API or online image service was used.','',
        'All raw and converted PNGs carry generated=true metadata. Manifest rows also record model revision, seed, prompt, style, class, source hash, conversion version, both teacher decisions and keep/reject reason. Samples, soft labels and strokes stay offline and are never shown to players.','',
        '## Measured speed and remaining work','',
        '| Size | Batch | Steps | Measured generated images/s | Full benchmark pipeline images/s | Kept / tested |','|---|---:|---:|---:|---:|---:|']
    for r in benchmark:lines.append(f"| {r['size']} | {r['batch']} | {r['steps']} | {r['generation_images_per_second']:.3f} | {r['pipeline_images_per_second']:.3f} | {r['kept']} / {r['images']} |")
    lines += ['', 'The benchmark uses the same bomb prompt/seeds across settings, with cached text embeddings. Its small keep sample is not a class-wide quality estimate. The pilot is the better end-to-end measurement: it includes varied prompts, encoding, generation, teacher transfers/inference, stroke conversion, saving and filtering.',
        f"Pilot throughput: **{pilot['images_per_second']:.3f} attempted images/s; {rate:.3f} kept/s** over **{pilot['session_seconds']/60:.1f} minutes**. Counts were set at the lower requested target, **5,000 kept/custom class**, plus **100 filled/painted images/real class** (172,000 kept total, split 80/10/10).",
        f"At the pilot aggregate keep rate, the remaining aggregate work projects to **{projected:.1f} days continuously** (about {projected*3:.0f} eight-hour nights). This is an extrapolation, not a completion guarantee: prompt-cache reuse may improve throughput, class keep rates vary, and zero-acceptance classes have no finite class-specific ETA." if projected is not None else 'No finite ETA yet.',
        f"The latest worker snapshot has **{live['attempted']:,} attempts / {live['kept']:,} kept**, **{live.get('images_per_second',0.):.3f} attempts/s / {live_rate:.3f} kept/s**. That short-window rate projects to **{live_days:.1f} more continuous days** in aggregate; uncached prompts, version changes and class bottlenecks make this provisional. The 30,000-attempt cap may leave difficult classes below their 5,000 target." if live_days is not None else 'The worker has just restarted; its new throughput estimate is not ready yet.',
        f"Background generation is running as **PID {process['pid']}**. It resumes from the manifest after restart. Creating `data/generated/STOP_GENERATION` requests a clean stop after the current batch. A class with zero keeps after 100 attempts is skipped for review; the per-class attempt cap is 30,000, so poor classes cannot spin forever.",'',
        '## Conversion and teacher filter','',
        'Generation cycles through quick black-marker doodles, careful outlines, filled flat colour, painted broad strokes and children’s crayon/scribble styles. Prompts also vary viewpoint and amount of detail, with one isolated object on light paper and no surrounding scene. Real-class generation targets filled/painted styles because their line data is already plentiful.',
        'The ongoing corpus worker adds larger-face/head/bust variants for zombie and robot, and clearer iconic variants for dynamite, poop, cursed face and straw hat, after their low pilot acceptance exposed loss of small features. Those new prompts are **not** part of the frozen pilot comparison below. Every image retains its exact prompt and source-code snapshot; no outside test results drove this prompt revision.',
        'The contact-sheet review also found a conversion fault: raw RGB nearest-neighbour matching could turn pale blue into grey/pink. The ongoing worker now uses **conversion v4**, matching white-paper dilution and darker shades of each game ink; palette-identity and diluted/shaded-hue checks pass. The frozen 1,409-image pilot remains **v2** and its actual inputs/metrics are preserved. The v4 correction has **not** been trained or evaluated here. A future comparison must freeze a version-specific bank or reconvert the older coloured samples; the first pilot does not establish v4 accuracy. [Before/after conversion picture](generated-colour-conversion-fix.png).',
        'Conversion estimates plain light background and foreground, maps to the ten game inks, traces outline skeletons and paints colour regions with scan strokes at varied directions and spacings. Neutral antialiased edges stay black/grey. Paint coverage varies from overlapping passes to loose stripes, with varied display brush widths. The resulting bounded point lists are then passed through the exact 40px nearest-stroke-owner rasteriser; there is no direct-bitmap training shortcut. Width stays display-only at serving time. Legal payload caps are 128 strokes and 4,096 points.','',
        'SigLIP scores four caption templates per class, averaged in text-feature space, plus six blank/noise/scene rejection captions. Both the source image and its converted game-format preview must have the intended class as top word and intended sigmoid match score at least 0.05. This score is a filtering heuristic, not calibrated probability or human quality. Rejected drawings are never training examples. Parent/subclass overlaps such as hat/straw hat can reject valid-looking drawings; filtering also risks rejecting unfamiliar styles. The sample picture includes these failures.','',
        'Exact duplicate four-plane inputs are excluded, including duplicates across different prompts/splits. The frozen ID and prompt-group snapshot is authoritative. This does not remove every visually similar drawing; seed/prompt separation is still weaker than holding out real human artists.','',
        '## Matched comparison','',
        'All three pilot branches start from the same selected GACL control checkpoint and run the same number of updates with the same QD batches, seeds, optimiser and learning-rate schedule. Each retains 16 of the control’s broader handwritten procedural examples per 256 inputs. The generated branches replace another 16 QD inputs with the same frozen generated train samples. Generated sampling allocates 80% to custom and 20% to real labels, balances available classes within each cohort, then available styles within each class; easy classes/styles cannot monopolise the pilot. Classes with only a few accepted images are necessarily repeated heavily, so their pilot generalisation remains uncertain. The hard-label branch uses the intended class; the distillation branch additionally uses teacher soft labels on those same converted images (KL weight 0.25, temperature 2). Black-input distillation from the control preserves existing behaviour in all three branches. GACL remains active. No generated test examples or outside test inputs are used for optimisation.','',
        '| Test (top-1 / top-3 %) | Installed colour299_40 | Wider GACL control | Matched no generated | Generated hard labels | Generated + SigLIP |','|---|---:|---:|---:|---:|---:|']
    names=['colour299_40','gacl299_40_wide']+MODELS
    record_students(names[1:]+(['gacl299_40_genkd_unscaled'] if 'gacl299_40_genkd_unscaled' in scores else []))
    for mode,label in [('black','QD real black'),('policy','QD real colourise'),('random','QD real random colours'),('filled','QD real filled'),('painted','QD real brush/scribble')]:
        lines.append('| '+label+' | '+' | '.join(pair(scores[name]['quickdraw'][mode]['real']) for name in names)+' |')
    for mode,label in [('black','All 299 black'),('policy','All 299 colourise'),('random','All 299 random colours'),('filled','All 299 filled'),('painted','All 299 brush/scribble')]:
        lines.append('| '+label+' | '+' | '.join(pair(scores[name]['quickdraw'][mode]['all']) for name in names)+' |')
    domains=[('tu','TU Berlin'),('qd_unrecognized','QD unrecognised'),('clipart','DomainNet clipart'),('painting','DomainNet painting'),('sketch','DomainNet sketch')]
    for domain,label in domains:
        mode=scores[MODELS[0]]['external'][domain]['primary_mode']
        lines.append('| '+label+' | '+' | '.join(pair(scores[name]['external'][domain]['modes'][mode]['all']) for name in names)+' |')
    for label,extract in [('Original synthetic, black',lambda s:s['quickdraw']['black']['synthetic']),
                          ('Original synthetic, colourise',lambda s:s['quickdraw']['policy']['synthetic']),
                          ('Original synthetic, random colours',lambda s:s['quickdraw']['random']['synthetic']),
                          ('Original synthetic, filled',lambda s:s['quickdraw']['filled']['synthetic']),
                          ('Original synthetic, brush/scribble',lambda s:s['quickdraw']['painted']['synthetic']),
                          ('New procedural views',lambda s:s['diverse_synthetic']['all']),
                          ('TU pistol + grenade',lambda s:s['external']['tu']['modes']['line']['custom']),
                          ('Generated pilot hold-out',lambda s:s['generated_holdout']['all']),
                          ('Same generated hold-out, all black',lambda s:s['generated_holdout_black']['all'])]:
        lines.append('| '+label+' | '+' | '.join(pair(extract(scores[name])) for name in names)+' |')
    lines += ['', 'Teacher loss in the main comparison has weight 0.25 **per generated example**, temperature 2, and is normalised over the full 256-example batch. With 16 generated examples the coefficient on their batch-mean KL is 0.015625 before the temperature-squared factor. Control/hard checkpoints were unchanged; the teacher branch was rerun on exactly the same frozen 1,170 training IDs after a reduction audit.','',
        'Synthetic tests and the tiny generated hold-out are upper bounds, not real-player accuracy. The generated hold-out is selected by the same teacher and can flatter teacher-distilled students. Only TU pistol/grenade provide natural drawings of our custom labels; the other 27 still lack an independent human benchmark.','',
        f"The additional all-black counterfactual zeros only the RGB planes of those same {len(snap['test_ids'])} generated test inputs. Teacher filtering used their original previews. The distilled model retains {100*scores['gacl299_40_genkd']['generated_holdout_black']['all']['top1']:.2f}% first guesses versus {100*scores['gacl299_40_genkd']['generated_holdout']['all']['top1']:.2f}% with original colours, so its generated-domain gain is not entirely a colour cue. This does not remove the observed real-to-custom bias or establish player accuracy.",'',
        'TU uses deduplicated images. QD unrecognised labels are noisy failed prompts. DomainNet retains the prior frozen raster adapters and label mapping; thinning, foreground extraction, missing stroke order and colour quantisation bias its scores, and its licence is unconfirmed, so it remains offline test-only. See [external evaluation](external-evaluation.md). Foundation pretraining (SigLIP WebLI; FLUX pretraining) is not fully auditable for overlap with public benchmarks, even though none of those test files were used in our training.','',
        '## Custom-class and black-hole false alarms','',
        'Top-1 output on mapped real-class truth; true custom and unmapped labels are excluded from the denominator.','',
        '| Test | Installed any custom / BH | Wider control any custom / BH | Matched no-gen any custom / BH | Generated hard any custom / BH | Generated + SigLIP any custom / BH |','|---|---:|---:|---:|---:|---:|']
    def alarms(s,mode=None,domain=None):
        if mode:
            r=s['quickdraw'][mode];return r['false_alarms']['rate'],r['black_hole_false_alarms']['rate']
        r=s['external'][domain]['modes'][s['external'][domain]['primary_mode']];return r['real_to_custom']['rate'],r['black_hole_false_alarms']['rate']
    for mode,label in [('black','QD black'),('filled','QD filled'),('painted','QD brush')]:
        lines.append('| '+label+' | '+' | '.join(' / '.join(percent(v) for v in alarms(scores[name],mode=mode)) for name in names)+' |')
    for domain,label in domains:
        lines.append('| '+label+' | '+' | '.join(' / '.join(percent(v) for v in alarms(scores[name],domain=domain)) for name in names)+' |')
    if 'gacl299_40_genkd_unscaled' in scores:
        old=scores['gacl299_40_genkd_unscaled'];new=scores['gacl299_40_genkd']
        lines += ['', '### Teacher-loss reduction audit','',
            'The first teacher trial averaged its KL over only 16 generated inputs while the main loss averaged over 256. That made the intended per-example teacher term **16 times too strong**. Its weights, protocol and metrics are preserved under `data/generated/experiments/unscaled-teacher-pilot/` and the `gacl299_40_genkd_unscaled` results key; they are not the teacher model in the main comparison. The correction changes only this reduction, not the dataset, seed, schedule, architecture, nominal weight parameter (0.25) or temperature.','',
            '| Diagnostic | Initial unscaled teacher | Corrected teacher |','|---|---:|---:|',
            f"| Real QD black top-1 | {100*old['quickdraw']['black']['real']['top1']:.2f}% | {100*new['quickdraw']['black']['real']['top1']:.2f}% |",
            f"| Real QD black to custom | {percent(old['quickdraw']['black']['false_alarms']['rate'])} | {percent(new['quickdraw']['black']['false_alarms']['rate'])} |",
            f"| DN sketch real to custom | {percent(alarms(old,domain='sketch')[0])} | {percent(alarms(new,domain='sketch')[0])} |",'',
            'The reduction bug was found during the initial result audit. Corrected outside-set results are repeat diagnostic evaluations, not a newly unseen benchmark; no outside inputs, labels, thresholds or hyperparameter search were used to optimise the correction. The author’s untouched private split remains the next independent check.','']
    lines += ['', 'Full per-class scores and worst custom destinations are in [results JSON](training-round-next.json).','',
        'The black guard is deliberately stricter than r4 alone: each real/custom top-1/top-3 score must stay within 0.2 percentage points of the better of installed colour299_40 and r4.','',
        '## Minimal bomb and quality checks','', '| Model | Minimal top-fuse bomb, black | Minimal top-fuse bomb, coloured | Black acceptance | TU > QD norm AUC |','|---|---|---|---|---:|']
    for name in names:
        s=scores[name];p={r['name']:r for r in s['scripted_probes']}
        def guesses(key):return ', '.join(r['label'] for r in p[key]['top3'])
        lines.append(f"| {name} | {guesses('top-fuse bomb black')} | {guesses('top-fuse bomb coloured')} | {s.get('black_acceptance_guard','baseline')} | {s['quality']['macro_tu_over_qd_auc']:.3f} |")
    lines += ['', '| Model | TU / QD mean feature norm | TU-higher classes / 170 | Within-class norm/ink Spearman, TU / QD |','|---|---:|---:|---:|']
    for name in names:
        q=scores[name]['quality'];rho=q['within_class_norm_ink_spearman']
        lines.append(f"| {name} | {q['tu_mean_norm']:.3f} / {q['qd_mean_norm']:.3f} | {q['tu_higher_classes']} | {rho['tu']:.3f} / {rho['quickdraw']:.3f} |")
    failed=[name for name in names if scores[name]['quality']['trained_with_gacl'] and scores[name]['quality']['tu_mean_norm']<scores[name]['quality']['qd_mean_norm']]
    if failed:lines += ['', '**TU-higher quality sanity check fails for:** '+', '.join(failed)+'. The learned norm is available, but this quality goal is not validated.']
    lines += ['', 'GACL exposes the raw feature norm, but these dataset comparisons do not validate an artistic-quality score. The old model already ranks TU above QD and correlates with ink coverage. Dataset identity, recognisability and raster conversion remain confounds. The author’s Rough/Good/Detailed ratings remain the ground truth for tiers later; none were opened or trained on here. No score threshold, tier or game effect was added.','',
        '## Previously captured mouse gestures','',
        'These twelve independently authored, paced Studio-MCP gestures were retained from the earlier live diagnosis. This turn replays their exact strokes offline, without opening Studio. They are developer regression cases, never training examples or model-selection inputs, and do not estimate player accuracy. The author’s own held-out drawings are separate and remain unopened.','',
        '| Drawing | Installed | Wider control | Matched no-gen | Generated hard | Generated + SigLIP |','|---|---|---|---|---|---|']
    for i,row in enumerate(scores[names[0]]['captured_replays']['records']):
        title=f"{row['label']}, {'painted' if row['paint'] else 'outline'}, scale {row['nominal_scale']}"
        lines.append('| '+title+' | '+' | '.join(scores[name]['captured_replays']['records'][i]['top3'][0]['label'] for name in names)+' |')
    lines += ['| Exact-label hits | '+' | '.join(str(scores[name]['captured_replays']['correct'])+' / 12' for name in names)+' |','',
        '## Cost, parity and current state','',
        f"The student architecture remains four-plane 40px, conv 48/96/192, dense 384, 299 outputs, int8. SigLIP and FLUX never run in Roblox. Measured native / interpreted Luau CLI forward medians for the distilled branch: **{timing['native']['FORWARD'][0]:.2f} / {timing['interpreted']['FORWARD'][0]:.2f} ms**; calibrated native server estimate: **{native['calibrated_server_forward_ms']:.2f} ms**, using the prior installed-server 6.87 ms reference. Eight scripted recognitions at a 2 ms / 60 Hz queue budget project to **{native['eight_recognize_wall_seconds_at_2ms_60hz']:.2f} s** plus network/scheduler overhead. Worst offline native slice across recognition/forward jobs: **{worst_native:.2f} ms**. This is **not a fresh server measurement**; Studio was forbidden.",
        f"The largest retained dev-vector drawing has **{int(timing['native']['COMPLEX_RECOGNIZE'][3])} points / {int(timing['native']['COMPLEX_RECOGNIZE'][4])} strokes**. Its sliced rasterisation plus network uses **{timing['native']['COMPLEX_RECOGNIZE'][0]:.2f} ms native / {timing['interpreted']['COMPLEX_RECOGNIZE'][0]:.2f} ms interpreted** of active CLI work, including cooperative scheduling overhead.",
        'Python float64 / native and interpreted Luau checks cover logits, colour planes, feature norms, synchronous/sliced recognition and interleaved jobs. Exported weights plus DrawModelInfo stay compatible; recognise’s first four returns and the 40px preview layout stay unchanged. New artifacts are dev modules only. No DrawTest, DrawCollect, place, remote or live model changes were made.','',
        '**Not checked:** fresh Studio Verify/server round-trip and burst timing; the author’s real held-out drawings and quality ratings; completion of the full generated corpus. No installation is recommended solely from synthetic scores.','',
        *style_lines,'## Pilot keep rate by custom class','', '| Class | Attempted | Kept | Train | Validation | Test | Keep rate |','|---|---:|---:|---:|---:|---:|---:|']
    for row in pilot['classes'][270:]:
        n=row['label'];partition=' | '.join(str(split_counts[(n,s)]) for s in ['train','validation','test'])
        lines.append(f"| {n} | {row['attempted']} | {row['kept']} | {partition} | {percent(row['keep_rate']) if row['keep_rate'] is not None else 'not attempted'} |")
    lines += ['', '## Real classes attempted in this pilot','', '| Class | Attempted | Kept | Keep rate |','|---|---:|---:|---:|']
    for row in pilot['classes'][:270]:
        if row['attempted']:lines.append(f"| {row['label']} | {row['attempted']} | {row['kept']} | {percent(row['keep_rate'])} |")
    lines += ['', 'All 299 per-class/per-style counts, including not-yet-attempted classes, are in [pilot status](../data/generated/pilot-generation-status.json). Zero-kept custom classes in this pilot: '+(', '.join(zero) or 'none')+'.','']
    lines += ['Custom classes with no accepted **training** image after prompt-group splitting: '+(', '.join(missing_train) or 'none')+'. Those outputs still train on the original and broader procedural sources; their AI-domain coverage remains missing in this pilot.','']
    (NOTES/'generated-training-round.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')

    # Kept and rejected *converted* inputs. No QD or external benchmark art appears.
    candidates=[r for r in pilot_rows if r.get('converted_png')]
    def diverse(keep):
        result=[];seen=set()
        for style in ['marker','outline','flat','painted','kids']:
            for r in candidates:
                if r['keep']==keep and r['style']==style and r['word'] not in seen:
                    result.append(r);seen.add(r['word']);break
        for r in candidates:
            if len(result)>=6:break
            if r['keep']==keep and r['word'] not in seen:result.append(r);seen.add(r['word'])
        return result[:6]
    kept=diverse(True);rejected=diverse(False)
    fig,axes=plt.subplots(3,4,figsize=(20,11),facecolor='white')
    for colgroup,examples,title,color in [(0,kept,'KEPT','#287956'),(2,rejected,'REJECTED','#a34d3a')]:
        for index,r in enumerate(examples):
            source=Image.open(DATA/r['raw_png']).convert('RGB').resize((384,384),Image.Resampling.LANCZOS)
            converted=Image.open(DATA/r['converted_png']).convert('RGB').resize((384,384),Image.Resampling.NEAREST)
            pair_image=Image.new('RGB',(772,384),'#d5d5d5');pair_image.paste(source,(0,0));pair_image.paste(converted,(388,0))
            ax=axes[index//2,colgroup+index%2];ax.imshow(pair_image);ax.set_xticks([]);ax.set_yticks([])
            decision=r['converted_teacher'];ax.set_title(f"{title}: {r['word']} / {r['style']}\nTeacher: {decision['top_label']} | match {decision['intended_score']:.3f}\n{r['reason']}",fontsize=10,color=color)
    for ax in axes.flat:
        if not ax.images:ax.axis('off')
    fig.suptitle('AI-generated samples: original | 40px game input',fontsize=19,weight='bold')
    fig.text(.5,.935,'Kept samples on the left, rejects on the right. Frozen SigLIP can reject valid drawings. These are not player drawings.',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.025,1,.91),h_pad=2);fig.text(.5,.01,'FLUX.1-schnell / SigLIP, Apache-2.0. Offline only; never displayed in the game.',ha='center',fontsize=10)
    fig.savefig(NOTES/'generated-kept-rejected.png',dpi=160,metadata={'generated_content':'true','generator':'FLUX.1-schnell','usage':'Offline development only, never player-facing'});plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(16,11));width=.24
    def chart(ax,labels,values,baseline,title,ylabel='Top-1 (%)',fmt='%.1f'):
        x=np.arange(len(labels))
        for i,(name,label,color) in enumerate(zip(MODELS,LABELS,COLORS)):
            bars=ax.bar(x+(i-1)*width,np.asarray(values[i])*100,width,label=label,color=color)
            ax.bar_label(bars,fmt=fmt,fontsize=8,padding=2)
        if baseline is not None:ax.scatter(x+.41,np.asarray(baseline)*100,color='#222222',marker='D',s=25,label='Installed baseline',zorder=5)
        ax.set_xticks(x,labels);ax.set_title(title,loc='left',weight='bold');ax.set_ylabel(ylabel);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    modes=['black','policy','random','filled','painted']
    chart(axes[0,0],['Black','Colourise','Random','Filled','Brush'],[[scores[name]['quickdraw'][m]['real']['top1'] for m in modes] for name in MODELS],
          [scores['colour299_40']['quickdraw'][m]['real']['top1'] for m in modes],'A  Real Quick, Draw! held-outs');axes[0,0].set_ylim(0,100)
    vals=[]
    for name in MODELS:vals.append([scores[name]['external'][d]['modes'][scores[name]['external'][d]['primary_mode']]['all']['top1'] for d,_ in domains])
    base=[scores['colour299_40']['external'][d]['modes'][scores['colour299_40']['external'][d]['primary_mode']]['all']['top1'] for d,_ in domains]
    chart(axes[0,1],['TU Berlin','QD failed','Clipart','Painting','Sketch'],vals,base,'B  Outside tests (fixed adapters)')
    getters=[lambda s:s['quickdraw']['black']['synthetic']['top1'],lambda s:s['diverse_synthetic']['all']['top1'],lambda s:s['external']['tu']['modes']['line']['custom']['top1'],lambda s:s['generated_holdout']['all']['top1']]
    chart(axes[1,0],['Old synth','New synth views','TU weapons','AI hold-out'],[[f(scores[name]) for f in getters] for name in MODELS],None,'C  Custom transfer: synthetic scores are upper bounds');axes[1,0].set_ylim(0,115)
    false_domains=['tu','qd_unrecognized','sketch']
    values=[[alarms(scores[name],domain=d)[1] for d in false_domains] for name in MODELS]
    chart(axes[1,1],['TU Berlin','QD failed','DN sketch'],values,None,'D  Black-hole false alarms on real classes','Real drawings predicted black hole (%)',fmt='%.3f')
    axes[0,0].legend(loc='lower left',fontsize=9)
    fig.suptitle('Generated-data pilot: matched student comparison, not installed',fontsize=20,weight='bold')
    fig.text(.5,.93,f"{pilot['attempted']} generated / {pilot['kept']} kept | {len(snap['rows'])} train | {len(snap['test_ids'])} AI test | No outside-test training or player-quality tier fitting",ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.08,1,.90),h_pad=3,w_pad=3)
    fig.text(.5,.025,f"Native CLI {native['candidate']['native']['FORWARD'][0]:.1f} ms; server estimate ~{native['calibrated_server_forward_ms']:.1f} ms (not fresh Studio timing). Full-corpus generation continues in the background.",ha='center',fontsize=11)
    fig.savefig(NOTES/'generated-training-results.png',dpi=160);plt.close(fig)
    print('REPORT',NOTES/'generated-training-round.md',flush=True)
    print('SAMPLES',NOTES/'generated-kept-rejected.png',flush=True)
    print('RESULTS',NOTES/'generated-training-results.png',flush=True)


if __name__=='__main__':main()
