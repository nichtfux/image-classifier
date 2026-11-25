"""Compare the paired 270-category models, including paired-test uncertainty."""
import json
import hashlib
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    models, predictions = {}, {}
    for size in (28,40):
        name=f"wide270_{size}"
        models[size]=json.loads((ROOT/"training/out"/(name+".json")).read_text())
        with np.load(ROOT/"training/out"/(name+".predictions.npz")) as p:
            predictions[size]=(p["truth"],p["top3"])
    a,b=models[28],models[40]
    assert a["categories"]==b["categories"] and len(a["categories"])==270
    assert a["train_ids_sha256"]==b["train_ids_sha256"]
    y,pa=predictions[28]
    yb,pb=predictions[40]
    assert np.array_equal(y,yb)
    right28,right40=pa[:,0]==y,pb[:,0]==y
    gain=right40.astype(np.float64)-right28.astype(np.float64)
    se=float(gain.std(ddof=1)/math.sqrt(len(gain)))
    report={"same_data_and_splits":True,"test_drawings":len(y),"models":models,
            "top1_gain40_percentage_points":100*float(gain.mean()),
            "paired_gain95_interval_percentage_points":[100*(float(gain.mean())-1.96*se),100*(float(gain.mean())+1.96*se)],
            "only40_correct":int((~right28 & right40).sum()),"only28_correct":int((right28 & ~right40).sum()),
            "cohorts":{}}
    for label,mask in (("original100",y<100),("added170",y>=100)):
        report["cohorts"][label]={}
        for size,p in ((28,pa),(40,pb)):
            report["cohorts"][label][size]={"top1":float((p[mask,0]==y[mask]).mean()),
                  "top3":float((p[mask]==y[mask,None]).any(1).mean())}
    report["studio"]={}
    for size in (28,40):
        path=ROOT/"notes"/f"studio-benchmark-270-{size}.json"
        if path.exists():
            saved=json.loads(path.read_text())
            weight_hash=hashlib.sha256((ROOT/"training/out"/f"wide270_{size}.npz").read_bytes()).hexdigest()
            source_hash=hashlib.sha256((ROOT/"luau/DrawRecognizer.luau").read_bytes()).hexdigest()
            if saved.get("model_sha256")==weight_hash and saved.get("inference_source_sha256")==source_hash:
                report["studio"][size]=saved
    (ROOT/"notes/resolution-comparison.json").write_text(json.dumps(report,indent=1),encoding="utf-8",newline="\n")
    print(json.dumps({k:v for k,v in report.items() if k!="models"},indent=1))


if __name__=="__main__":
    main()
