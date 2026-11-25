"""Export the completed 64px checkpoint on CPU through its original run path."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from gacl import QualityNet, quantize
from long_run64 import RUN, ROOT, export, atomic


def main():
    torch.set_num_threads(3)
    checkpoint = RUN / 'last.pt'
    verified = json.loads((RUN / 'checkpoint-verified.json').read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == verified['sha256']
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    step = ck['state']['step']
    assert step == verified['step'] == json.loads((RUN / 'DONE.json').read_text())['step']
    net = QualityNet(ck['widths'], ck['fc'], ck['size'], in_channels=4).eval()
    net.load_state_dict(ck['state_dict'])
    net.cosine = ck['cosine']
    net.quantized_head = ck.get('head_scale_folded', False)
    final = json.loads((RUN / f'validation-s{step:09d}.json').read_text())
    artifact = export(net, quantize(net), final['scores'], final['guards'], step)
    package = ROOT / 'training/out' / ('studio-test-' + Path(artifact['path']).name)
    subprocess.run([sys.executable, str(ROOT/'training/package_studio_model.py'),
                    '--export', artifact['path'], '--out', str(package)], check=True)
    atomic(RUN/'final-export.json', {**artifact, 'package': str(package),
                                    'checkpoint_sha256': verified['sha256'],
                                    'selection': 'Final checkpoint explicitly approved by the author; provisional guards.'})
    assert not torch.cuda.is_initialized()
    print('FINAL_EXPORT', artifact['path'], flush=True)


if __name__ == '__main__':
    main()
