"""CPU-only, lossless Studio package. Never contacts or installs into Studio.

Keep source modules below 180 KB rather than relying on large Source writes.
The full model and training-only parity vectors belong in ServerStorage only.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


MAX_SOURCE = 180000
CHUNK_CHARACTERS = 168000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    source = args.export.resolve()
    destination = args.out.resolve()
    assert source != destination
    destination.mkdir(parents=True, exist_ok=True)
    original = (source / 'Weights.luau').read_text(encoding='utf-8')
    match = re.search(r'\tdata = table\.concat\(\{\n(.*?)\t\}\),', original, re.S)
    assert match, 'Expected generated base64 weight table'
    encoded = ''.join(re.findall(r'"([A-Za-z0-9+/=]+)"', match.group(1)))
    raw = base64.b64decode(encoded, validate=True)
    parts = [encoded[i:i+CHUNK_CHARACTERS] for i in range(0,len(encoded),CHUNK_CHARACTERS)]
    modules = {}
    for i,part in enumerate(parts,1):
        name = f'WeightPart{i:03d}'
        # Short literals also avoid depending on a single huge string token.
        lines = ['--!nocheck', '-- Generated private weight data; never replicate.',
                 'return table.concat({']
        lines += ['\t'+json.dumps(part[j:j+120])+',' for j in range(0,len(part),120)]
        lines += ['})', '']
        modules[name] = '\n'.join(lines)
    reconstructed = ''.join(''.join(re.findall(r'"([A-Za-z0-9+/=]+)"',modules[f'WeightPart{i:03d}']))
                            for i in range(1,len(parts)+1))
    assert reconstructed == encoded
    references = ',\n'.join(f'\t\trequire(script:WaitForChild("WeightPart{i:03d}"))'
                              for i in range(1,len(parts)+1))
    weights = original[:match.start()]+'\tdata = table.concat({\n'+references+'\n\t}),'+original[match.end():]
    root = ET.Element('roblox', {'version':'4'})
    serial = 0
    sources = {}

    def instance(parent, class_name, name, code=None, path=''):
        nonlocal serial
        serial += 1
        node = ET.SubElement(parent,'Item',{'class':class_name,'referent':f'RBX{serial}'})
        props = ET.SubElement(node,'Properties')
        ET.SubElement(props,'string',{'name':'Name'}).text = name
        if code is not None:
            assert len(code.encode('utf-8')) < MAX_SOURCE, name
            ET.SubElement(props,'ProtectedString',{'name':'Source'}).text = code
            sources[path+name] = code
        return node

    model_name = json.loads((source/'model.json').read_text())['name']
    folder = instance(root,'Folder',model_name)
    recognizer = instance(folder,'ModuleScript','DrawRecognizer',
                          (source/'DrawRecognizer.luau').read_text(encoding='utf-8'))
    weight_node = instance(recognizer,'ModuleScript','Weights',weights,'DrawRecognizer/')
    for name,code in modules.items():
        instance(weight_node,'ModuleScript',name,code,'DrawRecognizer/Weights/')
    rasterizer = instance(recognizer,'ModuleScript','DrawRasterizer',
                          (source/'DrawRasterizer.luau').read_text(encoding='utf-8'),'DrawRecognizer/')
    instance(rasterizer,'ModuleScript','DrawJob',
             (source/'DrawJob.luau').read_text(encoding='utf-8'),'DrawRecognizer/DrawRasterizer/')
    for name,filename in [('DrawModelInfo','DrawModelInfo.luau'),('TestVectors','TestVectors.luau')]:
        instance(folder,'ModuleScript',name,(source/filename).read_text(encoding='utf-8'))
    model = destination / (model_name+'.rbxmx')
    ET.ElementTree(root).write(model,encoding='utf-8',xml_declaration=True)
    parsed = ET.parse(model)
    xml_sources = [p.text for p in parsed.findall('.//ProtectedString[@name="Source"]')]
    assert sorted(xml_sources) == sorted(sources.values()), 'XML source roundtrip mismatch'
    for path,code in sources.items():
        output = destination/'modules'/(path+'.luau')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(code,encoding='utf-8',newline='\n')
    manifest = {'name':model_name,'source_export':str(source),'installed':False,
                'provisional':True,'private_only':True,'source_limit_used_bytes':MAX_SOURCE,
                'part_count':len(parts),'max_source_bytes':max(len(s.encode('utf-8')) for s in sources.values()),
                'decoded_weight_bytes':len(raw),'decoded_weights_sha256':sha(raw),
                'original_weights_source_sha256':sha((source/'Weights.luau').read_bytes()),
                'rbxmx':str(model),'rbxmx_sha256':sha(model.read_bytes()),
                'sources':{path:{'bytes':len(code.encode('utf-8')),'sha256':sha(code.encode('utf-8'))}
                           for path,code in sources.items()},
                'checks':['base64 reconstructed exactly','decoded weights unchanged',
                          'all module sources below package bound','XML source roundtrip identical'],
                'studio_checks':'pending: NN test is not connected to Studio MCP'}
    (destination/'package.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({k:manifest[k] for k in ['name','part_count','max_source_bytes',
                     'decoded_weight_bytes','rbxmx','installed']}))


if __name__=='__main__':
    main()
