#!/usr/bin/env python3
"""Static asset and source provenance audit; creates no simulator or data."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
C = json.loads(Path(__file__).with_name('config.json').read_text())


def main():
    subprocess.run(['git', 'merge-base', '--is-ancestor', C['source_commit'], 'HEAD'], cwd=ROOT, check=True)
    urdfs = ['high-level/data/asset/a2wz1/urdf/a2wz1.urdf',
             'high-level/data/asset/z1/urdf/z1_arm.urdf', 'high-level/data/asset/door_set/wc4/model.urdf']
    references = []
    for name in urdfs:
        p = ROOT / name
        for node in ET.parse(p).iter('mesh'):
            target = (p.parent / node.attrib['filename']).resolve()
            assert target.is_file(), (p, target)
            references.append({'urdf': name, 'mesh': str(target.relative_to(ROOT)),
                               'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    audited = urdfs + [C['door_cfg'], C['skill'],
        'high-level/data/asset/door_set/wc4/bounding_box.json',
        'high-level/data/asset/door_set/wc4/handle_bounding.json',
        'high-level/float_ik/config/a2w_float_ik_push_door_parallel.yaml',
        'high-level/dp/config/depth_camera_aug_default.yaml',
        'high-level/dp/door_policy_backend.py', 'high-level/dp/door_policy_worker.py',
        'high-level/lerobot/src/lerobot/policies/act/configuration_act.py',
        'high-level/lerobot/src/lerobot/policies/act/modeling_act.py']
    audited += sorted({r['mesh'] for r in references})
    hashes = {}
    for name in audited:
        expected = subprocess.check_output(['git', 'rev-parse', C['source_commit'] + ':' + name], cwd=ROOT, text=True).strip()
        actual = subprocess.check_output(['git', 'hash-object', name], cwd=ROOT, text=True).strip()
        assert actual == expected, name
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    print(json.dumps({'root': str(ROOT), 'source_commit': C['source_commit'],
                      'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      'source_files_unchanged': hashes, 'mesh_references': references}, indent=2))


if __name__ == '__main__':
    main()
