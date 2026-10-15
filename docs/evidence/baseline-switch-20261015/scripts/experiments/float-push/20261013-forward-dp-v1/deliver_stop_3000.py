"""CPU-only final evidence for the user-requested stop at 3000 updates."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

REPO=Path('/data/zyf/video2door')
sys.path.insert(0,str(REPO/'high-level/float_ik/learning'))
from deliver_v3 import paired_video
from forward_report import report
from forward_results import episode_result, aggregate


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main(root):
    state=read(root/'pipeline_state.json')
    assert state['phase']=='stopped_by_user' and state['completed_updates']==3000
    assert not (root/'active_command.json').exists()
    checkpoints=list((root/'training/checkpoints').glob('step_*.pt'))
    assert sorted(int(p.stem.split('_')[-1]) for p in checkpoints)==[1000,2000,3000]
    checkpoint=root/'training/checkpoints/step_003000.pt'
    assert sha(checkpoint)==state['checkpoint_sha256']==read(checkpoint.with_suffix('.json'))['sha256']
    expected=read(root/'plan.json')['splits']['val'][::2]
    expected_seeds=[c['seed'] for c in expected]
    suite=root/'evaluation/step_003000'
    attempts=read(suite/'attempts.json')
    original=read(suite/'results.json')
    assert len(attempts)==len(original)==9
    assert [c['seed'] for c in attempts]==expected_seeds
    assert [c['seed'] for c in original]==expected_seeds
    rows=[]
    for case,prior in zip(expected,original):
        folder=suite/('seed_%d'%case['seed'])
        row=episode_result(folder,case,prior['exit_code'])
        for k in ('success','opened','pass_done','autonomous','frames','max_signed_door_deg'):
            assert row[k]==prior[k],(case,k,row[k],prior[k])
        video=folder/'rollout.mp4'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-xerror','-threads','2',
                        '-i',str(video),'-threads','2','-f','null','-'],check=True)
        row['video_sha256']=sha(video);row['full_video_decode_passed']=True
        rows.append(row)
    assert read(root/'checkpoint_checks_003000.json')['passed']
    report(root)
    out=root/'analysis/video_review_3000';out.mkdir(parents=True,exist_ok=True)
    pair=paired_video(root/'dataset/val_seed_16108',suite/'seed_16108',out,updates=3000)
    result=dict(completed_updates=3000,stopped_by_user=True,objective_complete=False,
                checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                aggregate=aggregate(rows,9),maximum_door_angle_deg=max(r['max_signed_door_deg'] for r in rows),
                all_episodes_autonomous=all(r['autonomous'] for r in rows),
                process_errors=sum(r['exit_code']!=0 for r in rows),
                invalid_image_frames=sum(r['invalid_image_frames'] or 0 for r in rows),
                end_reasons=[r['end_reason'] for r in rows],paired_video=pair,episodes=rows)
    (root/'delivery_3000.json').write_text(json.dumps(result,indent=2))
    local=Path('/Users/yu/Data/Video2Door/reproduction/pro5000/forward-dp-v1')
    lines=['# 3000更新记录与完整视频','',
           '本轮已按用户要求在3000更新结束，相关进程退出；最终80°开门并直接前穿目标未通过验收。',
           '', '| 更新数 | ≥80°开门 | 通过 | 自主联合 | 标称 | 最大门角 |',
           '|---:|---:|---:|---:|---:|---:|']
    for step in (1000,2000,3000):
        group=read(root/('evaluation/step_%06d/results.json'%step));a=aggregate(group,9)
        lines.append('| %d | %d/9 | %d/9 | %d/9 | %d/%d | %.4f° |'%(step,a['opened'],a['passed'],
              a['successes'],a['nominal_successes'],a['nominal_attempts'],max(r['max_signed_door_deg'] for r in group)))
    lines += ['', '9个3000更新评测均正常运行至2500步（50秒仿真时间），无专家回退及无效图像。',
              '其中2回合推开约41.0°和31.6°，2回合仅约0.5°，其余门角近0°；全部未达到80°，因步数上限结束。',
              '', '[3000更新模型](%s)'%(local/'training/checkpoints/step_003000.pt'),
              '[训练曲线](%s)'%(local/'analysis/training_curves.png'),
              '[逐回合指标](%s)'%(local/'evaluation/step_003000/results.csv'),
              '[标称同初态专家对比](%s)'%(local/'analysis/video_review_3000/comparison_seed_16108.mp4'),
              '', '对比视频左侧为专家、右侧为3000更新DP；专家结束后保持最终帧，DP完整50秒回合不裁剪。',
              '配对已验证相同源码、seed、初始状态和四路初始观测完全相同。',
              '', '| seed | x | y参数 | 最大门角 | 完整视频 |', '|---:|---:|---:|---:|---|']
    for r in rows:
        video=local/'evaluation/step_003000'/('seed_%d'%r['seed'])/'rollout.mp4'
        lines.append('| %d | %.1f | %.2f | %.4f° | [完整50秒](%s) |'%(r['seed'],r['robot_x'],r['robot_y'],r['max_signed_door_deg'],video))
    lines += ['', '模型SHA256：`'+result['checkpoint_sha256']+'`。',
              '本地文件哈希对照与完整视频解码证据见local_verification_3000.json、delivery_sha256.json。']
    (root/'analysis/INDEX_3000.md').write_text('\n'.join(lines)+'\n')
    selected=set()
    for name in ('sources','environment','analysis','evaluation/step_003000','user_stop_3000'):
        selected.update(p for p in (root/name).rglob('*') if p.is_file())
    for step in (1000,2000):
        selected.update(p for p in (root/('evaluation/step_%06d'%step)).glob('*') if p.is_file())
    selected.update(p for p in (root/'training').rglob('*') if p.is_file() and
                    (p.suffix!='.pt' or p.name=='step_003000.pt'))
    selected.update(p for p in (root/'logs').glob('*') if p.is_file())
    selected.update(p for p in root.glob('*.json') if p.name not in ('delivery_sha256.json','heartbeat_state.json'))
    selected.update(root/name for name in ('supervisor.log','stop_watch.log','stop_guard_tests.log',
                    'source_changes.patch','stop_after_3000.py','deliver_stop_3000.py','test_stop_after_3000.py'))
    hashes={str(p.relative_to(root)):sha(p) for p in sorted(selected) if p.exists()}
    (root/'delivery_sha256.json').write_text(json.dumps(dict(files=hashes,count=len(hashes)),indent=2))
    print(json.dumps(dict(aggregate=result['aggregate'],maximum_door_angle_deg=result['maximum_door_angle_deg'],
                         checkpoint_sha256=result['checkpoint_sha256'],full_videos_decoded=9,
                         paired_initial_observations_identical=True,delivery_files=len(hashes))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(Path(a.root).resolve())
