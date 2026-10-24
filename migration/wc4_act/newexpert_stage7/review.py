"""R11 + full measured motion/contact/source review. No video is claimed."""
import argparse,collections,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'migration/wc4_act'),str(ROOT/'migration/wc4_act/r11')]
from criteria import score
from newexpert_stage7.common import read,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run;out=run/'attempt_000';audit=run/'audit';audit.mkdir(exist_ok=True)
    result=read(out/'result.json');quality=read(audit/'record-validation.json');frozen=read(out/'frozen-trial.json')
    states=[json.loads(line) for line in (out/'states.jsonl').open()];n=len(states);assert n==result['steps'] and n>0
    locks=[json.loads(line) for line in (out/'lock-writes.jsonl').open()]
    groups={};points=[];aligned_loaded=[];ambiguous=[];external={};phases=[]
    for i,s in enumerate(states):
        if not i or s['phase']!=states[i-1]['phase']:
            if phases:phases[-1].update(last_step=i-1,last_time_s=states[i-1]['t'])
            phases.append({'phase':s['phase'],'first_step':i,'first_time_s':s['t']})
    phases[-1].update(last_step=n-1,last_time_s=states[-1]['t'])
    for i,line in enumerate((out/'contacts.jsonl').open()):
        row=json.loads(line);assert row['step']==i;s=states[i]
        s.update(end_panel_raw_contact=False,loaded_door_contact=False,zero_lambda_door_records=0,nonend_door_records=0)
        for c in row['contacts']:
            door=[x for x in c['names'] if x.get('actor_handle')==2];robot=[x for x in c['names'] if x.get('actor_handle') in [0,1]]
            loaded=c.get('lambda',0)!=0
            if loaded and c.get('classification') in ['unexpected_external_loaded','other_self_contact_loaded']:
                key=s['phase']+' | '+c['classification']+' | '+' / '.join(x.get('body','unknown') for x in c['names'])
                if key not in external:external[key]={'records':0,'first_step':i,'last_step':i}
                external[key]['records']+=1;external[key]['last_step']=i
            if not door or not robot:continue
            end=robot[0]['body'] in ['gripperStator','gripperMover'];panel=door[0]['body']=='door_panel'
            s['end_panel_raw_contact']|=end and panel;s['loaded_door_contact']|=loaded
            s['zero_lambda_door_records']+=not loaded;s['nonend_door_records']+=not end
            if loaded:
                key=s['phase']+' | '+robot[0]['body']+' / '+door[0]['body']
                if key not in groups:groups[key]={'records':0,'first_step':i,'last_step':i}
                groups[key]['records']+=1;groups[key]['last_step']=i
            if loaded and s['phase']=='close':
                if not end:ambiguous.append({'step':i,'robot_body':robot[0]['body'],'door_body':door[0]['body']})
                elif panel:
                    j=next(j for j,x in enumerate(c['names']) if x.get('actor_handle') in [0,1])
                    local=[c['localPos'+str(j)][k] for k in ['x','y','z']]
                    if robot[0]['body']=='gripperStator':points.append(local)
                    if i>0 and s['door_q'][0]<s['pre_physics']['door_q'][0] and np.linalg.norm(np.asarray(s['ee_actual'])-states[i-1]['ee_actual'])>0:
                        aligned_loaded.append(i)
    assert i+1==n
    scored=score(states,locks,read(out/'reference.json'))
    root_z=np.array([s['root_rb']['pose']['p']['z'] for s in states]);z=frozen['expected_spawn']['robot_z']
    if np.max(abs(root_z-z))>1e-7:raise RuntimeError('SYSTEMIC: base height changed within episode')
    source=read(run/'manifest.json');code_names=['migration/wc4_act/fullcycle_expert.py','migration/wc4_act/fullcycle_compact.py',
        'migration/wc4_act/newexpert_stage2/push_geometry.py','migration/wc4_act/r11/criteria.py']
    source_binding={name:sha(ROOT/name) for name in code_names}
    assert all(source['source_sha256'][name]==h for name,h in source_binding.items())
    reasons=list(scored['failures'])
    if not quality['recording_checks_pass']:reasons.append('raw_recording_quality_failed')
    if not aligned_loaded:reasons.append('no_loaded_end_contact_with_measured_ee_and_door_motion')
    # Non-end contact alone is not an R11 veto. Ambiguous attribution is held
    # for explicit joint review; it cannot be admitted merely by auto success.
    if ambiguous:reasons.append('requires_joint_review_of_nonend_close_contacts')
    complete=scored['measured_task_gates_pass'] and bool(aligned_loaded) and not ambiguous
    eligible=bool(complete and quality['recording_checks_pass'])
    contact_face=None
    if points:
        pts=np.array(points);contact_face={'stator_loaded_records':len(points),'min_xyz_m':pts.min(0).tolist(),
            'max_xyz_m':pts.max(0).tolist(),'source_pad_positive_x_face_m':.15,
            'max_x_face_difference_m':float(np.max(abs(pts[:,0]-.15))),
            'meaning':'Raw points compared with unchanged gripper collision geometry; diagnostic, not a new force/coverage/gap threshold. No cooked subshape ID is available.'}
    review={'source_index':frozen['source_index'],'new_trial_index':frozen['index'],'seed':frozen['seed'],
        'automatic_terminal':result['automatic_final_conditions_met'],'r11_measured_gates_pass':scored['measured_task_gates_pass'],
        'complete_task_review':'supported_by_measured_contact_motion_and_frozen_source' if complete else ('pending_ambiguous_contact_review' if ambiguous and scored['measured_task_gates_pass'] else 'not_complete'),
        'raw_recording_quality_pass':quality['recording_checks_pass'],'eligible_raw_episode':eligible,'exclusion_or_hold_reasons':reasons,
        'video_recorded':False,'video_waiver':'Explicit user stage-seven task: no video required; no visual-review claim.',
        'review_method':'Full R11 measured gates + actual EE/door motion coupled with loaded intended end contact + raw body/local positions + unchanged full-cycle/hinge/lock source + release/home/common window. Not auto terminal or a contact flag alone.',
        'frames':quality['frames'],'physics_steps':n,'sim_seconds':result['sim_seconds'],'stop_reason':result['reason'],
        'failure_or_stop_phase':states[-1]['phase'],'root_z_range_m':[float(root_z.min()),float(root_z.max())],
        'initial_binding_pass':read(out/'initial-frozen-comparison.json')['matches_frozen'],
        'physical_binding_pass':read(out/'physical-camera-binding.json')['all_physics_match'],
        'aligned_loaded_push_steps':sorted(set(aligned_loaded)),'loaded_door_contact_groups':groups,
        'ambiguous_nonend_close_records':ambiguous,'contact_face_diagnostic':contact_face,
        'other_loaded_contact_diagnostics':external,'source_binding':source_binding,
        'record_quality_file_sha256':sha(audit/'record-validation.json'),
        'raw_npz':str(out/'raw/episode_000000.npz'),'raw_npz_sha256':quality['raw_npz_sha256'],
        'entry_into_final_candidate_index':'Requires final per-source joint review decision; this file prepares evidence and does not itself assemble a dataset.',
        'limitations':['No original RGB video for this new trial, by explicit task order','Contact subshape and dynamics cannot be certified from a scalar contact flag; raw complete records retained','Selection is raw-stage only; no conversion or training acceptance has been performed']}
    def save(name,value):(audit/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    save('r11-score.json',scored);save('phase-intervals.json',phases);save('task-review.json',review)
    events={'whole_body_pass':scored['whole_body_pass_step'],'first_aligned_loaded_push':aligned_loaded[0] if aligned_loaded else None,
        'latch':next((x['step'] for x in locks if x['kind']=='relocked'),None),'stop':n-1}
    for phase in ['compact_rendezvous','compact_seek','close','release','retract','final_hold']:
        events[phase]=next((x['first_step'] for x in phases if x['phase']==phase),None)
    save('event-index.json',{k:None if v is None else {'step':v,'post_time_s':states[v]['t'],
        'raw_state_line_1based':v+1,'raw_contact_line_1based':v+1,'nearest_raw_npz_frame':v//2} for k,v in events.items()})
    print(json.dumps({k:review[k] for k in ['new_trial_index','seed','automatic_terminal','r11_measured_gates_pass','complete_task_review','raw_recording_quality_pass','eligible_raw_episode','frames','exclusion_or_hold_reasons']}),flush=True)
if __name__=='__main__':main()
