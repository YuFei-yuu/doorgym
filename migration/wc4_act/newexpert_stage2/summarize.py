"""Stage-two evidence summary and plots. Never grants live authorization."""
import hashlib,json,math,os,sys,subprocess,ast
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from fullcycle_compact import Arm
from fullcycle_geometry import URDF
from newexpert_stage2.push_geometry import PAD_LOCAL
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
run=Path(sys.argv[1]);out=run/(sys.argv[2] if len(sys.argv)>2 else 'audit');out.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text())
def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
base='c2e9e20f8d797a2d567e4d0984968e9f22a740bf';changed=git('diff','--name-only',base).splitlines()
assert all(p=='migration/wc4_act/fullcycle_compact.py' or p.startswith('migration/wc4_act/newexpert_stage2/') for p in changed)
old=git('show',base+':migration/wc4_act/fullcycle_compact.py');new=(ROOT/'migration/wc4_act/fullcycle_compact.py').read_text()
def definition(source,name):
 n=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name==name)
 return ast.dump(n,include_attributes=False)
def command(source):
 c=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='CompactExpert')
 return ast.dump(next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='command'),include_attributes=False)
protected=dict(base=base,head=git('rev-parse','HEAD'),branch=git('branch','--show-current'),worktree=git('status','--porcelain=v1'),main=git('rev-parse','main'),changed=changed,
    old_Arm_unchanged=definition(old,'Arm')==definition(new,'Arm'),old_ClosingCurve_unchanged=definition(old,'ClosingCurve')==definition(new,'ClosingCurve'),
    unload_withdraw_fold_bypass_unchanged=old.split('    def command(')[1].split('        elif self.phase=="compact_rendezvous":')[0]==new.split('    def command(')[1].split('        elif self.phase=="compact_rendezvous":')[0],
    seek_close_release_final_unchanged=old.split('        elif self.phase=="compact_seek":')[1]==new.split('        elif self.phase=="compact_seek":')[1],all_other_tracked_files_unchanged=True,stage1_evidence_unmodified=True,
    branch_cleanup='User requested transfer to existing stage1 branch and deletion of stage2 branch; no evidence removed.')
assert all(protected[k] for k in ['old_Arm_unchanged','old_ClosingCurve_unchanged','unload_withdraw_fold_bypass_unchanged','seek_close_release_final_unchanged'])
write('protected-scope.json',protected);(out/'source.diff').write_text(git('diff',base));(out/'refs.txt').write_text(git('show-ref','--heads'))
refs={}
for label in ['old','new']:
 d=read(run/'reference-complete'/(label+'-reference.json'));rows=d['sweep_samples'];grid=[r for r in rows if r['case'].startswith('grid_')]
 def stats(rs):return dict(samples=len(rs),max_position_error_m=max(r['position_error_m'] for r in rs),max_orientation_or_axis_error_rad=max(r['orientation_error_rad'] for r in rs),min_joint_slack_rad=min(r['joint_slack_rad'] for r in rs),saturated_samples=sum(r['joint_slack_rad']<1e-6 for r in rs))
 refs[label]=dict(comparable_grid=stats(grid),all=stats(rows),worst_position=max(rows,key=lambda r:r['position_error_m']))
write('reachability.json',refs)
traces={};simplicity={}
for p in sorted(run.glob('*/trace.jsonl')):
 rs=[json.loads(s) for s in p.open()];traces[p.parent.name]=rs
 a=np.array([r['action'] for r in rs]);pad=np.array([r['pad'] for r in rs]);ee=np.array([r['ee'] for r in rs]);R=np.array([r['ee_rotation'] for r in rs]);drot=Rotation.from_matrix(R[1:]@R[:-1].transpose(0,2,1)).magnitude()
 close=[r for r in rs if r['phase']=='close'];cp=np.array([r['pad'] for r in close]);theta=np.array([r['theta'] for r in close]);norm=np.c_[np.cos(theta),np.sin(theta),np.zeros(len(theta))]
 along=np.sum(np.diff(cp,axis=0)*norm[:-1],axis=1)
 # q6 is not globally equal to tool roll; plot as an interpretable joint trace.
 target_z=np.array([r['z'] for r in rs]);events=[]
 for prev,r in zip(rs,rs[1:]):
  if prev['phase']!=r['phase']:
   events.append(dict(t=r['t'],previous=prev['phase'],phase=r['phase'],q_delta=float(max(abs(np.array(r['q'])-prev['q']))),pad_step_m=float(np.linalg.norm(np.array(r['pad'])-prev['pad']))))
 simplicity[p.parent.name]=dict(base_z_min=float(target_z.min()),base_z_max=float(target_z.max()),base_height_constant=bool(target_z.max()==target_z.min()),
    contact_height_close_min=float(cp[:,2].min()),contact_height_close_max=float(cp[:,2].max()),
    close_total_backward_projection_m=float(-along[along<0].sum()),close_largest_backward_step_m=float(max(0,-along.min())),
    max_ee_rotation_step_rad=float(drot.max()),max_joint_command_rate_rad_s=float(abs(np.diff(a[:,2:],axis=0)).max()/.02),events=events,
    caveat='Whole-suffix tool height changes during required fold/deploy/retract. Constant height is a closing contact design; base height is constant throughout. Rotation increment includes intentional deployment/withdrawal, not only closing.')
write('simplicity.json',simplicity)
clear={}
for p in sorted(run.glob('clearance*/clearance.json')):
 d=read(p);samples=d['samples'];push=[s['contact_design'] for s in samples if s['phase']=='close'];m=d['minimums']
 clear[p.parent.name]=dict(minimums=m,finite_wall_min_m=min(v['gap_m'] for k,v in m.items() if 'finite_walls' in k),
    push_axis_error_max_rad=max(s['push_axis_error_rad'] for s in push),pad_gap_range_m=[min(s['pad_panel_normal_gap_m'] for s in push),max(s['pad_panel_normal_gap_m'] for s in push)],
    contact_face_radial_range_m=[min(s['face_radial_range_m'][0] for s in push),max(s['face_radial_range_m'][1] for s in push)],
    contact_face_height_range_m=[min(s['face_height_range_m'][0] for s in push),max(s['face_height_range_m'][1] for s in push)],
    nonend_overlaps_after_unload={k:v for k,v in m.items() if not k.startswith('compact_unload/') and not k.endswith('/end_panel') and v['gap_m']<0})
write('clearance-contact.json',clear)
# Time and path plots from saved synthetic traces, explicitly not execution.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(14,9))
for label,color in [('old-nominal','#777777'),('new-linear-nominal','#006eb8')]:
 rs=traces[label];t=np.array([r['t'] for r in rs]);b=np.array([r['base'] for r in rs]);pad=np.array([r['pad'] for r in rs]);a=np.array([r['action'] for r in rs]);z=np.array([r['z'] for r in rs]);phase=np.array([r['phase'] for r in rs]);close=phase=='close'
 axes[0,0].plot(b[:,0],b[:,1],color=color,label=label+' base');axes[0,0].plot(pad[close,0],pad[close,1],':',color=color,label=label+' pad (close)')
 axes[0,1].plot(t,a[:,0],color=color,label=label+' v');axes[0,1].plot(t,a[:,1],'--',color=color,alpha=.55,label=label+' yaw rate')
 axes[1,0].plot(t,z,color=color,label=label+' base z');axes[1,0].plot(t,pad[:,2],':',color=color,label=label+' pad z')
 axes[1,1].plot(t,a[:,7],color=color,label=label+' joint6');
 for ax in axes.flat:ax.grid(alpha=.2)
for i,act in enumerate(read(run/'clearance-linear/finite-wall-binding.json')['0']['actors']):
 from matplotlib.patches import Rectangle
 c=act['center'];s=act['size'];axes[0,0].add_patch(Rectangle((c[0]-s[0]/2,c[1]-s[1]/2),s[0],s[1],color='#ad6221',alpha=.4,label='finite walls' if i==0 else None))
axes[0,0].set(xlabel='Reference x (m)',ylabel='Reference y (m)',title='Full suffix + actual finite wall opening');axes[0,0].set_aspect('equal');axes[0,0].legend(fontsize=7)
axes[0,1].set(xlabel='Synthetic suffix time (s)',ylabel='m/s or rad/s',title='Base commands');axes[0,1].legend(fontsize=7)
axes[1,0].set(xlabel='Synthetic suffix time (s)',ylabel='Height (m)',title='No base lift compensation');axes[1,0].legend(fontsize=7)
axes[1,1].set(xlabel='Synthetic suffix time (s)',ylabel='Joint6 command (rad)',title='Free-roll IK still follows a continuous joint solution');axes[1,1].legend(fontsize=7)
fig.suptitle('Stage 1 vs Stage 2 | Synthetic exact-tracking feedback, NOT physics',fontsize=15);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'motion-comparison.png',dpi=170);fig.savefig(out/'motion-comparison.svg');plt.close(fig)
# Explicit face geometry illustration at nominal 45 degree target.
from fullcycle_compact import CompactExpert
from newexpert_stage2.push_geometry import PadArm,PadCurve
from fullcycle_geometry import transform,apply
robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf');arm=PadArm(robot);c=read(Path(__file__).with_name('geometry.json'))['compact_expert'];curve=PadCurve(c);theta=math.pi/4;ref=curve.at(theta);B=transform([*ref[:2],.5],[0,0,ref[2]])
q,err=arm.solve(np.linalg.inv(B)@curve.tool(theta),c['hover_ik_seed'],1.,120);E=B@arm.fk(q)
from fullcycle_collision import collision_pieces
rp=robot.fk(dict(zip(arm.names+['jointGripper'],np.r_[q,0])),B)
fig,ax=plt.subplots(figsize=(8,7));radial=np.array([-math.sin(theta),math.cos(theta),0]);normal=np.array([math.cos(theta),math.sin(theta),0]);hinge=np.array([2.467,-.463,.01])
for part in collision_pieces(robot.path):
 if part.link not in ['gripperStator','gripperMover','link06','wrist_down_mount','wrist_up_mount']:continue
 v=part.world(rp[part.link])[0];x=(v-hinge)@normal;y=(v-hinge)@radial
 from scipy.spatial import ConvexHull
 pts=np.c_[x,y];h=ConvexHull(pts);idx=np.r_[h.vertices,h.vertices[0]]
 ax.plot(pts[idx,0],pts[idx,1],alpha=.7,label=part.link if part.name=='0' else None)
face=apply(E,np.array([[.015,y,z] for y in [-.03,.03] for z in [-.03,-.01]]));ax.scatter((face-hinge)@normal,(face-hinge)@radial,c='red',label='chosen stator distal face')
ax.axvline(-.002,color='black',label='near wood surface');ax.axhspan(.661,.795,color='gray',alpha=.15,label='glass/gasket radial zone (reference)')
ax.set(xlabel='Door-normal coordinate (m); push toward +x',ylabel='Hinge radial coordinate (m)',title='Nominal 45-deg contact geometry | source hull projection');ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'contact-design.png',dpi=170);plt.close(fig)
files=[p for p in (ROOT/'migration/wc4_act').rglob('*') if p.is_file() and '__pycache__' not in str(p)]
for subtree in ['high-level/data/asset/a2wz1','high-level/data/asset/door_set/wc4']:
 files.extend(p for p in (ROOT/subtree).rglob('*') if p.is_file())
files += [ROOT/'high-level/float_ik/door_common.py']
write('source-asset-sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))})
write('budget.json',dict(physical_episodes=0,approved_six_trials_consumed=0,collection_episodes=0,optimizer_updates=0,policy_evaluations=0,gpu_leases=0,monitors_started=0,
    bounded_probe_ik_samples=2*3*9*31,reference_ik_samples=3006,partial_failed_reference_samples=819+1,synthetic_steps={n:len(rs) for n,rs in traces.items()},clearance_cases=len(clear),historical_handoff_states=221,
    failure='First reference output raised duplicate joint_slack_rad field TypeError after old grid; preserved reference and reference.log; output-only fix rerun in reference-complete.',branch_cleanup='Both stage2 branch refs deleted after preserving commits on stage1; no evidence deletion.'))
write('review-status.json',dict(status='FROZEN_CPU_ONLY_FOR_INDEPENDENT_REVIEW',physical_success=False,live_authorized_this_round=False,
    source_risks={k:v['nonend_overlaps_after_unload'] for k,v in clear.items()},
    limitations=['Finite sample source geometry and synthetic feedback do not certify dynamics, contact force or complete task success.','Full randomized opening was not re-executed.','Historical handoff contact explained by old raw logs; no new physical demonstration of withdrawal.','Release uses inherited lower-weight pose solve; residuals are separately reported, not silently relaxed.']))
print(json.dumps(dict(protected=protected,reachability=refs,simplicity=simplicity,clearance={k:{kk:vv for kk,vv in v.items() if kk!='minimums'} for k,v in clear.items()}),indent=2))
