"""Static asset checks and reference drawings. NOT a physical rollout."""
import argparse,json,math,os,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'migration/wc4_act'))
from author_fast5.geometry import DoorFrame,closing_reference,transform,apply
from fullcycle_geometry import URDF
from fullcycle_collision import collision_pieces,signed_gap
from newexpert_stage2.finite_walls import walls


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--effective',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    rows=[json.loads(x) for x in (a.run/'trace.jsonl').read_text().splitlines()]
    effective=json.loads(a.effective.read_text())
    robot=URDF(ROOT/'high-level/data/asset/a2wz1/urdf/a2wz1.urdf',include_visual=True)
    frame=DoorFrame(transform([2.61,0,.01],[0,0,math.pi/2]));door=frame.door
    legs={s+j:v for s in ['FL','FR','RL','RR'] for j,v in [('_hip_joint',0),('_thigh_joint',.8),('_calf_joint',-1.5)]}
    names=['joint%d'%i for i in range(1,7)]+['jointGripper']
    def poses(B,q):return robot.fk(dict(legs,**dict(zip(names,[*q,-1.1775000095367432]))),frame.base_transform(B))
    def hull(B,q,arm=True):
        fs=poses(B,q)
        clouds=[apply(fs[n],v) for n,v in robot.clouds.items() if arm or not n.startswith(('link','wrist','gripper','ee_'))]
        pts=np.concatenate(clouds)[:,:2];return pts[ConvexHull(pts).vertices]
    R=collision_pieces(robot.path);D=collision_pieces(door.path)
    W,wm=walls(effective,json.loads((door.path.parent/'bounding_box.json').read_text()))
    samples=[r for i,r in enumerate(rows) if i%(10 if r['phase']=='close' else 50)==0]+[rows[-1]]
    # Nominal turn/relocation geometry at HOME (not observed leg poses).
    start=np.array([1.26,.18,math.pi]);end=closing_reference(cfg,frame,0)[0]
    for yaw in np.linspace(math.pi,2*math.pi+math.radians(cfg["closing_start_yaw_deg"]),10):
        samples.append(dict(phase='front_turn_HOME',t=0,base=[*end[:2],yaw],q=[0]*6,theta=math.pi/2))
    finalbase=closing_reference(cfg,frame,cfg['close_s'])[0]
    for yaw in np.linspace(finalbase[2],-math.pi,13):
        samples.append(dict(phase='restore_heading_HOME',t=0,base=[*finalbase[:2],yaw],q=[0]*6,theta=0.))
    checks=[]
    for k,r in enumerate(samples):
        fs=poses(r['base'],r['q']);dp=door.fk({'door_hinge_joint':r['theta'],'handle_spindle_joint':0},frame.actor)
        bw=[(p,p.world(dp[p.link])) for p in D]+W
        minimum={'external':(1e9,None),'gripper_panel':(1e9,None)}
        for rp in R:
            av=rp.world(fs[rp.link])
            for dpiece,bv in bw:
                cat='gripper_panel' if rp.link in ('gripperStator','gripperMover') and dpiece.link=='door_panel' else 'external'
                broad=float(np.maximum(bv[0].min(0)-av[0].max(0),av[0].min(0)-bv[0].max(0)).max())
                if broad>minimum[cat][0]:continue
                gap=signed_gap(av,bv) if broad<.03 else broad
                if gap<minimum[cat][0]:minimum[cat]=(gap,[rp.link,rp.name,dpiece.link,dpiece.name])
        whole=robot.cloud(fs)[:,0].max()
        checks.append(dict(phase=r['phase'],t=r['t'],whole_max_x=float(whole),minimums={k:{'gap_m':v[0],'pair':v[1]} for k,v in minimum.items()}))
    (a.run/'static-geometry.json').write_text(json.dumps(dict(samples=checks,walls=wm,
        scope='Discrete source collision convex hulls against door/walls; fixed nominal leg targets; not physics or continuous clearance. Gripper-panel overlaps are reported separately, never automatically accepted.'),indent=2)+'\n')
    close=[r for r in rows if r['phase']=='close'];base=np.array([r['base'] for r in close]);ee=np.array([r['target'] for r in close]);hinge=frame.hinge[:2]
    fig,(ax,detail)=plt.subplots(1,2,figsize=(15,7.8),gridspec_kw={'width_ratios':[1,1.1]})
    for panel in [ax,detail]:
        panel.set_aspect('equal');panel.grid(alpha=.18);panel.set_xlabel('World x (m)');panel.set_ylabel('World y (m)')
        panel.axvline(2.5,color='#666666',lw=1,ls=':',label='Door / wall reference plane')
        for ang in [90,45,0]:
            t=math.radians(ang);tip=hinge+.93*np.array([-math.sin(t),math.cos(t)])
            panel.plot(*np.array([hinge,tip]).T,color='#b47a40',lw=2,alpha=.7)
            panel.text(tip[0]-.02,tip[1]+.04,f'{ang}°',color='#835626',fontsize=10)
        panel.scatter(*hinge,color='#6b4929',s=28,zorder=5)
        for piece,world in W:
            pts=world[0][:,:2];panel.add_patch(Polygon(pts[ConvexHull(pts).vertices],facecolor='#d8d8d8',edgecolor='#aaa',alpha=.6))
        panel.plot(base[:,0],base[:,1],color='#147d92',ls='--',lw=3,label=f"Closing base: {cfg['follow_m']:.2f} m")
        panel.plot(ee[:,0],ee[:,1],color='#bb3f60',lw=2,label='EE target (clock-driven)')
    # Author nominal base targets, from unchanged skill / frozen nominal config.
    ax.plot([4.1,3.2,2.0,1.26],[.18]*4,color='#5075b8',lw=2.5,label='Author opening / traverse centerline')
    ax.annotate('',xy=(2.8,.18),xytext=(3.25,.18),arrowprops={'arrowstyle':'->','color':'#5075b8','lw':2})
    adjust=np.array([[1.26,.18],base[0,:2]])
    ax.plot(*adjust.T,color='#137f56',lw=4,label=f'Front adjustment: {np.linalg.norm(adjust[1]-adjust[0]):.2f} m nominal')
    for frac,col in [(0,'#197c9a'),(.5,'#739cab'),(1,'#284b63')]:
        r=close[round(frac*(len(close)-1))];B=np.array(r['base']);q=r['q']
        footprint=hull(B,q,False)
        detail.add_patch(Polygon(footprint,fill=False,edgecolor=col,alpha=.6,lw=1.4))
        F=poses(B,q);arm=np.array([F[n][:3,3] for n in ['link00','link02','link03','link04','link06','ee_gripper_link']])
        detail.plot(arm[:,0],arm[:,1],color=col,lw=1.6,marker='o',markersize=2)
        detail.scatter(*B[:2],color=col,s=35,zorder=5)
        detail.arrow(B[0],B[1],.18*math.cos(B[2]),.18*math.sin(B[2]),width=.006,color=col,length_includes_head=True)
    detail.annotate(f'Closing start\n({base[0,0]:.2f}, {base[0,1]:.2f}), yaw {cfg["closing_start_yaw_deg"]:.0f}°',xy=base[0,:2],xytext=(.6,.75),fontsize=10,arrowprops={'arrowstyle':'->'})
    detail.annotate(f'Final base\n({base[-1,0]:.2f}, {base[-1,1]:.2f}), yaw 0°',xy=base[-1,:2],xytext=(1.83,.85),fontsize=10,arrowprops={'arrowstyle':'->'})
    ax.set_xlim(.45,4.35);ax.set_ylim(-1.03,1.05);detail.set_xlim(.55,2.72);detail.set_ylim(-1.03,1.12)
    ax.set_title('Original traverse endpoint, no micro-relocation',fontsize=12)
    detail.set_title('Closing detail: base poses, arm and door references',fontsize=12)
    ax.legend(loc='lower right',fontsize=8,framealpha=.95)
    fig.suptitle('Planned expert reference — NOT a rollout',fontsize=18,fontweight='bold')
    fig.text(.5,.025,'Door angles are drawing references, not feedback. Footprints use asset bounds and nominal leg targets.\nBase height / pitch remain the sampled episode values. Physical contact, feet and closure remain untested.',ha='center',fontsize=10,color='#555')
    fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig(a.run/'top-view.png',dpi=180);fig.savefig(a.run/'top-view.svg');plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(10,7),sharex=True)
    tt=np.array([r['t'] for r in close]);qq=np.array([r['q'] for r in close])
    axs[0].plot(tt,np.degrees([r['theta'] for r in close]),label='Path angle');axs[0].plot(tt,np.degrees(base[:,2]),label='Base yaw');axs[0].set_ylabel('deg');axs[0].legend()
    axs[1].plot(tt,base[:,0]-base[0,0],label='Forward base distance');axs[1].set_ylabel('m');axs[1].legend()
    axs[2].plot(tt,qq);axs[2].set_ylabel('Arm joint rad');axs[2].set_xlabel('Closing reference time (s)')
    for x in axs:x.grid(alpha=.2)
    fig.suptitle('Clock-driven targets / ideal DLS joints — NOT measured motion');fig.tight_layout();fig.savefig(a.run/'reference-timing.png',dpi=160)
    print(json.dumps({'geometry_samples':len(checks),'minimum_external_m':min(x['minimums']['external']['gap_m'] for x in checks),'maximum_whole_x':max(x['whole_max_x'] for x in checks)}))
if __name__=='__main__':main()
