"""Continuous forward nominal approach, joined to the existing deploy tail.

Builds Joint9 commands only. No scene creation, pose writes, or door control.
The geometric curve passes counterclockwise around the free edge. Retiming
keeps the existing base speed/rate limits and stops only at the contact handoff.
"""
import math
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fullcycle_geometry import transform


def _quintic(q0, v0, a0, q1, v1, a1, duration, times):
    """Endpoint position/velocity/acceleration interpolation."""
    q0, v0, a0, q1, v1, a1 = [np.asarray(x, float) for x in (q0,v0,a0,q1,v1,a1)]
    T=float(duration)
    c0=q0; c1=v0*T; c2=.5*a0*T*T
    rhs=np.stack([q1-c0-c1-c2,v1*T-c1-2*c2,a1*T*T-2*c2])
    c3,c4,c5=np.linalg.solve(np.array([[1,1,1],[3,4,5],[6,12,20]],float),rhs)
    u=np.clip(np.asarray(times,float)/T,0,1)[:,None]
    return c0+u*(c1+u*(c2+u*(c3+u*(c4+u*c5))))


def integrate_commands(actions, start_pose, dt):
    """Same forward-Euler planar motion as the existing Joint9 integrator."""
    xy=np.asarray(start_pose[:2],np.float32).copy();yaw=float(start_pose[2]);poses=[]
    for action in actions:
        heading=np.array([math.cos(yaw),math.sin(yaw)],np.float32)
        xy=(xy+heading*(float(action[0])*dt)).astype(np.float32)
        yaw+=float(action[1])*dt
        poses.append([*xy,yaw])
    return np.asarray(poses)


def build_approach(start_pose, start_action, previous_action, recipe, qclose, dt,
                   robot=None, measured_start_q=None, start_step=0):
    start_pose=np.asarray(start_pose,float);start_action=np.asarray(start_action,float)
    previous_action=np.asarray(previous_action,float)
    goal=np.asarray(recipe["contact_base_pose"],float)
    r1,r2,r3=map(float,recipe["radii"])
    # The source task is nominal: initial heading west, door free edge west.
    psi0=float(start_pose[2]%(2*math.pi))
    if abs(psi0-math.pi)>.05 or start_action[0]<=0:
        raise ValueError("Smooth approach requires the nominal forward traversal handoff")
    psi_goal=float(goal[2]%(2*math.pi))
    if not 0<psi_goal<math.pi:
        raise ValueError("Unsupported closing contact heading")
    c1=start_pose[:2]-r1*np.array([math.sin(psi0),-math.cos(psi0)])
    end1=c1+np.array([-r1,0.])
    c3=goal[:2]-r3*np.array([math.sin(psi_goal),-math.cos(psi_goal)])
    start3=c3+np.array([0.,-r3])
    c2=np.array([end1[0]+r2,start3[1]+r2])
    start2=c2+np.array([-r2,0.]);end2=c2+np.array([0.,-r2])
    if start2[1]>=end1[1] or start3[0]<=end2[0]:
        raise ValueError("Approach arc recipe does not fit the current handoff")
    points=[]
    def append(p):
        if not points or np.linalg.norm(np.array(p)-points[-1])>1e-8:
            points.append(np.asarray(p,float))
    def arc(c,r,a,b,n):
        for psi in np.linspace(a,b,n):
            append(c+r*np.array([math.sin(psi),-math.cos(psi)]))
    def line(a,b):
        for u in np.linspace(0,1,max(3,int(np.linalg.norm(b-a)/.10)+1)):
            append((1-u)*a+u*b)
    arc(c1,r1,psi0,1.5*math.pi,13)
    line(end1,start2)
    arc(c2,r2,1.5*math.pi,2*math.pi,9)
    line(end2,start3)
    arc(c3,r3,0,psi_goal,13)
    points=np.asarray(points);knots=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    tangent0=np.array([math.cos(psi0),math.sin(psi0)])
    tangent1=np.array([math.cos(psi_goal),math.sin(psi_goal)])
    spline=make_interp_spline(knots,points,k=5,bc_type=([(1,tangent0),(2,np.zeros(2))],[(1,tangent1),(2,np.zeros(2))]))
    u=np.linspace(0,knots[-1],max(1000,int(knots[-1]/.0015)))
    p=spline(u);d=spline(u,1);dd=spline(u,2)
    norm=np.linalg.norm(d,axis=1)
    if norm.min()<.2:raise ValueError("Approach spline has a cusp")
    heading=np.unwrap(np.arctan2(d[:,1],d[:,0]))
    heading+=round((float(start_pose[2])-heading[0])/(2*math.pi))*2*math.pi
    curvature=(d[:,0]*dd[:,1]-d[:,1]*dd[:,0])/norm**3
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    vmax=float(recipe["max_speed"]);wmax=float(recipe["max_yaw_rate"])
    accel=float(recipe["linear_acceleration"])
    velocity=np.minimum(vmax,wmax/np.maximum(abs(curvature),1e-8))
    velocity[0]=float(start_action[0]);velocity[-1]=0.
    for i in range(1,len(velocity)):
        velocity[i]=min(velocity[i],math.sqrt(velocity[i-1]**2+2*accel*(distance[i]-distance[i-1])))
    for i in range(len(velocity)-2,-1,-1):
        velocity[i]=min(velocity[i],math.sqrt(velocity[i+1]**2+2*accel*(distance[i+1]-distance[i])))
    if abs(velocity[0]-start_action[0])>1e-5:
        raise ValueError("Approach cannot preserve the incoming forward speed")
    times=np.r_[0.,np.cumsum(2*np.diff(distance)/(velocity[:-1]+velocity[1:]))]
    T=float(times[-1]);n=int(math.ceil(T/dt))
    # Resume the original 25 Hz deploy tail on its normal even control step.
    if (int(start_step)+n)%2:n+=1
    command_times=(np.arange(n)+1)*dt
    midpoint=(np.arange(n)+.5)*dt
    v=np.interp(midpoint,times,velocity)
    # Translate along midpoint headings under the existing Euler convention.
    next_heading=np.interp((np.arange(n)+1.5)*dt,times,heading)
    w=np.diff(np.r_[start_pose[2],next_heading])/dt
    home=np.asarray(recipe["home_q"],float)
    origin=np.asarray(recipe["deploy_origin_q"],float);qclose=np.asarray(qclose,float)
    elapsed=int(recipe["deploy_entry_offset_steps"])*dt
    alpha=elapsed/4.
    blend=alpha**3*(10-15*alpha+6*alpha**2)
    qend=origin+(qclose-origin)*blend
    vend=(qclose-origin)*30*alpha**2*(1-alpha)**2/4
    aend=(qclose-origin)*60*alpha*(1-alpha)*(1-2*alpha)/16
    retract=float(recipe["retract_seconds"]);deploy=float(recipe["deploy_seconds"])
    if T<retract+deploy+1:raise ValueError("Insufficient continuous-approach time for arm transitions")
    q=np.tile(home,(n,1))
    withdraw_diagnostics={}
    if "withdraw_seconds" in recipe:
        if robot is None or measured_start_q is None:
            raise ValueError("Withdrawal requires the measured arm state and source kinematics")
        withdraw=float(recipe["withdraw_seconds"])
        if not 0<withdraw<retract:raise ValueError("Invalid withdrawal time")
        bases=integrate_commands(np.column_stack([v,w,np.zeros((n,7))]),start_pose,dt)
        wi=int(np.argmin(abs(command_times-withdraw)))
        base_z=float(recipe.get("base_z",.5));pitch=float(recipe.get("base_pitch",0.))
        B=transform([*bases[wi,:2],base_z],[0,pitch,bases[wi,2]])
        E=transform([*start_pose[:2],base_z],[0,pitch,start_pose[2]])@robot.arm_fk(measured_start_q[:6])
        E[:3,3]+=np.asarray(recipe["withdraw_delta_world"],float)
        target=np.linalg.inv(B)@E
        lo=np.asarray(recipe["joint_lower"],float);hi=np.asarray(recipe["joint_upper"],float)
        seed=np.clip(np.asarray(measured_start_q[:6],float),lo+1e-8,hi-1e-8)
        def residual(values):
            F=robot.arm_fk(values)
            return np.r_[4*(F[:3,3]-target[:3,3]),
                         .1*Rotation.from_matrix(target[:3,:3]@F[:3,:3].T).as_rotvec(),
                         .02*(values-seed)]
        solved=least_squares(residual,seed,bounds=(lo,hi),max_nfev=120)
        F=robot.arm_fk(solved.x)
        error=float(np.linalg.norm(F[:3,3]-target[:3,3]))
        if error>.015:raise ValueError("Withdrawal position is not reachable within the existing IK tolerance")
        via=np.r_[solved.x,home[-1]]
        a=command_times<withdraw
        q[a]=_quintic(start_action[2:],(start_action[2:]-previous_action[2:])/dt,np.zeros(7),via,np.zeros(7),np.zeros(7),withdraw,command_times[a])
        a=(command_times>=withdraw)&(command_times<retract)
        q[a]=_quintic(via,np.zeros(7),np.zeros(7),home,np.zeros(7),np.zeros(7),retract-withdraw,command_times[a]-withdraw)
        withdraw_diagnostics={
            "withdraw_position_error_m":error,
            "withdraw_orientation_error_rad":float(np.linalg.norm(Rotation.from_matrix(target[:3,:3]@F[:3,:3].T).as_rotvec())),
            "withdraw_q":via.tolist(),
        }
    else:
        a=command_times<retract
        q[a]=_quintic(start_action[2:],(start_action[2:]-previous_action[2:])/dt,np.zeros(7),home,np.zeros(7),np.zeros(7),retract,command_times[a])
    b=command_times>T-deploy
    q[b]=_quintic(home,np.zeros(7),np.zeros(7),qend,vend,aend,deploy,command_times[b]-(T-deploy))
    actions=np.column_stack([v,w,q])
    if not np.isfinite(actions).all() or np.any(v<0) or np.max(v)>.250001 or np.max(abs(w))>.600001:
        raise ValueError("Approach violates existing Joint9 base command limits")
    if "joint_lower" in recipe:
        if (np.any(q[:,:6]<np.asarray(recipe["joint_lower"])-1e-6)
                or np.any(q[:,:6]>np.asarray(recipe["joint_upper"])+1e-6)
                or np.any(q[:,6]<home[6]-1e-6) or np.any(q[:,6]>1e-6)):
            raise ValueError("Approach joint targets exceed the existing controller limits")
    poses=integrate_commands(actions,start_pose,dt)
    reference=np.column_stack([np.interp(command_times,times,p[:,0]),np.interp(command_times,times,p[:,1]),np.interp(command_times,times,heading)])
    return {
        "actions":actions,"base":poses,"reference":reference,"time":command_times,
        "duration_s":n*dt,"contact_position_error_m":float(np.linalg.norm(poses[-1,:2]-goal[:2])),
        "contact_heading_error_rad":float(abs((poses[-1,2]-goal[2]+math.pi)%(2*math.pi)-math.pi)),
        "minimum_forward_speed_before_final_braking":float(v[command_times<T-1.].min()),
        "continuous_curve_duration_s":T,"waypoints":points,
        **withdraw_diagnostics,
    }
