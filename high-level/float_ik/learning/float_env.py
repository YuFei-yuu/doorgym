"""Independent 60 Hz floating-base environment; no locomotion policy dependency."""
import sys
import math
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isaacgym_float_ik_b1z1_basearn_push_door as demo
from isaacgym import gymapi
from kinematics import FK

from schema import SCHEMA, ACTION_NAMES, STATE_NAMES, task_success


def rotate(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c*v[0]-s*v[1], s*v[0]+c*v[1]], dtype=np.float64)


def forbidden(*args, **kwargs):
    raise RuntimeError('Expert function called in independent policy process')


class FloatEnv:
    def __init__(self, expert=False, video=False):
        self.expert_enabled, self.video_enabled = expert, video
        self.sim = None
        if not expert:
            demo.trajectory_targets = forbidden
            demo.update_arm_ik_targets = forbidden
            demo.base_ik.setup_ik_controller = forbidden

    def reset(self, seed, config=None):
        self.close()
        config = config or {}
        old = sys.argv
        sys.argv = [old[0], '--asset_root', str(Path(__file__).resolve().parents[3] / 'low-level/resources/robots/b1z1'), '--door_name', '99650089960001', '--headless', '--graphics_device_id', '0', '--sim_device', 'cuda:0', '--num_threads', '2', '--steps', '2850']
        try:
            self.args = a = demo.parse_args()
        finally:
            sys.argv = old
        self.seed = int(seed)
        rng = np.random.RandomState(seed)
        self.randomization = dict(dx=0., dy=0., dyaw=0.)
        if config.get('randomize', True):
            self.randomization = dict(dx=float(rng.uniform(-.05, .05)), dy=float(rng.uniform(-.03, .03)), dyaw=float(rng.uniform(-math.radians(2), math.radians(2))))
        a.robot_x += self.randomization['dx']
        a.robot_y += self.randomization['dy']
        a.robot_yaw += self.randomization['dyaw']
        assert a.door_auto_open_force == 0
        self.gym = g = gymapi.acquire_gym()
        self.sim, self.dt = demo.base_ik.create_sim(g, a)
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        g.add_ground(self.sim, plane)
        self.temp = tempfile.TemporaryDirectory(prefix='float_learning_')
        base_asset, arm_asset = demo.base_ik.load_robot_assets(g, self.sim, a, Path(self.temp.name))
        self.door = demo.load_door_asset(g, self.sim, a)
        data = demo.base_ik.configure_dofs(g, arm_asset, a)
        self.names, props, states, self.targets, self.lower, self.upper, defaults, _, _ = data
        self.action_lower = np.asarray(props['lower'], dtype=np.float32).copy()
        self.action_upper = np.asarray(props['upper'], dtype=np.float32).copy()
        assert self.names == ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'jointGripper'], self.names
        self.targets[-1] = a.gripper_open
        self.env, self.arm, self.actors, self.door_actor, initial = demo.create_env_actors(g, self.sim, base_asset, arm_asset, self.door, props, states, a)
        self.cameras = demo.create_low_level_cameras(g, self.env, self.arm, self.actors, a)
        assert set(self.cameras) == {'wrist', 'front'}
        if self.video_enabled:
            p = gymapi.CameraProperties()
            p.width, p.height = 640, 360
            self.overview = g.create_camera_sensor(self.env, p)
            g.set_camera_location(self.overview, self.env, gymapi.Vec3(5.2, 3.6, 3.0), gymapi.Vec3(2.5, 0, 1.1))
        arm_urdf = next(p for p in Path(self.temp.name).rglob('*.urdf') if a.ik_ee_link in p.read_text())
        self.fk = FK(arm_urdf, a.ik_ee_link, self.names)
        self.ee_index = g.find_actor_rigid_body_index(self.env, self.arm, a.ik_ee_link, gymapi.DOMAIN_ACTOR)
        self.xy = initial.astype(np.float64)
        self.yaw = float(a.robot_yaw)
        self.start_xy, self.start_yaw = self.xy.copy(), self.yaw
        self.previous = np.r_[np.zeros(3), self.targets].astype(np.float32)
        self.previous_quat = None
        self.n = 0
        self.phase = None
        self.reach_hold = self.grasp_hold = 0
        self.reached = self.grasped = False
        self.max_angle = self.max_handle = 0.
        self.phase_counts = {}
        if self.expert_enabled:
            self.ik = demo.base_ik.setup_ik_controller(g, self.sim, self.env, self.arm, arm_asset, self.names, self.lower, self.upper, a)
            heading = np.array([math.cos(self.yaw), math.sin(self.yaw)], dtype=np.float32)
            self.start = initial.copy()
            front = self.start + heading * a.robot_front_offset
            dist = max(0., float(np.dot(np.array([a.door_x, a.door_y])-front, heading))-a.stop_distance)
            self.stop = self.start + heading * dist
            self.push = demo.compute_base_push_target(a, self.stop, heading)
            self.traj = {'base_xy': self.start.copy()}
            self.home = defaults.copy()
            self.home[-1] = np.clip(a.gripper_open, self.lower[-1], self.upper[-1])
        else:
            g.prepare_sim(self.sim)
        return self.observe()

    def observe(self):
        g, a = self.gym, self.args
        q, dq = demo.get_actor_dof_state(g, self.env, self.arm)
        ee, quat = demo.get_body_pose(g, self.env, self.arm, self.ee_index)
        inverse = demo.base_ik.quat_conjugate(demo.base_ik.yaw_quat(self.yaw))
        ee_local = demo.quat_apply(inverse, ee - np.r_[self.xy, a.robot_z])
        quat = demo.base_ik.normalize_quat(demo.base_ik.quat_multiply(inverse, quat))
        if self.previous_quat is not None and np.dot(quat, self.previous_quat) < 0:
            quat = -quat
        self.previous_quat = quat.copy()
        angle = self.yaw-self.start_yaw
        state = np.r_[rotate(self.xy-self.start_xy, -self.start_yaw), math.sin(angle), math.cos(angle), self.previous[:3], q, dq, ee_local, quat, self.previous].astype(np.float32)
        assert len(state) == len(STATE_NAMES) and np.isfinite(state).all()
        if self.video_enabled:
            # Overview only: keep an escaping failed policy visible together with the door.
            distance = float(np.linalg.norm(self.xy - np.array([a.door_x, a.door_y])))
            if distance > 3.0:
                center = (self.xy + np.array([a.door_x, a.door_y])) / 2
                radius = max(4.0, distance * 1.2)
                g.set_camera_location(self.overview, self.env, gymapi.Vec3(float(center[0]+radius*.65), float(center[1]+radius), radius*.9), gymapi.Vec3(float(center[0]), float(center[1]), .7))
        g.step_graphics(self.sim)
        g.render_all_camera_sensors(self.sim)
        images, depths, sensor_depths = [], [], []
        for name in ('wrist', 'front'):
            handle = self.cameras[name]
            seg = np.asarray(g.get_camera_image(self.sim, self.env, handle, gymapi.IMAGE_SEGMENTATION)).reshape(54, 96)
            raw = np.asarray(g.get_camera_image(self.sim, self.env, handle, gymapi.IMAGE_DEPTH)).reshape(54, 96)
            # Audit channel only: preserve original Gym signed metric values, including -inf.
            sensor_depths.append(raw.astype(np.float32).copy())
            depth = np.nan_to_num(np.abs(raw), nan=0., posinf=0., neginf=0.)
            depth[depth < a.camera_depth_clip_lower] = 0
            depth = np.clip(depth, 0, a.camera_depth_clip_far).astype(np.float32)
            mask = seg == a.handle_seg_id
            masked = depth * mask
            valid = masked[mask & (masked > 0)]
            encoded = np.zeros_like(depth, dtype=np.uint8)
            if valid.size:
                lo, hi = float(valid.min()), float(valid.max())
                scaled = masked / max(hi, 1e-4) if hi-lo < 1e-4 else (masked-lo)/(hi-lo)
                encoded = (255*np.clip(scaled, 0, 1)*mask).astype(np.uint8)
            images.extend([mask.astype(np.uint8)*255, encoded])
            depths.append(depth)
        return {'state': state, 'images': np.stack(images), 'depth_m': np.stack(depths), 'gym_depth_m': np.stack(sensor_depths), 'frame': self.n}

    def expert_action(self):
        if not self.expert_enabled:
            forbidden()
        a, g = self.args, self.gym
        phase, xy, yaw, pos, quat, grip, _ = demo.trajectory_targets(self.n, a, self.door, g, self.env, self.door_actor, self.ik, self.start, self.stop, self.push, self.start_yaw, self.start_yaw+a.push_base_yaw_delta, self.traj)
        self.phase = phase
        self.phase_counts[phase] = self.phase_counts.get(phase, 0)+1
        # Original expert moves base BEFORE computing IK. This temporary mutation
        # occurs after obs_t capture; step applies the same transform, then physics.
        demo.set_robot_base_pose(g, self.env, self.actors, xy, a.robot_z, yaw)
        if phase == 'return_home':
            self.traj.setdefault('return_home_start_dofs', self.targets.copy())
            self.targets[:] = demo.lerp(self.traj['return_home_start_dofs'], self.home, self.traj.get('return_home_alpha', 0.))
        elif phase == 'hold_home':
            self.targets[:] = self.home
        else:
            demo.set_ik_target(self.ik, pos, quat)
            demo.update_arm_ik_targets(g, self.sim, self.targets, self.ik, a, len(self.targets))
            self.targets[-1] = np.clip(grip, self.lower[-1], self.upper[-1])
        velocity = rotate(np.asarray(xy, dtype=np.float64)-self.xy, -self.yaw)/self.dt
        return np.r_[velocity, (yaw-self.yaw)/self.dt, self.targets].astype(np.float32)

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32)
        if raw.shape != (10,) or not np.isfinite(raw).all():
            raise ValueError('Invalid/nonfinite policy action')
        applied = np.clip(raw, np.r_[[-.6, -.3, -.4], self.action_lower], np.r_[[.6, .3, .4], self.action_upper]).astype(np.float32)
        self.xy += rotate(applied[:2], self.yaw)*self.dt
        self.yaw += float(applied[2])*self.dt
        self.targets[:] = applied[3:]
        demo.set_robot_base_pose(self.gym, self.env, self.actors, self.xy, self.args.robot_z, self.yaw)
        self.gym.set_actor_dof_position_targets(self.env, self.arm, self.targets)
        demo.enforce_locked_door_hinge(self.gym, self.env, self.door_actor, self.door)
        q, dq = demo.get_actor_dof_state(self.gym, self.env, self.door_actor)
        forces = demo.compute_door_efforts(self.door, q, dq, self.args)
        assert self.args.door_auto_open_force == 0
        self.gym.apply_actor_dof_efforts(self.env, self.door_actor, forces)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        previous = self.previous.copy()
        self.previous = applied.copy()
        self.n += 1
        obs = self.observe()
        metrics = self.metrics()
        metrics.update(action_l2=float(np.linalg.norm(applied)), action_delta_l2=float(np.linalg.norm(applied-previous)), clipped_fraction=float(np.mean(raw != applied)))
        return obs, metrics, self.n >= 2850, applied

    def metrics(self):
        a, d = self.args, self.door
        q, _ = demo.get_actor_dof_state(self.gym, self.env, self.door_actor)
        aq, _ = demo.get_actor_dof_state(self.gym, self.env, self.arm)
        ep, _ = demo.get_body_pose(self.gym, self.env, self.arm, self.ee_index)
        hp, hq = demo.get_body_pose(self.gym, self.env, self.door_actor, d.handle_body_index)
        goal = hp + demo.quat_apply(hq, d.handle_goal_offset)
        target_local = self.fk.position(self.targets)
        actual_local = self.fk.position(aq)
        target_world = demo.quat_apply(demo.base_ik.yaw_quat(self.yaw), target_local) + np.r_[self.xy, a.robot_z]
        actual_world = demo.quat_apply(demo.base_ik.yaw_quat(self.yaw), actual_local) + np.r_[self.xy, a.robot_z]
        fk_error = float(np.linalg.norm(actual_world-ep))
        if fk_error > .005:
            raise RuntimeError('FK diagnostic mismatch: %s m' % fk_error)
        angle = float(math.degrees(a.door_motion_sign*q[0]))
        ratio = float((q[1]-d.dof_lower[1])/(d.dof_upper[1]-d.dof_lower[1]))
        error = float(np.linalg.norm(ep-goal))
        closed = float(np.clip((self.targets[-1]-a.gripper_open)/(a.gripper_closed-a.gripper_open), 0, 1))
        actual_closed = float(np.clip((aq[-1]-a.gripper_open)/(a.gripper_closed-a.gripper_open), 0, 1))
        self.reach_hold = self.reach_hold+1 if error < .08 else 0
        self.grasp_hold = self.grasp_hold+1 if error < .12 and closed > .65 and actual_closed > .35 else 0
        self.reached |= self.reach_hold >= 10
        self.grasped |= self.grasp_hold >= 10
        self.max_angle = max(self.max_angle, angle)
        self.max_handle = max(self.max_handle, float(math.degrees(q[1])))
        # Fixed door normal, and current base yaw for rear point projection.
        clearance = a.door_x - self.xy[0] - a.robot_rear_offset*abs(math.cos(self.yaw))
        return dict(step=self.n, sim_time=self.n*self.dt, signed_angle_deg=angle, signed_open_ratio=float(demo.door_hinge_open_ratio(d, float(q[0]), a)), handle_angle_deg=float(math.degrees(q[1])), handle_open_ratio=ratio, ee_to_handle_m=error, fk_goal_to_handle_m=float(np.linalg.norm(target_world-goal)), fk_ee_tracking_m=float(np.linalg.norm(target_world-ep)), fk_validation_m=fk_error, arm_base_to_handle_m=float(np.linalg.norm(np.r_[self.xy + rotate([.2, 0], self.yaw), a.robot_z+.1]-goal)), joint_tracking_l2_rad=float(np.linalg.norm(self.targets-aq)), rear_clearance_m=float(clearance), reach=self.reached, grasp=self.grasped, unlocked=bool(d.open_stage), open80=angle >= 80., ever_open80=self.max_angle >= 80., success=task_success(angle, d.open_stage, clearance), legacy_open_80deg_actual20=abs(angle) >= 20., legacy_pass20=bool(abs(angle) >= 20 and a.door_x-self.xy[0] > .7), base_x=float(self.xy[0]), base_y=float(self.xy[1]), base_yaw=self.yaw)

    def frame(self):
        return np.asarray(self.gym.get_camera_image(self.sim, self.env, self.overview, gymapi.IMAGE_COLOR)).reshape(360, 640, 4)[..., :3].copy()

    def close(self):
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None
            self.temp.cleanup()
