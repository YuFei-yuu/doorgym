"""Source whole-body push task with the source split-asset floating robot.

The task loop is upstream.main; this class supplies only the simulator backend.
No low-level walking policy, synthetic leg observations or alternative expert.
"""
import json
import math
import sys
import tempfile
import shutil
import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'high-level'))
sys.path.insert(0, str(ROOT / 'high-level/float_ik'))
import play_b1z1_push_with_door_asset_camera as upstream
import isaacgym_visualize_b1z1_basearn as loader
import isaacgym_float_ik_b1z1_basearn_push_door as float_source
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import (quat_apply, quat_rotate_inverse,
                                quat_from_euler_xyz, euler_from_quat, orientation_error)
import numpy as np
import torch
import yaml
from dp.door_dp_common import (RawDoorDPRecorder, get_door_dp_action,
                              dp_image_inputs_from_camera_tensors,
                              make_door_dp_log_record, apply_door_dp_action, ACTION_NAMES)
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.utils.helpers import class_to_dict
from kinematics import FK

SCHEMA = 'float_push_upstream_ee_v3'
PROFILE = 'upstream_wholebody_float_v3'


class WholebodyFloat(upstream.ManipLocoDoorAsset):
    execution_profile = PROFILE
    instance = None

    @classmethod
    def load_runtime(cls, args):
        # The original loader reads every configured bounding box before selecting
        # the default assets. Select source defaults first on partial asset installs.
        cfg = yaml.safe_load(Path(args.door_cfg).read_text())
        asset = cfg['env']['asset']
        block = asset['load_block']
        specs = upstream._sorted_asset_items(asset['trainAssets'][block])
        names = upstream.DEFAULT_DOOR_ASSET_NAMES
        selected = [s for name in names for s in specs if s['name'] == name]
        if not selected:
            raise ValueError('Source default door assets absent from configuration')
        asset['trainAssets'][block] = {str(i):s for i,s in enumerate(selected)}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
            yaml.safe_dump(cfg, f)
            f.flush()
            return upstream._load_door_runtime(f.name)

    @classmethod
    def configure(cls, args):
        if args.num_envs != 1:
            raise ValueError('Floating backend currently records one isolated attempt per process')
        if not args.float_output or args.record_dp_dataset:
            raise ValueError('Use --float_output; the backend supplies pre-execution raw recording')
        cls.args = args
        cls.execution_profile = PROFILE + ('_forward_pass' if args.float_pass_mode == 'forward' else '')

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        type(self).instance = self
        self.closed = False
        self.cfg, self.num_envs = cfg, 1
        self.device = torch.device('cpu' if self.args.float_execution == 'source' else sim_device)
        output = Path(self.args.float_output)
        output.mkdir(parents=True, exist_ok=False)
        self.output = output
        self.viewer = None
        self.dt = sim_params.dt * cfg.control.decimation
        self.physics_dt = sim_params.dt
        self.num_actions = 0
        self.max_episode_length = int(cfg.env.episode_length_s / self.dt)
        self.episode_length_buf = torch.zeros(1, dtype=torch.long, device=self.device)
        self.physical_steps = 0
        self.time_out_buf = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.commands = torch.zeros(1, 3, device=self.device)
        self.env_origins = torch.zeros(1, 3, device=self.device)
        self.gym = gymapi.acquire_gym()
        sim_params.physx.num_threads = 2
        if self.args.float_execution == 'source':
            sim_params.use_gpu_pipeline = False
        graphics = int(self.args.graphics_device_id or 0)
        self.sim = self.gym.create_sim(0, graphics, physics_engine, sim_params)
        if self.sim is None:
            raise RuntimeError('create_sim failed')
        ground = gymapi.PlaneParams()
        ground.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, ground)
        self.asset_temp = tempfile.TemporaryDirectory(prefix='upstream-float-v3-')
        load_args = SimpleNamespace(
            asset_root=str(ROOT/'low-level/resources/robots/b1z1'),
            asset_file='urdf/b1z1_basearn.urdf', single_asset=False,
            disable_arm_visual_flip=False, base_visual_flip=False,
            no_disable_gravity=False, flip_visual_attachments=False,
        )
        base_asset, arm_asset = loader.load_robot_assets(self.gym, self.sim, load_args, Path(self.asset_temp.name))
        (self.output/'assets').mkdir()
        for path in Path(self.asset_temp.name).rglob('*.urdf'):
            shutil.copy2(path, self.output/'assets'/path.name)
        self.body_names = self.gym.get_asset_rigid_body_names(arm_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(arm_asset)
        shapes = self.gym.get_asset_rigid_shape_properties(arm_asset)
        self.cfg.asset.gripper_name = loader.EE_GRIPPER_LINK
        self.gym.set_asset_rigid_shape_properties(arm_asset, self._apply_gripper_shape_contact_props(arm_asset, shapes))
        dof_data = loader.configure_dofs(self.gym, arm_asset, SimpleNamespace(
            stiffness=400.0, damping=40.0, range_scale=1.0, speed_scale=1.0,
            joint_filter='', ik_demo=True, ik_target_pose=''))
        props = dof_data[1]
        self.dof_names = self.gym.get_asset_dof_names(arm_asset)
        self.num_dofs = len(self.dof_names)
        if self.num_dofs != 7 or self.dof_names[:6] != ['joint%d' % i for i in range(1, 7)]:
            raise ValueError('Unexpected source split robot DOF order: %s' % self.dof_names)
        props['driveMode'][:] = gymapi.DOF_MODE_POS
        props['stiffness'][:] = 400
        props['damping'][:] = 40
        props['stiffness'][-1] = self.args.gripper_stiffness
        props['damping'][-1] = self.args.gripper_damping
        props['friction'][-1] = self.args.gripper_joint_friction
        self.lower = torch.tensor(props['lower'].copy(), device=self.device)
        self.upper = torch.tensor(props['upper'].copy(), device=self.device)
        # Source floating entry initializes each joint at clamped zero.
        self.default_dof_pos = torch.tensor(dof_data[6], device=self.device)
        self._load_door_assets()
        env = self.gym.create_env(self.sim, gymapi.Vec3(-5, -5, 0), gymapi.Vec3(5, 5, 3), 1)
        self.envs = [env]
        pose = gymapi.Transform()
        rt = upstream.DOOR_RUNTIME
        pose.p = gymapi.Vec3(rt['robot_x'], rt['robot_y_by_spec'][0], rt['robot_z'])
        pose.r = gymapi.Quat.from_euler_zyx(0, 0, rt['robot_yaw'])
        self.base_actor = self.gym.create_actor(env, base_asset, pose, 'b1_base_visual', 0, 0)
        self.arm_actor = self.gym.create_actor(env, arm_asset, pose, loader.ARM_ACTOR_NAME, 0, 0)
        self.gym.set_actor_dof_properties(env, self.arm_actor, props)
        self.door_handles, self.door_actor_spec_ids = [], []
        self._create_door_actor(env, 0)
        self.wrist_camera_handles = [self._create_wrist_camera(env, self.arm_actor, 0)]
        self.front_camera_handles = [self._create_front_camera(env, self.base_actor, 0)]
        self.video_camera = None
        self.writer = None
        if self.args.float_video:
            import imageio.v2 as imageio
            cp = gymapi.CameraProperties()
            cp.width, cp.height = 960, 640
            self.video_camera = self.gym.create_camera_sensor(env, cp)
            self.gym.set_camera_location(self.video_camera, env, gymapi.Vec3(4.5, -3.5, 2.5), gymapi.Vec3(2.4, 0, .8))
            self.writer = imageio.get_writer(str(self.output/'rollout.mp4'), fps=25, codec='libx264', quality=7,
                                            ffmpeg_params=['-threads','2'])
        self.gym.prepare_sim(self.sim)
        self._roots = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states, self.door_root_state = self._roots[0:1], self._roots[2:3]
        self.initial_roots = self._roots.clone()
        self._full_dof_state_flat = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.dof_pos = self._full_dof_state_flat[:7, 0].view(1, 7)
        self.dof_vel = self._full_dof_state_flat[:7, 1].view(1, 7)
        self._door_dof_pos = self._full_dof_state_flat[7:, 0].view(1, -1)
        self._door_dof_vel = self._full_dof_state_flat[7:, 1].view(1, -1)
        self._rigid_body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).unsqueeze(0)
        self.body_index_names = {-1:'ground'}
        for actor, prefix in [(self.base_actor,'base'),(self.arm_actor,'arm'),(self.door_handles[0],'door')]:
            for name in self.gym.get_actor_rigid_body_names(env,actor):
                index = self.gym.find_actor_rigid_body_index(env,actor,name,gymapi.DOMAIN_ENV)
                self.body_index_names[index] = prefix+':'+name
        self.jacobian = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, loader.ARM_ACTOR_NAME))
        ee_asset_idx = self.body_names_to_idx[loader.EE_GRIPPER_LINK]
        ee_env_idx = self.gym.find_actor_rigid_body_index(env, self.arm_actor, loader.EE_GRIPPER_LINK, gymapi.DOMAIN_ENV)
        self.ee_pos = self._rigid_body_state[:, ee_env_idx, :3]
        self.ee_orn = self._rigid_body_state[:, ee_env_idx, 3:7]
        self.ee_j_eef = self.jacobian[:, ee_asset_idx-1, :6, :6]
        self.contact_forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        self.base_pos, self.base_quat = self.root_states[:, :3], self.root_states[:, 3:7]
        self.base_lin_vel, self.base_ang_vel = torch.zeros(1, 3, device=self.device), torch.zeros(1, 3, device=self.device)
        self._refresh()
        self._init_door_tensors()
        self._init_wrist_camera_tensors()
        self.curr_ee_goal_cart_world = self.ee_pos.clone()
        self.ee_goal_orn_quat = self.ee_orn.clone()
        self.ee_goal_orn_delta_rpy = torch.zeros(1, 3, device=self.device)
        self.freeze_arm_default = torch.ones(1, dtype=torch.bool, device=self.device)
        self.freeze_arm_zero = torch.zeros_like(self.freeze_arm_default)
        self.external_gripper_target = torch.full((1, 1), self.args.gripper_open, device=self.device)
        self.external_pos_gain, self.external_orn_gain = self.args.external_pos_gain, self.args.external_orn_gain
        self.arm_base_offset = torch.tensor(FK(Path(load_args.asset_root)/load_args.asset_file, 'link00', []).position([]), device=self.device, dtype=torch.float32).view(1, 3)
        self.dp_state_names = ['base_roll', 'base_pitch', 'base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z']
        self.dp_state_names += ['dof_pos_'+n for n in self.dof_names] + ['dof_vel_'+n for n in self.dof_names]
        self.dp_state_names += ['ee_base_x', 'ee_base_y', 'ee_base_z', 'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw', 'gripper_pos']
        self.raw = RawDoorDPRecorder(self.output/'raw', 50, self.dp_state_names, 'push lever door open')
        self.rows, self.transitions = [], []
        self.previous_quat = None
        self.snap_events = []
        self.was_snapped = False
        self.replay = None
        if self.args.float_replay:
            with np.load(self.args.float_replay) as replay:
                self.replay = replay['action'].copy()
                self.replay_quaternions = replay['executed_goal_quaternion'].copy() if 'executed_goal_quaternion' in replay else None
            self.replay_rows = [json.loads(s) for s in Path(self.args.float_replay).with_name('execution.jsonl').read_text().splitlines()]
        self.started = time.monotonic()
        self.log = (self.output/'execution.jsonl').open('x')
        self.config = dict(schema=SCHEMA, execution_profile=self.execution_profile, seed=self.args.float_seed,
                           args=vars(self.args), env=class_to_dict(cfg), dt=self.dt,
                           door_runtime=upstream.DOOR_RUNTIME,
                           physics_dt=self.physics_dt, dof_names=self.dof_names,
                           tensor_pipeline='cpu' if not sim_params.use_gpu_pipeline else 'gpu',
                           physx_use_gpu=bool(sim_params.physx.use_gpu),
                           arm_base_offset=self.arm_base_offset.cpu().tolist(),
                           default_dof_pos=self.default_dof_pos.cpu().tolist(),
                           joint_lower=self.lower.cpu().tolist(), joint_upper=self.upper.cpu().tolist(),
                           arm_body_names=self.body_names,
                           source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [Path(__file__).resolve(), Path(upstream.__file__).resolve(), Path(loader.__file__).resolve(),
                                         ROOT/'high-level/dp/door_dp_common.py',
                                         ROOT/'high-level/dp/models/door_diffusion_policy.py',
                                         ROOT/'low-level/legged_gym/envs/manip_loco/manip_loco.py',
                                         Path(float_source.__file__).resolve(),
                                         Path(load_args.asset_root)/load_args.asset_file]},
                           contact_pairs_requested=self.args.float_execution == 'source',
                           body_index_names=self.body_index_names)
        if self.args.float_push_target == 'source_float':
            self.config['source_float_push_parameters'] = dict(handle_follow_push_ratio=0.45,
                push_contact_bias=0.025, door_freeze_blend_start_ratio=0.82, door_freeze_target_ratio=0.94)
        for relative in self.config['source_hashes']:
            destination = self.output/'source'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, destination)
        (self.output/'config.json').write_text(json.dumps(self.config, indent=2,
            default=lambda value: value.tolist() if hasattr(value, 'tolist') else str(value)))

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def reset(self):
        self._roots[:] = self.initial_roots
        self._full_dof_state_flat.zero_()
        self.dof_pos[:] = self.default_dof_pos
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._roots))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._full_dof_state_flat))
        self._refresh()
        return self.get_observations(), None

    def get_observations(self):
        return torch.zeros(1, 1, device=self.device)

    def _get_ee_goal_spherical_center(self):
        return self.base_pos + quat_apply(self.base_quat, self.arm_base_offset)

    def dp_state(self):
        roll, pitch, _ = euler_from_quat(self.base_quat)
        pos = quat_rotate_inverse(self.base_quat, self.ee_pos-self._get_ee_goal_spherical_center())
        quat = self.ee_orn[0].detach().cpu().numpy().copy()
        quat /= np.linalg.norm(quat)
        if self.previous_quat is not None and np.dot(quat, self.previous_quat) < 0:
            quat = -quat
        self.previous_quat = quat.copy()
        values = [torch.stack([roll, pitch], -1), self.base_ang_vel, self.dof_pos,
                  self.dof_vel, pos, torch.tensor(quat, device=self.device).view(1, 4), self.dof_pos[:, -1:]]
        return torch.cat(values, -1)[0].detach().cpu().numpy().copy()

    def before_step(self, step, phase, names, opened, done, snapped, target, dp_action,
                    inference_sec=None, replanned=False):
        if self.replay is not None:
            if step >= len(self.replay):raise ValueError('Replay exhausted')
            apply_door_dp_action(self, self.replay[step])
            if self.replay_quaternions is not None:
                self.ee_goal_orn_quat[0] = torch.as_tensor(self.replay_quaternions[step],device=self.device)
            # Expert replay includes original task freeze flags; policy inference
            # still uses the source action applicator without restoring flags.
            self.freeze_arm_default[0] = self.replay_rows[step]['freeze_arm_default']
            self.freeze_arm_zero[0] = self.replay_rows[step]['freeze_arm_zero']
        self._refresh()  # Includes original snap_arm_to_zero before this callback.
        camera = self.capture_wrist_camera_images()
        images = dp_image_inputs_from_camera_tensors(camera)
        image_valid = all(im is not None for im in images)
        # Preserve original policy fallback to script commands on missing wrist
        # tensors. Mark placeholders explicitly so they cannot enter training.
        images = [np.zeros((54,96,3),np.uint8) if im is None else im for im in images]
        state = self.dp_state()
        snap_event = bool(snapped[0]) and not self.was_snapped
        self.snap_events.append('upstream_snap_arm_to_zero' if snap_event else 'none')
        self.was_snapped = bool(snapped[0])
        action = get_door_dp_action(self)
        self.raw.add_frame(state, images[0], images[1], action, int(phase[0]), images[2], images[3])
        q = self.dof_pos[0].cpu().numpy().copy()
        self.pending = dict(state=state, q=q, action=action)
        extra = dict(pass_started=bool(opened[0]), pass_done=bool(done[0]),
                     pass_arm_zero_snapped=bool(snapped[0]), through_target=target[0].cpu().tolist(),
                     dp_action_applied=dp_action is not None, image_valid=image_valid,
                     command_source='replay' if self.replay is not None else ('dp' if dp_action is not None else 'script'),
                     inference_sec=inference_sec, dp_replanned=replanned,
                     contact_pairs_available=False)
        self.task_done = bool(done[0])
        self.record = make_door_dp_log_record(self, step, action if dp_action is None else dp_action,
                                            phase_id=phase, phase_names=names, extra=extra)
        self.record['sim_time'] = step*self.dt
        self.record['q_before'] = q.tolist()
        self.record['executed_goal_quaternion'] = self.ee_goal_orn_quat[0].cpu().tolist()
        self.record['root_before'] = self.root_states[0].cpu().tolist()
        self.record['pre_observation_event'] = self.snap_events[-1]
        self.record['freeze_arm_default'] = bool(self.freeze_arm_default[0])
        self.record['freeze_arm_zero'] = bool(self.freeze_arm_zero[0])
        self.record['dq_before'] = self.dof_vel[0].cpu().tolist()
        self.record['jacobian_world'] = self.ee_j_eef[0].cpu().tolist()

    def adapt_expert_push(self, phase, manip_step, rotate, push, pass_started):
        if self.args.float_push_target != 'source_float' or int(phase[0]) != 7 or bool(pass_started[0]):
            return
        # Only position target generation changes, using the shared source helper.
        # Full-body orientation, gripper, IK, command order and pass processing stay upstream.
        handle = self._rigid_body_state[0, self.handle_body_idx]
        quat = handle[3:7].cpu().numpy().copy()
        goal = (handle[:3]+quat_apply(handle[3:7].view(1,4),self.goal_pos_offset_tensor)[0]).cpu().numpy().copy()
        if not hasattr(self, 'source_push_traj'):
            direction = float_source.normalize((push[0]-rotate[0]).cpu().numpy())
            self.source_push_traj = dict(rotate=rotate[0].cpu().numpy().copy(), push=push[0].cpu().numpy().copy(),
                                         push_dir=direction, approach_dir=-direction)
            self.source_push_args = SimpleNamespace(**dict(vars(self.args),
                handle_follow_push_ratio=0.45, push_contact_bias=0.025,
                door_freeze_blend_start_ratio=0.82, door_freeze_target_ratio=0.94,
                push_follow_orientation=False, ik_position_only=False))
        rotate_end = round(self.args.initial_hold_seconds/self.dt)+self.args.grasp_steps+self.args.grasp_hold_steps+self.args.gripper_close_steps+self.args.handle_rotate_steps
        t = float_source.smoothstep(float(manip_step[0]-rotate_end)/max(1,self.args.door_push_steps))
        angle = float(self._door_dof_pos[0,0])
        ratio = abs(angle)/(math.pi/2)
        position, frozen = float_source.source_float_push_target(
            self.source_push_args, self.source_push_traj, bool(self.open_door_stage[0]), ratio,
            goal, quat, self.ee_pos[0].cpu().numpy().copy(), t)
        self.curr_ee_goal_cart_world[0] = torch.as_tensor(position, device=self.device)

    def step(self, actions):
        base_before = self.root_states[:, :3].clone()
        yaw_before = euler_from_quat(self.base_quat)[2].clone()
        # Identical whole-body Cartesian DLS and original freeze / position servo.
        dpos = (self.curr_ee_goal_cart_world-self.ee_pos)*self.external_pos_gain
        quat = self.ee_orn/torch.norm(self.ee_orn, dim=-1, keepdim=True)
        drot = orientation_error(self.ee_goal_orn_quat, quat)*self.external_orn_gain
        dq = ManipLoco._control_ik(self, torch.cat([dpos, drot], -1).unsqueeze(-1))
        arm = self.dof_pos[:, :6]+dq
        arm[self.freeze_arm_default] = self.default_dof_pos[:6]
        arm[self.freeze_arm_zero] = 0
        targets = torch.cat([arm, self.external_gripper_target], -1)
        targets = torch.maximum(torch.minimum(targets, self.upper), self.lower)
        if not torch.isfinite(targets).all():
            raise ValueError('Nonfinite joint target')
        self.servo_target = targets[0].detach().cpu().numpy().copy()
        self.ik_delta = dq[0].detach().cpu().numpy().copy()
        all_targets = torch.zeros(7+self.num_door_dofs, device=self.device)
        all_targets[:7] = targets[0]
        for _ in range(self.cfg.control.decimation):
            yaw = euler_from_quat(self.base_quat)[2]
            vx, rate = self.commands[:, 0], self.commands[:, 2]
            new_yaw = yaw+rate*self.physics_dt
            if self.args.float_execution == 'source':
                xy = self.root_states[0,:2].numpy().copy()
                xy += np.array([float(vx*torch.cos(yaw)),float(vx*torch.sin(yaw))])*self.physics_dt
                float_source.set_robot_base_pose(self.gym,self.envs[0],
                    [self.base_actor,self.arm_actor],xy,self.args.robot_z,float(new_yaw))
            else:
                self._roots[:2, 0] += vx*torch.cos(yaw)*self.physics_dt
                self._roots[:2, 1] += vx*torch.sin(yaw)*self.physics_dt
                q = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), new_yaw)
                self._roots[:2, 3:7] = q
                self._roots[:2, 7:13] = 0
                self._roots[:2, 7] = vx*torch.cos(new_yaw)
                self._roots[:2, 8] = vx*torch.sin(new_yaw)
                self._roots[:2, 12] = rate
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._roots))
            all_targets[7:] = self._door_dof_pos[0]
            forces = torch.zeros_like(all_targets)
            forces[7:] = self._door_torques()[0]
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(all_targets))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(forces))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.physical_steps += 1
            self._refresh()
        # PhysX zeros fixed-root velocities. Derive achieved kinematic velocity
        # from measured root displacement, rather than report a fictitious zero.
        velocity = (self.root_states[:, :3]-base_before)/self.dt
        yaw_after = euler_from_quat(self.base_quat)[2]
        omega = torch.zeros_like(velocity)
        omega[:, 2] = torch.remainder(yaw_after-yaw_before+math.pi, 2*math.pi)-math.pi
        omega /= self.dt
        self.root_states[:, 7:10] = velocity
        self.root_states[:, 10:13] = omega
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, velocity)
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, omega)
        self.episode_length_buf += 1
        if not torch.isfinite(self._full_dof_state_flat).all():
            raise ValueError('Nonfinite simulation state')
        return self.get_observations(), None, None, None, self.time_out_buf, {}

    def after_step(self):
        next_state = self.dp_state()
        self.transitions.append(next_state)
        self.record.update(next_state=next_state.tolist(), q_after=self.dof_pos[0].cpu().tolist(),
                           completed_physical_steps=self.physical_steps,
                           dq_after=self.dof_vel[0].cpu().tolist(), ik_delta=self.ik_delta.tolist(),
                           servo_target=self.servo_target.tolist(),
                           door_after=self._door_dof_pos[0].cpu().tolist(),
                           base_after=self.root_states[0].cpu().tolist(),
                           contact_force=self.contact_forces.cpu().tolist())
        self.record['contacts'] = []
        if self.args.float_execution == 'source':
            try:
                contacts = self.gym.get_env_rigid_contacts(self.envs[0])
                for contact in contacts:
                    a,b = int(contact['body0']),int(contact['body1'])
                    if a not in self.body_index_names or b not in self.body_index_names:
                        raise ValueError('Contact body index outside documented DOMAIN_ENV mapping')
                    self.record['contacts'].append(dict(body0=a,body1=b,
                        body0_name=self.body_index_names[a],body1_name=self.body_index_names[b],
                        force_magnitude=float(contact['lambda'])))
                self.record['extra']['contact_pairs_available'] = True
            except Exception as exc:
                self.record['extra']['contact_pairs_available'] = False
                self.record['extra']['contact_pair_error'] = repr(exc)
        hand={'arm:gripperStator','arm:gripperMover'}
        handle='door:'+self.handle_body_name
        self.record['finger_handle_contact'] = any(c['force_magnitude']>0 and
            ((c['body0_name'] in hand and c['body1_name']==handle) or
             (c['body1_name'] in hand and c['body0_name']==handle)) for c in self.record['contacts'])
        self.log.write(json.dumps(self.record, allow_nan=False)+'\n')
        self.log.flush()
        self.rows.append(self.record)
        if self.writer is not None and len(self.rows) % 2 == 0:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            rgba = self.gym.get_camera_image(self.sim, self.envs[0], self.video_camera, gymapi.IMAGE_COLOR)
            self.writer.append_data(np.asarray(rgba).reshape(640, 960, 4)[..., :3])

    def close(self, error=None):
        if self.closed:
            return
        self.closed = True
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
        if hasattr(self, 'log'):
            self.log.close()
        if not getattr(self, 'rows', None):
            if hasattr(self, 'output'):
                (self.output/'failure.json').write_text(json.dumps(dict(error=error, frames=0), indent=2))
            return
        opened = any(r['extra']['pass_started'] for r in self.rows)
        done = any(r['extra']['pass_done'] for r in self.rows)
        if opened and error is None:
            self.raw.save_episode()
        n = len(self.rows)
        payload = {k:np.stack(v[:n]) for k,v in self.raw.frames.items()}
        payload.update(next_state=np.stack(self.transitions), frame=np.arange(n), next_frame=np.arange(1, n+1),
                       sim_time=np.arange(n)*self.dt, state_feature_names=np.asarray(self.dp_state_names),
                       action_names=np.asarray(ACTION_NAMES), schema=np.asarray(SCHEMA),
                       execution_profile=np.asarray(self.execution_profile), fps=np.asarray(50))
        payload['pre_observation_event'] = np.asarray(self.snap_events[:n])
        payload['image_valid'] = np.array([r['extra']['image_valid'] for r in self.rows],dtype=bool)
        payload['executed_goal_quaternion'] = np.array([r['executed_goal_quaternion'] for r in self.rows],dtype=np.float32)
        payload['finger_handle_contact'] = np.array([r.get('finger_handle_contact',False) for r in self.rows],dtype=bool)
        payload['contact_pairs_available'] = np.array([r['extra']['contact_pairs_available'] for r in self.rows],dtype=bool)
        np.savez_compressed(self.output/'trajectory.npz', **payload)
        last = self.rows[-1]
        distance = float(np.linalg.norm(np.array(last['base_after'][:2])-last['extra']['through_target'])) if opened else None
        summary = dict(seed=self.args.float_seed, schema=SCHEMA, execution_profile=self.execution_profile,
                       pass_navigation=self.args.float_pass_mode,
                       mode='replay' if self.replay is not None else ('policy' if self.args.dp_policy_checkpoint else 'expert'),
                       checkpoint=self.args.dp_policy_checkpoint,
                       opened=opened, pass_done=done, success=opened and done and error is None,
                       frames=n, max_signed_door_deg=max(-math.degrees(r['door_after'][0]) for r in self.rows),
                       completed_physical_steps=self.physical_steps,
                       final_signed_door_deg=-math.degrees(last['door_after'][0]),
                       through_distance=distance, elapsed_sec=time.monotonic()-self.started,
                       dp_frames=sum(r['extra']['dp_action_applied'] for r in self.rows),
                       script_frames=sum(r['extra']['command_source']=='script' for r in self.rows),
                       replay_frames=sum(r['extra']['command_source']=='replay' for r in self.rows),
                       invalid_image_frames=sum(not r['extra']['image_valid'] for r in self.rows),
                       contact_pairs_available=all(r['extra']['contact_pairs_available'] for r in self.rows),
                       finger_handle_contact_frames=sum(r.get('finger_handle_contact',False) for r in self.rows),
                       first_open_frame=next((r['step'] for r in self.rows if r['extra']['pass_started']), None),
                       first_pass_frame=next((r['step'] for r in self.rows if r['extra']['pass_done']), None),
                       end_reason='exception' if error else ('pass_done' if done else 'max_steps'), error=error)
        (self.output/'summary.json').write_text(json.dumps(summary, indent=2))
        print('FLOAT_SUMMARY', json.dumps(summary), flush=True)
        self.gym.destroy_sim(self.sim)
        self.asset_temp.cleanup()


if __name__ == '__main__':
    try:
        upstream.main(backend=WholebodyFloat)
    except BaseException as exc:
        if WholebodyFloat.instance is not None:
            WholebodyFloat.instance.close(error=repr(exc))
        raise
