import argparse
import os
import pickle
import shutil

import torch
import wandb

from reward_wrapper import Backflip
from rsl_rl.runners import OnPolicyRunner

# Compatibility shim for igl.signed_distance (returns >3 values on some builds)
try:
    import igl  # type: ignore
    _orig_signed_distance = igl.signed_distance
    def _signed_distance_compat(P, V, F, *args, **kwargs):
        out = _orig_signed_distance(P, V, F, *args, **kwargs)
        if isinstance(out, tuple) and len(out) > 3:
            return out[0], out[1], out[2]
        return out
    igl.signed_distance = _signed_distance_compat  # type: ignore
except Exception:
    pass

import genesis as gs


def get_train_cfg(args):
    return {
        'algorithm': {
            'clip_param': 0.2,
            'desired_kl': 0.01,
            'entropy_coef': 0.01,
            'gamma': 0.99,
            'lam': 0.95,
            'learning_rate': 0.001,
            'max_grad_norm': 1.0,
            'num_learning_epochs': 5,
            'num_mini_batches': 4,
            'schedule': 'adaptive',
            'use_clipped_value_loss': True,
            'value_loss_coef': 1.0,
        },
        'init_member_classes': {},
        'policy': {
            'activation': 'elu',
            'actor_hidden_dims': [512, 256, 128],
            'critic_hidden_dims': [512, 256, 128],
            'init_noise_std': 1.0,
        },
        'runner': {
            'algorithm_class_name': 'PPO',
            'checkpoint': -1,
            'experiment_name': args.exp_name,
            'load_run': -1,
            'log_interval': 1,
            'max_iterations': args.max_iterations,
            'num_steps_per_env': 24,
            'policy_class_name': 'ActorCritic',
            'record_interval': 50,
            'resume': False,
            'resume_path': None,
            'run_name': '',
            'runner_class_name': 'runner_class_name',
            'save_interval': 100,
        },
        'runner_class_name': 'OnPolicyRunner',
        'seed': 1,
    }


def get_cfgs():
    # Control all primary joints available in MJCF actuators (legs + waist + arms)
    dof_names = [
        # left leg (6)
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        # right leg (6)
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        # waist (3)
        'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
        # left arm (7)
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
        'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
        # right arm (7)
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
        'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
    ]

    env_cfg = {
        'mjcf_xml': 'robots/g1/g1_23dof.xml',
        'num_actions': len(dof_names),
        'num_dofs': len(dof_names),
        'default_joint_angles': {
            # legs (参考 bak/g1_config.py)
            'left_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': -0.1,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.1,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
            # waist
            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,
            # left arm
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.0,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            # right arm
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.0,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
        },
        'dof_names': dof_names,
        'termination_contact_link_names': ['pelvis'],
        'penalized_contact_link_names': ['pelvis', 'torso_link'],
        'feet_link_names': ['left_ankle_roll_link', 'right_ankle_roll_link'],
        'base_link_name': ['pelvis'],
        # PD (参考 bak/g1_config.py 的分组；保留 joint 作为兜底)
        'PD_stiffness': {
            'hip_yaw': 150.0,
            'hip_roll': 150.0,
            'hip_pitch': 150.0,
            'knee': 300.0,
            'ankle': 40.0,
            'joint': 50.0,
        },
        'PD_damping': {
            'hip_yaw': 2.0,
            'hip_roll': 2.0,
            'hip_pitch': 2.0,
            'knee': 4.0,
            'ankle': 2.0,
            'joint': 2.0,
        },
        'use_implicit_controller': False,
        # termination
        'termination_if_roll_greater_than': 0.7,
        'termination_if_pitch_greater_than': 0.7,
        'termination_if_height_lower_than': 0.35,
        # base pose
        'base_init_pos': [0.0, 0.0, 0.9],
        'base_init_quat': [1.0, 0.0, 0.0, 0.0],
        # ensure feet not under ground at reset
        'reset_base_height': 0.9,
        # time
        'episode_length_s': 2.0,
        'resampling_time_s': 4.0,
        'command_type': 'ang_vel_yaw',
        'action_scale': 0.5,
        'action_latency': 0.02,
        'clip_actions': 100.0,
        'send_timeouts': True,
        # random push (disabled by default)
        'push_interval_s': -1,
        'max_push_vel_xy': 1.0,
        'control_freq': 50,
        'decimation': 4,
        'feet_geom_offset': 1,
        'use_terrain': False,
        # domain randomization (conservative)
        'randomize_friction': True,
        'friction_range': [0.6, 1.2],
        'randomize_base_mass': True,
        'added_mass_range': [-1., 2.],
        'randomize_com_displacement': True,
        'com_displacement_range': [-0.01, 0.01],
        'randomize_motor_strength': False,
        'motor_strength_range': [0.9, 1.1],
        'randomize_motor_offset': True,
        'motor_offset_range': [-0.02, 0.02],
        'randomize_kp_scale': True,
        'kp_scale_range': [0.8, 1.2],
        'randomize_kd_scale': True,
        'kd_scale_range': [0.8, 1.2],
        'coupling': False,
    }

    # num_dof = 15 → obs dims
    num_dof = len(dof_names)
    obs_cfg = {
        'num_obs': 6 + 4 * num_dof + 6,
        'num_history_obs': 1,
        'obs_noise': {
            'ang_vel': 0.1,
            'gravity': 0.02,
            'dof_pos': 0.01,
            'dof_vel': 0.5,
        },
        'obs_scales': {
            'lin_vel': 2.0,
            'ang_vel': 0.25,
            'dof_pos': 1.0,
            'dof_vel': 0.05,
        },
        'num_priv_obs': 10 + 4 * num_dof + 6,
    }

    reward_cfg = {
        'soft_dof_pos_limit': 0.9,
        'reward_scales': {
            'ang_vel_y': 5.0,
            'ang_vel_z': -1.0,
            'lin_vel_z': 20.0,
            'orientation_control': -1.0,
            'feet_height_before_backflip': -25.0,
            'height_control': -8.0,
            'actions_symmetry': -0.05,
            'gravity_y': -8.0,
            'feet_distance': -1.0,
            'action_rate': -0.001,
        },
    }

    command_cfg = {
        'num_commands': 4,
        'lin_vel_x_range': [0.0, 0.0],
        'lin_vel_y_range': [0.0, 0.0],
        'ang_vel_range': [0.0, 0.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', type=str, default='g1_backflip')
    parser.add_argument('-v', '--vis', action='store_true', default=False)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-B', '--num_envs', type=int, default=50)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('-o', '--offline', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--ckpt', type=int, default=1000)
    args = parser.parse_args()

    if args.debug:
        args.vis = True
        args.offline = True
        args.num_envs = 1

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level='warning')

    log_dir = f'logs/{args.exp_name}'
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = Backflip(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
        eval=args.eval,
        debug=args.debug,
    )

    runner = OnPolicyRunner(env, get_train_cfg(args), log_dir, device='cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.resume is not None:
        resume_dir = f'logs/{args.resume}'
        resume_path = os.path.join(resume_dir, f'model_{args.ckpt}.pt')
        print('==> resume training from', resume_path)
        runner.load(resume_path)

    wandb.init(project='genesis', name=args.exp_name, dir=log_dir, mode='offline' if args.offline else 'online')

    pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg], open(f'{log_dir}/cfgs.pkl', 'wb'))

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    main()


