#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import ours_test as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
from raisimGymTorch.env.bin.ours_test import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose_universal

# import pdb

import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np

import argparse
from raisimGymTorch.helper import rotations
import torch


data_version = "chiral_220223"

path_mean_r = os.path.join("./../rsc/mano_double/right_pose_mean.txt")
pose_mean_r = np.loadtxt(path_mean_r)

exp_name = "floating_mixed"


weight_saved = 'full_5600_r.pt'


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg.yaml')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default=exp_name)
parser.add_argument('-w', '--weight', type=str, default=weight_saved)
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-itr', '--num_iterations', type=int, default=1)
parser.add_argument('-group', '--group_name', type=int, default=0)
parser.add_argument('-mode', '--mode', type=int, default=-1)


args = parser.parse_args()
weight_path = args.weight
cfg_grasp = args.cfg

print(f"Configuration file: \"{args.cfg}\"")
print(f"Experiment name: \"{args.exp_name}\"")

# task specification
task_name = args.exp_name
# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

if args.logdir is None:
    exp_path = home_path
else:
    exp_path = args.logdir

# config
cfg = YAML().load(open(task_path + '/cfgs/' + args.cfg, 'r'))

cfg['environment']['visualize'] = True


mode = args.mode
if mode == -1:
    cat_name = f"urdf_group{args.group_name}"
elif mode == 0:
    cat_name = f"surdf_group{args.group_name}_s"
elif mode == 1:
    cat_name = f"surdf_group{args.group_name}_m"
elif mode == 2:
    cat_name = f"surdf_group{args.group_name}_l"

cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path)

items = os.listdir(directory_path)

# # Filter out only the folders (directories) from the list of items
if mode == -1:
    load_ids = np.load(f'/mnt/ssd/data3/hui/processed_ids_{args.group_name}.npy').tolist()
elif mode == 0:
    load_ids = np.load(f'/mnt/ssd/data3/hui/processed_ids_{args.group_name}_s.npy').tolist()
elif mode == 1:
    load_ids = np.load(f'/mnt/ssd/data3/hui/processed_ids_{args.group_name}_m.npy').tolist()
elif mode == 2:
    load_ids = np.load(f'/mnt/ssd/data3/hui/processed_ids_{args.group_name}_l.npy').tolist()
print("number of objects", len(load_ids))


obj_path_list = []
obj_ori_list = load_ids

num_env_per_iter = 500
iter_num = args.num_iterations
obj_list = []
if iter_num < len(obj_ori_list) // num_env_per_iter:
    num_envs = num_env_per_iter
else:
    num_envs = len(obj_ori_list) % num_env_per_iter
for i in range(num_envs):
    obj_list.append(obj_ori_list[i + iter_num * num_env_per_iter])
print("iter_num", iter_num)

num_envs = len(obj_list)

activations = nn.LeakyReLU

cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)

# Environment definition
env = VecEnv(obj_list, mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

print("initialization finished")

for obj_item in obj_list:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)

ob_dim_r = 304
act_dim = 51
print('ob dim', ob_dim_r)
print('act dim', act_dim)

# Training
trail_steps = 30
reward_clip = -2.0
grasp_steps = 100
lift_step = 50
n_steps_r = grasp_steps + trail_steps + lift_step
total_steps_r = n_steps_r * env.num_envs

# RL network

actor_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)


test_dir = True

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/objaverse_eval.py"], test_dir=test_dir)


ppo_r = PPO.PPO(actor=actor_r,
                critic=critic_r,
                num_envs=num_envs,
                num_transitions_per_env=n_steps_r,
                num_learning_epochs=4,
                gamma=0.996,
                lam=0.95,
                num_mini_batches=4,
                device=device,
                log_dir=saver.data_dir,
                shuffle_batch=False
                )
load_param(saver.data_dir.split('eval')[0]+weight_path, env, actor_r, critic_r, ppo_r.optimizer, saver.data_dir, cfg_grasp)

success_rate = 0.0
ori_error = 0.0
direction_error = 0.0
rotation_error = 0.0
center_error = 0.0
contact_ratio = 0.0
suc_ori_error = 0.0
suc_direction_error = 0.0
suc_rotation_error = 0.0
suc_center_error = 0.0
suc_contact_ratio = 0.0

lifted_num = 0

for update in range(1):
    print("update", update)
    start = time.time()

    qpos_reset_r = np.zeros((num_envs, 51), dtype='float32')
    qpos_reset_l = np.zeros((num_envs, 51), dtype='float32')
    obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

    target_center = np.zeros_like(env.affordance_center)
    object_center = np.zeros_like(env.affordance_center)
    fake_non_aff_center = [0.346408, 0.346408, 0.346408]
    contain_non_aff = np.zeros((num_envs, 1), dtype='float32')

    for i in range(num_envs):
        got_proper_initial_pose = False
        lowest_point = 0.
        txt_file_path = os.path.join(directory_path, obj_list[i]) + "/lowest_point_new.txt"
        with open(txt_file_path, 'r') as txt_file:
            lowest_point = float(txt_file.read())

        obj_state_path = os.path.join(directory_path, obj_list[i]) + "/obj_stable_state.txt"
        with open(obj_state_path, 'r') as obj_state_file:
            obj_state = obj_state_file.read()
        values_str = obj_state.strip('[]').split()
        vector = np.array([float(value) for value in values_str[:-1]])
        if vector[2] < 0.3:
            vector[:3] = [1., -0., 0.502]
        obj_pose_reset[i, :7] = vector[:7]
        obj_pose_reset[i, 2] += 0.002

        qpos_reset_r[i, 6:] = pose_mean_r.copy() / 5
        qpos_reset_r[i, -9:-6] *= 5
        qpos_reset_r[i, -8] += 0.4
        qpos_reset_r[i, -7] += 0.6

        if np.linalg.norm(env.non_aff_mesh[i].centroid - fake_non_aff_center) < 0.01:
            non_aff_mesh = None
        else:
            non_aff_mesh = env.non_aff_mesh[i]
            contain_non_aff[i, 0] = 1.

        rot, pos, bias = get_initial_pose_universal(env.aff_mesh[i], non_aff_mesh)

        obj_mat = rotations.quat2mat(obj_pose_reset[i, 3:7])
        wrist_pose_obj = rotations.axisangle2euler(rot.reshape(-1, 3)).reshape(1, -1)
        wrist_mat = rotations.euler2mat(wrist_pose_obj)
        wrist_in_world = np.matmul(obj_mat, wrist_mat)
        wrist_pose = rotations.mat2euler(wrist_in_world)
        qpos_reset_r[i, :3] = obj_pose_reset[i, :3] + np.matmul(obj_mat, pos[0, :])
        qpos_reset_r[i, 3:6] = wrist_pose[0, :]

        target_center[i, :] = bias[:]
        object_center[i, :] = env.affordance_center[i]
    print("complete initial pose generation")


    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((num_envs, 51), 'float32'),
                    np.zeros((num_envs, 51), 'float32'),
                    obj_pose_reset,
                    )

    env.set_goals(target_center,
                  object_center,
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  np.zeros((num_envs, 1), 'float32'),
                  )

    obs_new_r, _ = env.observe(contain_non_aff, partial_obs=False)
    for step in range(n_steps_r):
        obs_r = obs_new_r
        obs_r = obs_r[:, :].astype('float32')
        if step == 130:
            env.switch_root_guidance(True)

        action_r = actor_r.architecture.architecture(torch.from_numpy(obs_r.astype('float32')).to(device))
        action_r = action_r.cpu().detach().numpy()
        action_l = np.zeros_like(action_r)

        frame_start = time.time()

        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, dis_info = env.observe(contain_non_aff, partial_obs=False)
        obs_new_r = obs_new_r[:].astype('float32')

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end - frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    obs_r, _ = env.observe(contain_non_aff, partial_obs=False)
    obs_r = obs_r[:].astype('float32')

    global_state = env.get_global_state()
    lifted = (global_state[:, 131] - obj_pose_reset[:, 2] > 0.1) * (np.linalg.norm(obs_r[:, :3], axis=1) < 0.1)
    current_center_error = np.linalg.norm(obs_r[:, :3], axis=1)
    current_center_error_lifted = current_center_error[lifted]
    current_direction_error = np.linalg.norm(global_state[:, 132:135], axis=1)
    current_direction_error_lifted = current_direction_error[lifted]
    current_angle_error = np.arccos(global_state[:, 135])
    current_angle_error_lifted = current_angle_error[lifted]

    success_rate = (update * success_rate + np.sum(lifted) / num_envs) / (update + 1)
    center_error = (update * center_error + np.sum(current_center_error) / num_envs) / (update + 1)
    suc_center_error = (lifted_num * suc_center_error + np.sum(current_center_error_lifted)) / (lifted_num + np.sum(lifted) + 0.0000001)
    direction_error = (update * direction_error + np.sum(current_direction_error) / num_envs) / (update + 1)
    suc_direction_error = (lifted_num * suc_direction_error + np.sum(current_direction_error_lifted)) / (lifted_num + np.sum(lifted) + 0.0000001)
    rotation_error = (update * rotation_error + np.sum(current_angle_error) / num_envs) / (update + 1)
    suc_rotation_error = (lifted_num * suc_rotation_error + np.sum(current_angle_error_lifted)) / (lifted_num + np.sum(lifted) + 0.0000001)
    lifted_num += np.sum(lifted)

    print("iter num", iter_num)
    print("final success rate", success_rate)
    print("final center error", center_error)
    print("final center error lifted", suc_center_error)
    print("final direction error", direction_error)
    print("final direction error lifted", suc_direction_error)
    print("final rotation error", rotation_error)
    print("final rotation error lifted", suc_rotation_error)
    print(" ")



direct_save = "/mnt/ssd/data3/hui/results/"

try:
    if mode == -1:
        pre_result = np.load(direct_save + f"group{args.group_name}_results_{iter_num - 1}.npy", allow_pickle=True).item()
    elif mode == 0:
        pre_result = np.load(direct_save + f"group{args.group_name}_s_results_{iter_num - 1}.npy", allow_pickle=True).item()
    elif mode == 1:
        pre_result = np.load(direct_save + f"group{args.group_name}_m_results_{iter_num - 1}.npy", allow_pickle=True).item()
    elif mode == 2:
        pre_result = np.load(direct_save + f"group{args.group_name}_l_results_{iter_num - 1}.npy", allow_pickle=True).item()
    print("load previous result")
except:
    pre_result = {}
    pre_result["success_rate"] = 0
    pre_result["center_error"] = 0
    pre_result["suc_center_error"] = 0
    pre_result["direction_error"] = 0
    pre_result["suc_direction_error"] = 0
    pre_result["rotation_error"] = 0
    pre_result["suc_rotation_error"] = 0
    pre_result["obj_num"] = 0
    print("no previous result")
pre_success_rate = pre_result["success_rate"]
pre_center_error = pre_result["center_error"]
pre_suc_center_error = pre_result["suc_center_error"]
pre_direction_error = pre_result["direction_error"]
pre_suc_direction_error = pre_result["suc_direction_error"]
pre_rotation_error = pre_result["rotation_error"]
pre_suc_rotation_error = pre_result["suc_rotation_error"]
pre_obj_num = pre_result["obj_num"]

result = {}
result["current_success_rate"] = success_rate
result["current_center_error"] = center_error
result["current_suc_center_error"] = suc_center_error
result["current_direction_error"] = direction_error
result["current_suc_direction_error"] = suc_direction_error
result["current_rotation_error"] = rotation_error
result["current_suc_rotation_error"] = suc_rotation_error
result["current_obj_num"] = num_envs
result["success_rate"] = (success_rate * num_envs + pre_success_rate * pre_obj_num) / (num_envs + pre_obj_num)
result["center_error"] = (center_error * num_envs + pre_center_error * pre_obj_num) / (num_envs + pre_obj_num)
result["suc_center_error"] = (suc_center_error * lifted_num + pre_suc_center_error * pre_obj_num) / (lifted_num + pre_obj_num)
result["direction_error"] = (direction_error * num_envs + pre_direction_error * pre_obj_num) / (num_envs + pre_obj_num)
result["suc_direction_error"] = (suc_direction_error * lifted_num + pre_suc_direction_error * pre_obj_num) / (lifted_num + pre_obj_num)
result["rotation_error"] = (rotation_error * num_envs + pre_rotation_error * pre_obj_num) / (num_envs + pre_obj_num)
result["suc_rotation_error"] = (suc_rotation_error * lifted_num + pre_suc_rotation_error * pre_obj_num) / (lifted_num + pre_obj_num)
result["obj_num"] = num_envs + pre_obj_num
if mode == -1:
    np.save(direct_save + f"group{args.group_name}_results_{iter_num}.npy", result)
elif mode == 0:
    np.save(direct_save + f"group{args.group_name}_s_results_{iter_num}.npy", result)
elif mode == 1:
    np.save(direct_save + f"group{args.group_name}_m_results_{iter_num}.npy", result)
elif mode == 2:
    np.save(direct_save + f"group{args.group_name}_l_results_{iter_num}.npy", result)


print("iter num", iter_num)
print("group", args.group_name)
if mode == -1:
    print("mode", "original")
elif mode == 0:
    print("mode", "small")
elif mode == 1:
    print("mode", "medium")
elif mode == 2:
    print("mode", "large")
print("current averate success rate", result["success_rate"])
print("current averate center error", result["center_error"])
print("current averate center error lifted", result["suc_center_error"])
print("current averate direction error", result["direction_error"])
print("current averate direction error lifted", result["suc_direction_error"])
print("current averate rotation error", result["rotation_error"])
print("current averate rotation error lifted", result["suc_rotation_error"])
print("current obj num", result["obj_num"])


