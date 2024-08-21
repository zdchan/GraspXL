#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import shadow_test as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
from raisimGymTorch.env.bin.shadow_test import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose_faive_label

import os
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import argparse
from raisimGymTorch.helper import rotations
import joblib
import torch


exp_name = "shadow_floating"

weight_saved = 'full_19000_r.pt'

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg.yaml')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default=exp_name)
parser.add_argument('-w', '--weight', type=str, default=weight_saved)
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-seed', '--seed', type=int, default=1)
parser.add_argument('-itr', '--num_batches', type=int, default=0)


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

if args.seed != 1:
    cfg['seed'] = args.seed

cfg['environment']['visualize'] = True

cat_name = 'large_scale_obj'
cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path)

items = os.listdir(directory_path)

# # Filter out only the folders (directories) from the list of items
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

ob_dim_r = 258
act_dim = 28
print('ob dim', ob_dim_r)
print('act dim', act_dim)

# Training
trail_steps = 30
reward_clip = -2.0
grasp_steps = 100
lift_step = 50
n_steps_r = grasp_steps + trail_steps + lift_step

test_dir = True

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/shapenet_eval.py"], test_dir=test_dir)

activations = nn.LeakyReLU


obj_path_list = []
obj_ori_list = folder_names


num_env_per_iter = 500
iter_num = args.num_batches

obj_list = []
if iter_num < len(obj_ori_list) // num_env_per_iter:
    num_envs = num_env_per_iter
else:
    num_envs = len(obj_ori_list) % num_env_per_iter
for i in range(num_envs):
    obj_list.append(obj_ori_list[i + iter_num * num_env_per_iter])
print(obj_list)
print("iter_num", iter_num)

label_list = {}

for obj in obj_list:
    label_pth = os.path.join(directory_path, obj) + "_label.pkl"
    label = joblib.load(label_pth)
    label_list[obj] = label


cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)

# Environment definition
env = VecEnv(obj_list, mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

print("initialization finished")

for obj_item in obj_list:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}.urdf"))
env.load_multi_articulated(obj_path_list)

# RL network

actor_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)


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

suc_ori_error = 0.0
suc_direction_error = 0.0
suc_rotation_error = 0.0
suc_center_error = 0.0
lifted_num = 0

for update in range(5):
    print("update", update)
    start = time.time()

    qpos_reset_r = np.zeros((num_envs, 28), dtype='float32')
    qpos_reset_l = np.zeros((num_envs, 28), dtype='float32')
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

        qpos_reset_r[i, -4] = 1.7


        if np.linalg.norm(env.non_aff_mesh[i].centroid - fake_non_aff_center) < 0.01:
            non_aff_mesh = None
        else:
            non_aff_mesh = env.non_aff_mesh[i]
            contain_non_aff[i, 0] = 1.


        obj_idx = obj_list[i] + f"_{update}"
        current_label = label_list[obj_list[i]][obj_idx]

        rot, pos, bias = get_initial_pose_faive_label(env.aff_mesh[i], non_aff_mesh, 'shadow',
                                                      current_label['target_direction'], current_label['rand_offset'], current_label['target_center'])


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
                    np.zeros((num_envs, 28), 'float32'),
                    np.zeros((num_envs, 28), 'float32'),
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

    global_state = env.get_global_state()
    lifted = (global_state[:, 131] - obj_pose_reset[:, 2] > 0.1) * (np.linalg.norm(obs_r[:, :3], axis=1) < 0.1)
    print("current success rate", np.sum(lifted) / num_envs)
    current_center_error = np.linalg.norm(obs_r[:, :3], axis=1)
    print("current center error", np.sum(current_center_error) / num_envs)
    current_center_error_lifted = current_center_error[lifted]
    print("current center error lifted", np.sum(current_center_error_lifted) / (np.sum(lifted) + 0.0000001))
    current_direction_error = np.linalg.norm(global_state[:, 132:135], axis=1)
    print("current direction error", np.sum(current_direction_error) / num_envs)
    current_direction_error_lifted = current_direction_error[lifted]
    print("current direction error lifted", np.sum(current_direction_error_lifted) / (np.sum(lifted) + 0.0000001))
    current_angle_error = np.arccos(global_state[:, 135])
    print("current angle error", np.sum(current_angle_error) / num_envs)
    current_angle_error_lifted = current_angle_error[lifted]
    print("current angle error lifted", np.sum(current_angle_error_lifted) / (np.sum(lifted) + 0.0000001))
    print(" ")

    success_rate = (update * success_rate + np.sum(lifted) / num_envs) / (update + 1)
    center_error = (update * center_error + np.sum(current_center_error) / num_envs) / (update + 1)
    suc_center_error = (lifted_num * suc_center_error + np.sum(current_center_error_lifted)) / (lifted_num + np.sum(lifted) + 0.0000001)
    direction_error = (update * direction_error + np.sum(current_direction_error) / num_envs) / (update + 1)
    suc_direction_error = (lifted_num * suc_direction_error + np.sum(current_direction_error_lifted)) / (lifted_num + np.sum(lifted) + 0.0000001)
    rotation_error = (update * rotation_error + np.sum(current_angle_error) / num_envs) / (update + 1)
    suc_rotation_error = (lifted_num * suc_rotation_error + np.sum(current_angle_error_lifted)) / (lifted_num + np.sum(lifted) + 0.0000001)
    print("average success rate", success_rate)
    print("average center error", center_error)
    print("average center error lifted", suc_center_error)
    print("average direction error", direction_error)
    print("average direction error lifted", suc_direction_error)
    print("average rotation error", rotation_error)
    print("average rotation error lifted", suc_rotation_error)
    print(" ")
    lifted_num += np.sum(lifted)

file_name = f"/large_result_0.1_0.1_{iter_num}.txt"
direct_save = saver.data_dir + "/../../../eval_result/ours/"
if not os.path.exists(direct_save):
    os.system(f"mkdir -p {direct_save}")
with open(direct_save + file_name, 'w') as result_file:
    result_file.write("final success rate " + str(success_rate) + "\n")
    result_file.write("final center error " + str(center_error) + "\n")
    result_file.write("final center error lifted " + str(suc_center_error) + "\n")
    result_file.write("final direction error " + str(direction_error) + "\n")
    result_file.write("final direction error lifted " + str(suc_direction_error) + "\n")
    result_file.write("final rotation error " + str(rotation_error) + "\n")
    result_file.write("final rotation error lifted " + str(suc_rotation_error) + "\n")
print("done")



