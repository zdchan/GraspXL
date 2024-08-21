import numpy
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import ours_fixed as mano
from raisimGymTorch.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.ours_fixed import NormalSampler
from raisimGymTorch.helper.initial_pose_final import get_initial_pose

import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
from datetime import datetime
import argparse
from raisimGymTorch.helper import rotations
import joblib
import random
import wandb
import torch
# from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput

from manotorch.anatomy_loss import AnatomyConstraintLossEE
from raisimGymTorch.helper.mano_amano import PoseTrans

ManoTrans = PoseTrans()
anatomyLoss = AnatomyConstraintLossEE(reduction='none')
anatomyLoss.setup()

exp_name = "fixed_mixed"

weight_saved = '/../../floating_mixed/full_5600_r.pt'

path_mean_r = os.path.join("./../rsc/mano_double/right_pose_mean.txt")
pose_mean_r = np.loadtxt(path_mean_r)

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg_reg.yaml')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default=exp_name)
parser.add_argument('-w', '--weight', type=str, default=weight_saved)
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-seed', '--seed', type=int, default=1)
parser.add_argument('-itr', '--num_iterations', type=int, default=50001)
parser.add_argument('-re', '--load_trained_policy', action="store_true")
parser.add_argument('-ln', '--log_name', type=str, default='single_obj')


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

wandb.init(project=task_name, config=cfg, name=args.log_name)

if args.seed != 1:
    cfg['seed'] = args.seed

obj_path_list = []
obj_list = []

cat_name = 'mixed_train'
cfg['environment']['load_set'] = cat_name
directory_path = home_path + f"/rsc/{cat_name}/"
print(directory_path)
items = os.listdir(directory_path)

# # Filter out only the folders (directories) from the list of items
folder_names = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

obj_path_list = []
obj_ori_list = []
for item in folder_names:
    if item.startswith('Mug'):
        obj_ori_list.append(item)
        obj_ori_list.append(item)
        obj_ori_list.append(item)
    elif item.startswith('Earphone'):
        obj_ori_list.append(item)
        obj_ori_list.append(item)
        obj_ori_list.append(item)
    elif item.startswith('Knife'):
        obj_ori_list.append(item)
        obj_ori_list.append(item)
        obj_ori_list.append(item)
    elif item.startswith('Scissors'):
        obj_ori_list.append(item)
        obj_ori_list.append(item)
        obj_ori_list.append(item)
    elif item.startswith('WineGlass'):
        obj_ori_list.append(item)
        obj_ori_list.append(item)
        obj_ori_list.append(item)
    else:
        obj_ori_list.append(item)
    if item.endswith('handle'):
        obj_ori_list.append(item)
        obj_ori_list.append(item)
        obj_ori_list.append(item)

num_envs = len(obj_ori_list)
activations = nn.LeakyReLU

cfg['environment']['num_envs'] = num_envs
print('num envs', num_envs)

# for i in range(3):
for item in obj_ori_list:
    obj_list.append(item)
# Environment definition
env = VecEnv(obj_list, mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'], cat_name=cat_name)

for obj_item in obj_list:
    obj_path_list.append(os.path.join(f"{obj_item}/{obj_item}_fixed_base.urdf"))
env.load_multi_articulated(obj_path_list)

ob_dim_r = 304
act_dim = 51
print('ob dim', ob_dim_r)
print('act dim', act_dim)

# Training
trail_steps = 30
reward_clip = -2.0
grasp_steps = 100
n_steps_r = grasp_steps + trail_steps
total_steps_r = n_steps_r * env.num_envs

print(env.num_envs)

# RL network
actor_r = ppo_module.Actor(
    ppo_module.MLP(cfg['architecture']['policy_net'], activations, ob_dim_r, act_dim),
    ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)), device)

critic_r = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], activations, ob_dim_r, 1), device)

test_dir = False

saver = ConfigurationSaver(log_dir=exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp",
                                       task_path + "/runner.py", task_path + "/../../RaisimGymVecEnvOther.py"], test_dir=test_dir)


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
                # learning_rate=1e-4
                )

if args.load_trained_policy:
    load_param(saver.data_dir.split('eval')[0] + weight_path, env, actor_r, critic_r, ppo_r.optimizer, saver.data_dir,
               cfg_grasp)


finger_weights = np.ones((num_envs, 21)).astype('float32')
for i in range(5):
    finger_weights[:, 4 * i+4] *= 4.0
finger_weights /= finger_weights.sum(axis=1).reshape(-1, 1)
finger_weights *= 21.0
affordance_reward_r = np.zeros((num_envs, 1))
not_affordance_reward_r = np.zeros((num_envs, 1))

qpos_reset_r = np.zeros((num_envs, 51), dtype='float32')
qpos_reset_l = np.zeros((num_envs, 51), dtype='float32')
obj_pose_reset = np.zeros((num_envs, 8), dtype='float32')

zero_shape = torch.from_numpy(np.zeros((num_envs, 10)).astype('float32')).to('cuda')


for update in range(args.num_iterations):
    start = time.time()

    ### Evaluate trained model visually (note always the first environment gets visualized)

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor_r.architecture.state_dict(),
            'actor_distribution_state_dict': actor_r.distribution.state_dict(),
            'critic_architecture_state_dict': critic_r.architecture.state_dict(),
            'optimizer_state_dict': ppo_r.optimizer.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '_r.pt')

        env.save_scaling(saver.data_dir, str(update))

    target_center = np.zeros_like(env.affordance_center)
    object_center = np.zeros_like(env.affordance_center)
    fake_non_aff_center = [0.346408, 0.346408, 0.346408]
    contain_non_aff = np.zeros((num_envs, 1), dtype='float32')

    for i in range(num_envs):
        got_proper_initial_pose = False
        obj_pose_reset[i, :] = [1., -0., 0.55, 1., -0., -0., 0., 0.]

        qpos_reset_r[i, 6:] = pose_mean_r.copy() / 5
        qpos_reset_r[i, -9:-6] *= 5
        qpos_reset_r[i, -8] += 0.4
        qpos_reset_r[i, -7] += 0.6

        if np.linalg.norm(env.non_aff_mesh[i].centroid - fake_non_aff_center) < 0.01:
            non_aff_mesh = None
        else:
            non_aff_mesh = env.non_aff_mesh[i]
            contain_non_aff[i, 0] = 1.

        rot, pos, bias = get_initial_pose(env.aff_mesh[i], non_aff_mesh)

        wrist_pose = rotations.axisangle2euler(rot.reshape(-1, 3)).reshape(1, -1)
        qpos_reset_r[i, :3] = obj_pose_reset[i, :3] + pos[0, :]
        qpos_reset_r[i, 3:6] = wrist_pose[0, :]

        target_center[i, :] = bias[:]
        object_center[i, :] = env.affordance_center[i]


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

    obs_new_r, _ = env.observe(contain_non_aff)
    rewards_r_sum = env.get_reward_info_r()
    for i in range(len(rewards_r_sum)):
        rewards_r_sum[i]['affordance_reward'] = 0
        rewards_r_sum[i]['anatomy_reward'] = 0

        for k in rewards_r_sum[i].keys():
            rewards_r_sum[i][k] = 0

    for step in range(n_steps_r):
        obs_r = obs_new_r
        obs_r = obs_r[:].astype('float32')

        if np.isnan(obs_r).any():
            np.savetxt(saver.data_dir + "/nan_obs.txt", obs_r)
        action_r = ppo_r.act(obs_r)
        action_l = np.zeros_like(action_r)

        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))

        obs_new_r, dis_info = env.observe(contain_non_aff)
        obs_new_r = obs_new_r[:].astype('float32')

        original_pose = obs_new_r[:, 3:51].astype('float32').copy()
        original_pose = rotations.euler2axisangle(original_pose.reshape(-1, 3)).reshape(num_envs, 48)
        mano_output: MANOOutput = ManoTrans.mano_layer(torch.from_numpy(original_pose).to('cuda'), zero_shape)
        T_g_p = mano_output.transforms_abs  # (B, 16, 4, 4)
        T_g_a, R_mat, ee = ManoTrans.axisFK(T_g_p)
        loss = np.sum(anatomyLoss(ee).cpu().detach().numpy(), axis=1)
        curr_ee = ee.reshape(num_envs, -1).cpu().detach().numpy()
        direction_loss = np.sum(np.square(obs_new_r[:, 3:6]), axis=1)
        center_loss = np.sum(np.square(obs_new_r[:, :3]), axis=1)

        rewards_r = env.get_reward_info_r()
        affordance_reward_r = - np.sum((dis_info[:, :21]) * finger_weights, axis=1)
        non_affordance_reward_r = - np.sum((dis_info[:, 21:]) * finger_weights, axis=1)

        for i in range(num_envs):
            rewards_r[i]['affordance_reward'] = affordance_reward_r[i] * cfg['environment']['reward']['affordance_reward']['coeff']
            rewards_r[i]['not_affordance_reward'] = non_affordance_reward_r[i] * cfg['environment']['reward']['not_affordance_reward']['coeff']
            rewards_r[i]['anatomy_reward'] = loss[i] * cfg['environment']['reward']['anatomy_reward']['coeff']
            rewards_r[i]['direction_reward'] = direction_loss[i] * cfg['environment']['reward']['direction_reward']['coeff']
            rewards_r[i]['center_reward'] = center_loss[i] * cfg['environment']['reward']['center_reward']['coeff']
            rewards_r[i]['reward_sum'] = (rewards_r[i]['reward_sum'] + rewards_r[i]['affordance_reward']
                                          + rewards_r[i]['not_affordance_reward'] + rewards_r[i]['anatomy_reward']
                                          + rewards_r[i]['direction_reward'] + rewards_r[i]['center_reward'])

            reward_r[i] = rewards_r[i]['reward_sum']
        reward_r.clip(min=reward_clip)

        for i in range(len(rewards_r_sum)):
            for k in rewards_r_sum[i].keys():
                rewards_r_sum[i][k] = rewards_r_sum[i][k] + rewards_r[i][k]

        ppo_r.step(value_obs=obs_r, rews=reward_r, dones=dones)

    obs_r, _ = env.observe(contain_non_aff)
    obs_r = obs_r[:, :].astype('float32')

    # update policy
    ppo_r.update(actor_obs=obs_r, value_obs=obs_r, log_this_iteration=update % 10 == 0, update=update)

    actor_r.distribution.enforce_minimum_std((torch.ones(act_dim) * 0.2).to(device))

    end = time.time()

    ave_reward = {}
    for k in rewards_r_sum[0].keys():
        ave_reward[k] = 0
    for k in rewards_r_sum[0].keys():
        for i in range(len(rewards_r_sum)):
            ave_reward[k] = ave_reward[k] + rewards_r_sum[i][k]
        ave_reward[k] = ave_reward[k] / (len(rewards_r_sum) * n_steps_r)
    wandb.log(ave_reward)

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average reward: ", '{:0.10f}'.format(ave_reward['reward_sum'])))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps_r / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps_r / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('std: ')
    print(np.exp(actor_r.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')