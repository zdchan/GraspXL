from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput
import numpy as np
import torch
from raisimGymTorch.helper import rotations


class PoseTrans():
    def __init__(self):
        self.axisFK = AxisLayerFK(mano_assets_root="~/manotorch/assets/mano").to('cuda')
        self.mano_layer = ManoLayer(rot_mode="axisang",
                               center_idx=9,
                               mano_assets_root="~/manotorch/assets/mano",
                               use_pca=False,
                               flat_hand_mean=True).to('cuda')
        self.action_clip_min = np.zeros((1, 20))
        self.action_clip_max = np.zeros((1, 20))
        for i in range(20):
            if i % 4 == 0:
                self.action_clip_min[:, i] = -0.2
                self.action_clip_max[:, i] = 0.2
            else:
                self.action_clip_min[:, i] = -0.05
                self.action_clip_max[:, i] = 1.8
        self.action_clip_min[:, -4] = -0.3
        self.action_clip_max[:, -4] = 1.0

    def obs_trans(self, gc):
        num_envs = gc.shape[0]
        obs_pose = np.zeros((num_envs, 26)).astype('float32')

        original_pose_ee = np.zeros((num_envs, 48)).astype('float32')
        original_pose_ee[:, :] = gc[:, 3:51].copy()
        original_pose = rotations.euler2axisangle(original_pose_ee.reshape(-1, 3)).reshape(num_envs, 48)
        zero_shape = torch.from_numpy(np.zeros((num_envs, 10)).astype('float32')).float().to('cuda')
        mano_output: MANOOutput = self.mano_layer(torch.from_numpy(original_pose).float().to('cuda'), zero_shape.float().to('cuda'))
        T_g_p = mano_output.transforms_abs  # (B, 16, 4, 4)
        T_g_a, R, ee = self.axisFK(T_g_p.float().to('cuda'))
        obs_full_pose = ee.reshape(num_envs, -1).cpu().numpy()  # (B, 48)

        obs_pose[:, :6] = gc[:, :6]
        # obs_pose[:, 3:6] = obs_full_pose[:, :3]
        obs_pose[:, 6] = obs_full_pose[:, 4]
        obs_pose[:, 7] = obs_full_pose[:, 5]
        obs_pose[:, 8] = obs_full_pose[:, 8]
        obs_pose[:, 9] = obs_full_pose[:, 11]
        obs_pose[:, 10] = obs_full_pose[:, 13]
        obs_pose[:, 11] = obs_full_pose[:, 14]
        obs_pose[:, 12] = obs_full_pose[:, 17]
        obs_pose[:, 13] = obs_full_pose[:, 20]
        obs_pose[:, 14] = obs_full_pose[:, 22]
        obs_pose[:, 15] = obs_full_pose[:, 23]
        obs_pose[:, 16] = obs_full_pose[:, 26]
        obs_pose[:, 17] = obs_full_pose[:, 29]
        obs_pose[:, 18] = obs_full_pose[:, 31]
        obs_pose[:, 19] = obs_full_pose[:, 32]
        obs_pose[:, 20] = obs_full_pose[:, 35]
        obs_pose[:, 21] = obs_full_pose[:, 38]
        obs_pose[:, 22] = obs_full_pose[:, 40]
        obs_pose[:, 23] = obs_full_pose[:, 41]
        obs_pose[:, 24] = obs_full_pose[:, 44]
        obs_pose[:, 25] = obs_full_pose[:, 47]

        return obs_pose

    def obs_trans_verse(self, obs_26):
        num_envs = obs_26.shape[0]
        obs_full_pose = np.zeros((num_envs, 48)).astype('float32')
        obs_full_pose[:, :3] = obs_26[:, 3:6]
        obs_full_pose[:, 4] = obs_26[:, 6]
        obs_full_pose[:, 5] = obs_26[:, 7]
        obs_full_pose[:, 8] = obs_26[:, 8]
        obs_full_pose[:, 11] = obs_26[:, 9]
        obs_full_pose[:, 13] = obs_26[:, 10]
        obs_full_pose[:, 14] = obs_26[:, 11]
        obs_full_pose[:, 17] = obs_26[:, 12]
        obs_full_pose[:, 20] = obs_26[:, 13]
        obs_full_pose[:, 22] = obs_26[:, 14]
        obs_full_pose[:, 23] = obs_26[:, 15]
        obs_full_pose[:, 26] = obs_26[:, 16]
        obs_full_pose[:, 29] = obs_26[:, 17]
        obs_full_pose[:, 31] = obs_26[:, 18]
        obs_full_pose[:, 32] = obs_26[:, 19]
        obs_full_pose[:, 35] = obs_26[:, 20]
        obs_full_pose[:, 38] = obs_26[:, 21]
        obs_full_pose[:, 40] = obs_26[:, 22]
        obs_full_pose[:, 41] = obs_26[:, 23]
        obs_full_pose[:, 44] = obs_26[:, 24]
        obs_full_pose[:, 47] = obs_26[:, 25]
        obs_full_pose = obs_full_pose.reshape(num_envs, 16, 3)
        obs_mano_pose = self.axisFK.compose(torch.from_numpy(obs_full_pose).float().to('cuda')).clone()  # (B=1, 16, 3)
        obs_mano_pose = obs_mano_pose.reshape(num_envs, -1)  # (1, 16x3)
        mano_gc = np.zeros((num_envs, 51))
        mano_gc[:, 3:] = obs_mano_pose.cpu().numpy()
        mano_gc[:, 6:] = rotations.axisangle2euler(mano_gc[:, 3:].reshape(-1, 3)).reshape(-1, 48)[:, 3:]
        mano_gc[:, :6] = obs_26[:, :6]
        return mano_gc


    def action_trans(self, action_26, current_state):
        num_envs = action_26.shape[0]
        action_clip_min = self.action_clip_min.repeat(num_envs, axis=0)
        action_clip_max = self.action_clip_max.repeat(num_envs, axis=0)

        action_26[:, :6] = action_26[:, :6] * 0.005
        action_26[:, 6:] = action_26[:, 6:] * 0.015

        # action_26[:] = action_26[:] + current_state[:]
        action_26[:, 6:] = action_26[:, 6:] + current_state[:, 6:]

        target_gc = action_26.copy()

        action_clipped_r_temp = np.clip(action_26[:, 6:], action_clip_min, action_clip_max)
        target_gc[:, 6:] = action_clipped_r_temp

        action_clipped_r = np.zeros((num_envs, 48)).astype('float32')
        action_clipped_r[:, :3] = action_26[:, 3:6]
        action_clipped_r[:, 4] = action_clipped_r_temp[:, 0]
        action_clipped_r[:, 5] = action_clipped_r_temp[:, 1]
        action_clipped_r[:, 8] = action_clipped_r_temp[:, 2]
        action_clipped_r[:, 11] = action_clipped_r_temp[:, 3]
        action_clipped_r[:, 13] = action_clipped_r_temp[:, 4]
        action_clipped_r[:, 14] = action_clipped_r_temp[:, 5]
        action_clipped_r[:, 17] = action_clipped_r_temp[:, 6]
        action_clipped_r[:, 20] = action_clipped_r_temp[:, 7]
        action_clipped_r[:, 22] = action_clipped_r_temp[:, 8]
        action_clipped_r[:, 23] = action_clipped_r_temp[:, 9]
        action_clipped_r[:, 26] = action_clipped_r_temp[:, 10]
        action_clipped_r[:, 29] = action_clipped_r_temp[:, 11]
        action_clipped_r[:, 31] = action_clipped_r_temp[:, 12]
        action_clipped_r[:, 32] = action_clipped_r_temp[:, 13]
        action_clipped_r[:, 35] = action_clipped_r_temp[:, 14]
        action_clipped_r[:, 38] = action_clipped_r_temp[:, 15]
        action_clipped_r[:, 40] = action_clipped_r_temp[:, 16]
        action_clipped_r[:, 41] = action_clipped_r_temp[:, 17]
        action_clipped_r[:, 44] = action_clipped_r_temp[:, 18]
        action_clipped_r[:, 47] = action_clipped_r_temp[:, 19]
        action_clipped_r = action_clipped_r.reshape(num_envs, 16, 3)
        action_mano_r = self.axisFK.compose(torch.from_numpy(action_clipped_r).float().to('cuda')).clone()  # (B=1, 16, 3)
        action_mano_r = action_mano_r.reshape(num_envs, -1)  # (1, 16x3)
        action_grasp_r = np.zeros((num_envs, 51))
        action_grasp_r[:, 3:] = action_mano_r.cpu().numpy()

        action_grasp_r[:, 6:] = rotations.axisangle2euler(action_grasp_r[:, 3:].reshape(-1, 3)).reshape(-1, 48)[:, 3:]
        action_grasp_r[:, :6] = action_26[:, :6]
        return action_grasp_r, target_gc



    def obs_trans_new(self, gc):
        num_envs = gc.shape[0]
        obs_pose = np.zeros((num_envs, 26)).astype('float32')

        # original_pose_ee = np.zeros((num_envs, 48)).astype('float32')
        original_pose_ee = gc[:, 3:51].copy()
        original_pose = rotations.euler2axisangle(original_pose_ee.reshape(-1, 3)).reshape(num_envs, 48)
        zero_shape = torch.from_numpy(np.zeros((num_envs, 10)).astype('float32')).float().to('cuda')
        mano_output: MANOOutput = self.mano_layer(torch.from_numpy(original_pose).float().to('cuda'), zero_shape.float().to('cuda'))
        T_g_p = mano_output.transforms_abs  # (B, 16, 4, 4)
        T_g_a, R, ee = self.axisFK(T_g_p.float().to('cuda'))
        obs_full_pose = ee.reshape(num_envs, -1).cpu().numpy()  # (B, 48)

        obs_pose[:, :6] = gc[:, :6]
        # obs_pose[:, 3:6] = obs_full_pose[:, :3]
        obs_pose[:, 6] = obs_full_pose[:, 4]
        obs_pose[:, 7] = obs_full_pose[:, 5]
        obs_pose[:, 8] = obs_full_pose[:, 8]
        obs_pose[:, 9] = obs_full_pose[:, 11]
        obs_pose[:, 10] = obs_full_pose[:, 13]
        obs_pose[:, 11] = obs_full_pose[:, 14]
        obs_pose[:, 12] = obs_full_pose[:, 17]
        obs_pose[:, 13] = obs_full_pose[:, 20]
        obs_pose[:, 14] = obs_full_pose[:, 22]
        obs_pose[:, 15] = obs_full_pose[:, 23]
        obs_pose[:, 16] = obs_full_pose[:, 26]
        obs_pose[:, 17] = obs_full_pose[:, 29]
        obs_pose[:, 18] = obs_full_pose[:, 31]
        obs_pose[:, 19] = obs_full_pose[:, 32]
        obs_pose[:, 20] = obs_full_pose[:, 35]
        obs_pose[:, 21] = obs_full_pose[:, 38]
        obs_pose[:, 22] = obs_full_pose[:, 40]
        obs_pose[:, 23] = obs_full_pose[:, 41]
        obs_pose[:, 24] = obs_full_pose[:, 44]
        obs_pose[:, 25] = obs_full_pose[:, 47]

        return obs_pose


    def action_trans_new(self, action_26, obs_full_pose):
        num_envs = action_26.shape[0]


        action_26[:, :6] = action_26[:, :6] * 0.005
        action_26[:, 6:] = action_26[:, 6:] * 0.015
        action_clipped_r = obs_full_pose.copy()


        # action_clipped_r[:, :3] = action_26[:, 3:6]
        # action_clipped_r[:, 4] += action_26[:, 6]
        # action_clipped_r[:, 5] += action_26[:, 7]
        # action_clipped_r[:, 8] += action_26[:, 8]
        # action_clipped_r[:, 11] += action_26[:, 9]
        # action_clipped_r[:, 13] += action_26[:, 10]
        # action_clipped_r[:, 14] += action_26[:, 11]
        # action_clipped_r[:, 17] += action_26[:, 12]
        # action_clipped_r[:, 20] += action_26[:, 13]
        # action_clipped_r[:, 22] += action_26[:, 14]
        # action_clipped_r[:, 23] += action_26[:, 15]
        # action_clipped_r[:, 26] += action_26[:, 16]
        # action_clipped_r[:, 29] += action_26[:, 17]
        # action_clipped_r[:, 31] += action_26[:, 18]
        # action_clipped_r[:, 32] += action_26[:, 19]
        # action_clipped_r[:, 35] += action_26[:, 20]
        # action_clipped_r[:, 38] += action_26[:, 21]
        # action_clipped_r[:, 40] += action_26[:, 22]
        # action_clipped_r[:, 41] += action_26[:, 23]
        # action_clipped_r[:, 44] += action_26[:, 24]
        # action_clipped_r[:, 47] += action_26[:, 25]


        action_clip_min = self.action_clip_min.repeat(num_envs, axis=0)
        action_clip_max = self.action_clip_max.repeat(num_envs, axis=0)

        action_26[:, 6] += obs_full_pose[:, 4]
        action_26[:, 7] += obs_full_pose[:, 5]
        action_26[:, 8] += obs_full_pose[:, 8]
        action_26[:, 9] += obs_full_pose[:, 11]
        action_26[:, 10] += obs_full_pose[:, 13]
        action_26[:, 11] += obs_full_pose[:, 14]
        action_26[:, 12] += obs_full_pose[:, 17]
        action_26[:, 13] += obs_full_pose[:, 20]
        action_26[:, 14] += obs_full_pose[:, 22]
        action_26[:, 15] += obs_full_pose[:, 23]
        action_26[:, 16] += obs_full_pose[:, 26]
        action_26[:, 17] += obs_full_pose[:, 29]
        action_26[:, 18] += obs_full_pose[:, 31]
        action_26[:, 19] += obs_full_pose[:, 32]
        action_26[:, 20] += obs_full_pose[:, 35]
        action_26[:, 21] += obs_full_pose[:, 38]
        action_26[:, 22] += obs_full_pose[:, 40]
        action_26[:, 23] += obs_full_pose[:, 41]
        action_26[:, 24] += obs_full_pose[:, 44]
        action_26[:, 25] += obs_full_pose[:, 47]
        action_clipped_r_temp = np.clip(action_26[:, 6:], action_clip_min, action_clip_max)

        action_clipped_r[:, 4] = action_clipped_r_temp[:, 0]
        action_clipped_r[:, 5] = action_clipped_r_temp[:, 1]
        action_clipped_r[:, 8] = action_clipped_r_temp[:, 2]
        action_clipped_r[:, 11] = action_clipped_r_temp[:, 3]
        action_clipped_r[:, 13] = action_clipped_r_temp[:, 4]
        action_clipped_r[:, 14] = action_clipped_r_temp[:, 5]
        action_clipped_r[:, 17] = action_clipped_r_temp[:, 6]
        action_clipped_r[:, 20] = action_clipped_r_temp[:, 7]
        action_clipped_r[:, 22] = action_clipped_r_temp[:, 8]
        action_clipped_r[:, 23] = action_clipped_r_temp[:, 9]
        action_clipped_r[:, 26] = action_clipped_r_temp[:, 10]
        action_clipped_r[:, 29] = action_clipped_r_temp[:, 11]
        action_clipped_r[:, 31] = action_clipped_r_temp[:, 12]
        action_clipped_r[:, 32] = action_clipped_r_temp[:, 13]
        action_clipped_r[:, 35] = action_clipped_r_temp[:, 14]
        action_clipped_r[:, 38] = action_clipped_r_temp[:, 15]
        action_clipped_r[:, 40] = action_clipped_r_temp[:, 16]
        action_clipped_r[:, 41] = action_clipped_r_temp[:, 17]
        action_clipped_r[:, 44] = action_clipped_r_temp[:, 18]
        action_clipped_r[:, 47] = action_clipped_r_temp[:, 19]

        action_mano_r = self.axisFK.compose(torch.from_numpy(action_clipped_r.reshape(num_envs, 16, 3)).float().to('cuda')) # (B=1, 16, 3)
        action_grasp_r = np.zeros((num_envs, 51))
        action_grasp_r[:, 3:] = action_mano_r.reshape(num_envs, -1).cpu().numpy()
        action_grasp_r[:, 6:] = rotations.axisangle2euler(action_grasp_r[:, 3:].reshape(-1, 3)).reshape(-1, 48)[:, 3:]
        action_grasp_r[:, :6] = action_26[:, :6]
        return action_grasp_r