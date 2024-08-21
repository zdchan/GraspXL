# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os


class RaisimGymVecEnv:

    def __init__(self, impl, cfg, normalize_ob=False, seed=0, normalize_rew=True, clip_obs=10.):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'

        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.num_obs_r = self.wrapper.getRightObDim()
        self.num_obs_l = self.wrapper.getLeftObDim()
        self.num_acts = self.wrapper.getActionDim()
        self.num_gs = self.wrapper.getGSDim()
        self._observation_r = np.zeros([self.num_envs, self.num_obs_r], dtype=np.float32)
        self._observation_l = np.zeros([self.num_envs, self.num_obs_l], dtype=np.float32)
        self._global_state = np.zeros([self.num_envs, self.num_gs], dtype=np.float32)
        self.obs_rms_r = RunningMeanStd(shape=[self.num_envs, self.num_obs_r])
        self.obs_rms_l = RunningMeanStd(shape=[self.num_envs, self.num_obs_l])
        self.gs_rms = RunningMeanStd(shape=[self.num_envs, self.num_gs])
        self._reward_r = np.zeros(self.num_envs, dtype=np.float32)
        self._reward_l = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def set_pd_wrist(self):
        self.wrapper.set_pd_wrist()

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action_r, action_l):
        self.wrapper.step(action_r, action_l, self._reward_r, self._reward_l, self._done)
        return self._reward_r.copy(), self._reward_l.copy(), self._done.copy() 

    def step2(self, action_r, action_l):
        self.wrapper.step2(action_r, action_l, self._reward_r, self._reward_l, self._done)
        return self._reward_r.copy(), self._reward_l.copy(), self._done.copy()

    def reset_right_hand(self, obj_pose_step_r, hand_ee_step_r, hand_pose_step_r):
        self.wrapper.reset_right_hand(obj_pose_step_r, hand_ee_step_r, hand_pose_step_r)

    def step_imitate(self, action_r, action_l, obj_pose_r, hand_ee_r, hand_pose_r, obj_pose_l, hand_ee_l, hand_pose_l, imitate_right, imitate_left):
        self.wrapper.step_imitate(action_r, action_l, obj_pose_r, hand_ee_r, hand_pose_r, obj_pose_l, hand_ee_l, hand_pose_l, imitate_right, imitate_left, self._reward_r, self._reward_l, self._done)
        return self._reward_r.copy(), self._reward_l.copy(), self._done.copy()

    def load_scaling(self, dir_name, iteration, count=1e5, cent_training=False):
        mean_file_name_r = dir_name + "/mean_r" + str(iteration) + ".csv"
        var_file_name_r = dir_name + "/var_r" + str(iteration) + ".csv"
        mean_file_name_l = dir_name + "/mean_l" + str(iteration) + ".csv"
        var_file_name_l = dir_name + "/var_l" + str(iteration) + ".csv"
        if cent_training:
            mean_file_name_g = dir_name + "/mean_g" + str(iteration) + ".csv"
            var_file_name_g = dir_name + "/var_g" + str(iteration) + ".csv"
        self.obs_rms_r.count = count
        self.obs_rms_r.mean = np.loadtxt(mean_file_name_r, dtype=np.float32)
        self.obs_rms_r.var = np.loadtxt(var_file_name_r, dtype=np.float32)
        self.obs_rms_l.count = count
        self.obs_rms_l.mean = np.loadtxt(mean_file_name_l, dtype=np.float32)
        self.obs_rms_l.var = np.loadtxt(var_file_name_l, dtype=np.float32)
        if cent_training:
            self.gs_rms.count = count
            self.gs_rms.mean = np.loadtxt(mean_file_name_g, dtype=np.float32)
            self.gs_rms.var = np.loadtxt(var_file_name_g, dtype=np.float32)

    def save_scaling(self, dir_name, iteration):
        mean_file_name_r = dir_name + "/mean_r" + iteration + ".csv"
        var_file_name_r = dir_name + "/var_r" + iteration + ".csv"
        mean_file_name_l = dir_name + "/mean_l" + iteration + ".csv"
        var_file_name_l = dir_name + "/var_l" + iteration + ".csv"
        np.savetxt(mean_file_name_r, self.obs_rms_r.mean)
        np.savetxt(var_file_name_r, self.obs_rms_r.var)
        np.savetxt(mean_file_name_l, self.obs_rms_l.mean)
        np.savetxt(var_file_name_l, self.obs_rms_l.var)

    def observe(self, update_mean=True):
        self.wrapper.observe(self._observation_r, self._observation_l)

        if self.normalize_ob:
            if update_mean:
                self.obs_rms_r.update(self._observation_r)
                self.obs_rms_l.update(self._observation_l)

            return self._normalize_observation(self._observation_r, True), self._normalize_observation(self._observation_l, False)
        else:
            return self._observation_r.copy(), self._observation_l.copy()

    def get_global_state(self, update_mean=True):
        self.wrapper.get_global_state(self._global_state)

        if self.normalize_ob:
            if update_mean:
                self.gs_rms.update(self._global_state)

            return self._normalize_global_state(self._global_state)
        else:
            return self._global_state.copy()

    def set_rootguidance(self):
        self.wrapper.set_rootguidance()

    def switch_root_guidance(self, is_on):
        self.wrapper.switch_root_guidance(is_on)

    def control_switch(self, left, right):
        self.wrapper.control_switch(left, right)

    def control_switch_all(self, left, right):
        self.wrapper.control_switch_all(left, right)

    def reset(self):
        self._reward_r = np.zeros(self.num_envs, dtype=np.float32)
        self._reward_l = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def add_stage(self, stage_dim, stage_pos):
        self.wrapper.add_stage(stage_dim, stage_pos)

    def switch_arctic(self, idx):
        self.wrapper.switch_arctic(idx)

    def load_object(self, obj_idx, obj_weight, obj_dim, obj_type):
        self.wrapper.load_object(obj_idx, obj_weight, obj_dim, obj_type)

    def load_articulated(self, obj_model):
        self.wrapper.load_articulated(obj_model)

    def load_multi_articulated(self, obj_models):
        self.wrapper.load_multi_articulated(obj_models)

    def reset_state(self, init_state_r, init_state_l, init_vel_r, init_vel_l, obj_pose):

        self.wrapper.reset_state(init_state_r, init_state_l, init_vel_r, init_vel_l, obj_pose)

    def set_goals_r(self, obj_pos_r, ee_pos_r, pose_r, qpos_r):
        self.wrapper.set_goals_r(obj_pos_r, ee_pos_r, pose_r, qpos_r)

    def set_imitation_goals(self, pose_l, pose_r, obj_pose):
        self.wrapper.set_imitation_goals(pose_l, pose_r, obj_pose)

    def set_goals_r2(self, obj_pos_r, ee_pos_r, pose_r, qpos_r, contact_r):
        self.wrapper.set_goals_r2(obj_pos_r, ee_pos_r, pose_r, qpos_r, contact_r)

    def set_ext(self, ext_force, ext_torque):
        self.wrapper.set_ext(ext_force, ext_torque)

    def set_pregrasp(self, obj_pos, ee_pos, pose):
        self.wrapper.set_pregrasp(obj_pos, ee_pos, pose)

    def set_goals(self, obj_angle, obj_pos, ee_pos_r, ee_pos_l, pose_r, pose_l, qpos_r, qpos_l, contact_r, contact_l):
        self.wrapper.set_goals(obj_angle, obj_pos, ee_pos_r, ee_pos_l, pose_r, pose_l, qpos_r, qpos_l, contact_r, contact_l)

    def set_obj_goal(self, obj_angle, obj_pos):
        self.wrapper.set_obj_goal(obj_angle, obj_pos)

    # def get_checkpoint(self, ptarget):
    #     self.wrapper.getPtarget(ptarget)

    def _normalize_observation(self, obs, is_rhand):
        if self.normalize_ob:
            if is_rhand:
                return np.clip((obs - self.obs_rms_r.mean) / np.sqrt(self.obs_rms_r.var + 1e-8), -self.clip_obs,
                                self.clip_obs)
            else:
                return np.clip((obs - self.obs_rms_l.mean) / np.sqrt(self.obs_rms_l.var + 1e-8), -self.clip_obs,
                                self.clip_obs)
        else:
            return obs

    def _normalize_global_state(self, gs):
        if self.normalize_ob:
            return np.clip((gs - self.gs_rms.mean) / np.sqrt(self.gs_rms.var + 1e-8), -self.clip_obs,
                                self.clip_obs)            
        else:
            return gs

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def get_reward_info(self):
        return self.wrapper.rewardInfoLeft()

    def get_reward_info_r(self):
        return self.wrapper.rewardInfoRight()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

