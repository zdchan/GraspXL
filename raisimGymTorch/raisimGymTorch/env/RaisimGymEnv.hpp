//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMENV_HPP
#define SRC_RAISIMGYMENV_HPP

#include <vector>
#include <memory>
#include <unordered_map>
#include <Eigen/Core>
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Yaml.hpp"
#include "Reward.hpp"

#define __RSG_MAKE_STR(x) #x
#define _RSG_MAKE_STR(x) __RSG_MAKE_STR(x)
#define RSG_MAKE_STR(x) _RSG_MAKE_STR(x)

#define READ_YAML(a, b, c) RSFATAL_IF(!&c, "Node "<<RSG_MAKE_STR(c)<<" doesn't exist") b = c.template As<a>();

namespace raisim {

using Dtype=float;
using Dtype_double=double;
using EigenRowMajorMat=Eigen::Matrix<Dtype, -1, -1, Eigen::RowMajor>;
//using EigenRowMajorMatDouble=Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;
using EigenRowMajorMatInt=Eigen::Matrix<int, -1, -1, Eigen::RowMajor>;
using EigenVec=Eigen::Matrix<Dtype, -1, 1>;
//using EigenVec=Eigen::Matrix<double, -1, 1>;
using EigenVecInt=Eigen::Matrix<int, -1, 1>;
using EigenBoolVec=Eigen::Matrix<bool, -1, 1>;

class RaisimGymEnv {

 public:
  explicit RaisimGymEnv (std::string resourceDir, const Yaml::Node& cfg) : resourceDir_(std::move(resourceDir)), cfg_(cfg) { }

  virtual ~RaisimGymEnv() { close(); };

  /////// implement these methods /////////
  virtual void init() = 0;
  virtual void reset() = 0;

  virtual void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                           const Eigen::Ref<EigenVec>& init_state_l,
                           const Eigen::Ref<EigenVec>& init_vel_r,
                           const Eigen::Ref<EigenVec>& init_vel_l,
                           const Eigen::Ref<EigenVec>& obj_pose) = 0;
  virtual void set_goals(const Eigen::Ref<EigenVec>& obj_angle,
                         const Eigen::Ref<EigenVec>& obj_pos,
                         const Eigen::Ref<EigenVec>& ee_pos_r,
                         const Eigen::Ref<EigenVec>& ee_pos_l,
                         const Eigen::Ref<EigenVec>& pose_r,
                         const Eigen::Ref<EigenVec>& pose_l,
                         const Eigen::Ref<EigenVec>& qpos_r,
                         const Eigen::Ref<EigenVec>& qpos_l,
                         const Eigen::Ref<EigenVec>& contact_r,
                         const Eigen::Ref<EigenVec>& contact_l) = 0;
  virtual void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) = 0;
  virtual void set_rootguidance() = 0;
  virtual float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) = 0;
  virtual bool isTerminalState(float& terminalReward) = 0;
  ////////////////////////////////////////

  /////// optional methods ///////
  virtual void add_stage(const Eigen::Ref<EigenVec>& stage_dim,
                         const Eigen::Ref<EigenVec>& stage_pos) {};
  virtual void set_goals_r(const Eigen::Ref<EigenVec>& obj_pos_r,
                           const Eigen::Ref<EigenVec>& ee_pos_r,
                           const Eigen::Ref<EigenVec>& pose_r,
                           const Eigen::Ref<EigenVec>& qpos_r){};
  virtual void set_imitation_goals(const Eigen::Ref<EigenVec>& pose_l,
                           const Eigen::Ref<EigenVec>& pose_r,
                           const Eigen::Ref<EigenVec>& obj_pos){};
  virtual void set_goals_r2(const Eigen::Ref<EigenVec>& obj_pos_r,
                            const Eigen::Ref<EigenVec>& ee_pos_r,
                            const Eigen::Ref<EigenVec>& pose_r,
                            const Eigen::Ref<EigenVec>& qpos_r,
                            const Eigen::Ref<EigenVec>& contact_r){};
  virtual void set_obj_goal(const Eigen::Ref<EigenVec>& obj_angle,
                            const Eigen::Ref<EigenVec>& obj_pos){};
  virtual void set_ext(const Eigen::Ref<EigenVec>& ext_force,
                       const Eigen::Ref<EigenVec>& ext_torque){};
  virtual void set_pd_wrist(){};
  virtual void set_pregrasp(const Eigen::Ref<EigenVec>& obj_pos,
                           const Eigen::Ref<EigenVec>& ee_pos,
                           const Eigen::Ref<EigenVec>& pose){};
  virtual void get_global_state(Eigen::Ref<EigenVec> gs){};
  virtual void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type){};
  virtual void load_articulated(const std::string& obj_model){};
  virtual void load_multi_articulated(const std::vector<std::string>& obj_models){};
  virtual void switch_root_guidance(bool is_on){};
  virtual void control_switch(int right, int left){};
  virtual void control_switch_all(const Eigen::Ref<EigenVec>& right, const Eigen::Ref<EigenVec>& left){};
  virtual void curriculumUpdate() {};
  virtual void close() { if(server_) server_->killServer(); };
  virtual void setSeed(int seed) {};
  virtual float* step2(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) {};
  virtual float* step_imitate(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l, const Eigen::Ref<EigenVec>& obj_pose_r, const Eigen::Ref<EigenVec>& hand_ee_r, const Eigen::Ref<EigenVec>& hand_pose_r, const Eigen::Ref<EigenVec>& obj_pose_l, const Eigen::Ref<EigenVec>& hand_ee_l, const Eigen::Ref<EigenVec>& hand_pose_l, const bool imitate_right, const bool imitate_left) {};
  virtual void reset_right_hand(const Eigen::Ref<EigenVec>& obj_pose_step_r,
                              const Eigen::Ref<EigenVec>& hand_ee_step_r,
                              const Eigen::Ref<EigenVec>& hand_pose_step_r){};
  virtual void switch_arctic(int idx){};
  virtual void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_vector){};

  ////////////////////////////////

  void setSimulationTimeStep(double dt) { simulation_dt_ = dt; world_->setTimeStep(dt); }
  void setControlTimeStep(double dt) { control_dt_ = dt; }
  int getRightObDim() { return obDim_r_; }
  int getLeftObDim() { return obDim_l_; }
  int getActionDim() { return actionDim_; }
  int getGSDim() { return gsDim_; }
  double getControlTimeStep() { return control_dt_; }
  double getSimulationTimeStep() { return simulation_dt_; }
  raisim::World* getWorld() { return world_.get(); }
  void turnOffVisualization() { server_->hibernate(); }
  void turnOnVisualization() { server_->wakeup(); }
  void startRecordingVideo(const std::string& videoName ) { server_->startRecordingVideo(videoName); }
  void stopRecordingVideo() { server_->stopRecordingVideo(); }
  raisim::Reward& getRewardsRight() { return rewards_r_; }
  raisim::Reward& getRewardsLeft() { return rewards_l_; }
  const std::vector<std::map<std::string, float>>& getRewardInfoRight();
  const std::vector<std::map<std::string, float>>& getRewardInfoLeft();
//  void getPtarget(Eigen::VectorXd& ptarget) {ptarget = ptarget_clipped;}

 protected:
  std::unique_ptr<raisim::World> world_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  std::string resourceDir_;
  Yaml::Node cfg_;
  int obDim_r_ = 0, obDim_l_ = 0, actionDim_ = 0, gsDim_ = 0;
  std::unique_ptr<raisim::RaisimServer> server_;
  raisim::Reward rewards_r_;
  raisim::Reward rewards_l_;
  Eigen::VectorXd ptarget_clipped;
};

}

#endif //SRC_RAISIMGYMENV_HPP
