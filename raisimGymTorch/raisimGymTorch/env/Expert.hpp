#ifndef _RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_EXPERT_HPP_
#define _RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_EXPERT_HPP_

#include <initializer_list>
#include <string>
#include <map>
#include "Yaml.hpp"
#include "Experts.hpp"

namespace raisim {

    struct ExpertElement {
        EigenRowMajorMat qpos;
        EigenRowMajorMat qvel;
        EigenRowMajorMat joint_3dpos;
    };

    class Expert{
    public:
        Expert (std::initializer_list<int> trajectory_steps) {
            for(auto& idx: trajectory_steps)
                trajectory_[idx] = raisim::ExpertElement();
        }

        Expert () = default;

        void loadExpert(int expert_idx, const Eigen::Ref<EigenRowMajorMat> &qpos, const Eigen::Ref<EigenRowMajorMat> &qvel, const Eigen::Ref<EigenRowMajorMat> &joint_3dpos) {
            // here a trajectory should be loaded into the expert
//            for(int i = 0; i < expert_len; i++) {
            trajectory_[expert_idx] = raisim::ExpertElement();
            trajectory_[expert_idx].qpos = qpos.cast<double>();
            trajectory_[expert_idx].qvel = qvel.cast<double>();
            trajectory_[expert_idx].joint_3dpos = joint_3dpos.cast<double>();
//            }
        }

        const Eigen::Ref<EigenVec>& sampleInitState(){
            //This function samples a random initial state from the trajectory
            auto it = trajectory_.begin();
            std::next(it, rand() % trajectory_.size());
            int random_key = it->first;
            return trajectory_[random_key].qpos;
        }



    private:
        std::map<int, raisim::ExpertElement> trajectory_;
    };
}// namespace raisim

#endif //_RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_REWARD_HPP_
