//#ifndef _RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_EXPERTS_HPP_
//#define _RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_EXPERTS_HPP_
//
//#include <initializer_list>
//#include <string>
//#include <map>
//#include "Yaml.hpp"
//#include <Expert.hpp>
//
//namespace raisim {
//
//
//    class Experts {
//    public:
//        Experts (std::initializer_list<int> experts) {
//            for(auto& idx: experts)
//                experts_[idx] = raisim::Expert();
//        }
//
//        Experts () = default;
//
//        void loadExperts(const Yaml::Node& cfg) {
//            //should load each expert into the experts map
//
//
//        }
//
//        raisim::Expert sampleRandomExpert(){
//            //This function samples a random expert from the expert map
//            auto it = experts_.begin();
//            std::next(it, rand() % experts_.size());
//            int random_key = it->first;
//            return experts_[random_key];
//        }
//
//
//    private:
//        std::map<int, raisim::Expert> experts_;
//    };
//
//}
//
//#endif //_RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_REWARD_HPP_
