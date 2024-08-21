//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

namespace py = pybind11;
using namespace raisim;

#ifndef ENVIRONMENT_NAME
  #define ENVIRONMENT_NAME RaisimGymEnv
#endif

PYBIND11_MODULE(RAISIMGYM_TORCH_ENV_NAME, m) {
  py::class_<VectorizedEnvironment<ENVIRONMENT>>(m, RSG_MAKE_STR(ENVIRONMENT_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<ENVIRONMENT>::init)
    .def("reset", &VectorizedEnvironment<ENVIRONMENT>::reset)
    .def("add_stage", &VectorizedEnvironment<ENVIRONMENT>::add_stage)
    .def("load_object", &VectorizedEnvironment<ENVIRONMENT>::load_object)
    .def("load_articulated", &VectorizedEnvironment<ENVIRONMENT>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<ENVIRONMENT>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<ENVIRONMENT>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<ENVIRONMENT>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<ENVIRONMENT>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<ENVIRONMENT>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<ENVIRONMENT>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<ENVIRONMENT>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<ENVIRONMENT>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<ENVIRONMENT>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<ENVIRONMENT>::observe)
    .def("get_global_state", &VectorizedEnvironment<ENVIRONMENT>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<ENVIRONMENT>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<ENVIRONMENT>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<ENVIRONMENT>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<ENVIRONMENT>::control_switch_all)
    .def("step", &VectorizedEnvironment<ENVIRONMENT>::step)
    .def("step2", &VectorizedEnvironment<ENVIRONMENT>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<ENVIRONMENT>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<ENVIRONMENT>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<ENVIRONMENT>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<ENVIRONMENT>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<ENVIRONMENT>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<ENVIRONMENT>::close)
    .def("isTerminalState", &VectorizedEnvironment<ENVIRONMENT>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<ENVIRONMENT>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<ENVIRONMENT>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<ENVIRONMENT>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<ENVIRONMENT>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<ENVIRONMENT>::getNumOfEnvs)
//    .def("getPtarget", &VectorizedEnvironment<ENVIRONMENT>::getPtarget)
    .def("turnOnVisualization", &VectorizedEnvironment<ENVIRONMENT>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<ENVIRONMENT>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<ENVIRONMENT>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<ENVIRONMENT>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<ENVIRONMENT>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<ENVIRONMENT>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<ENVIRONMENT> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<ENVIRONMENT> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
        }
    ));

    py::class_<NormalSampler>(m, "NormalSampler")
    .def(py::init<int>(), py::arg("dim"))
    .def("seed", &NormalSampler::seed)
    .def("sample", &NormalSampler::sample);
}
