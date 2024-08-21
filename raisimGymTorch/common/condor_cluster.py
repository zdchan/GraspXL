import os
import stat
import subprocess

from loguru import logger


def add_cluster_args(parser):
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--num_exp", type=int, default=1, help="log every k steps")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--cluster_node", type=str, default="")
    parser.add_argument("--bid", type=int, default=21, help="log every k steps")
    parser.add_argument(
        "--gpu_min_mem", type=int, default=16000, help="log every k steps"
    )
    parser.add_argument("--memory", type=int, default=8000, help="log every k steps")
    parser.add_argument("--num_workers", type=int, default=8, help="log every k steps")
    parser.add_argument("--gpu_arch", type=str, default="all", help="log every k steps")
    parser.add_argument("--num_gpus", type=int, default=0, help="log every k steps")
    return parser


class CondorCluster:
    GPUS = {
        "v100-p16": ('"Tesla V100-PCIE-16GB"', "tesla", 16000),
        "v100-p32": ('"Tesla V100-PCIE-32GB"', "tesla", 32000),
        "v100-s32": ('"Tesla V100-SXM2-32GB"', "tesla", 32000),
        "quadro6000": ('"Quadro RTX 6000"', "quadro", 24000),
        "rtx2080ti": ('"GeForce RTX 2080 Ti"', "rtx", 11000),
        "a100-80": ('"NVIDIA A100-SXM-80GB"', "ampere", 80000),
        "a100-40": ('"NVIDIA A100-SXM4-40GB"', "ampere", 40000),
    }
    SKIP_NODES = ["g120", "g124"]

    def __init__(
        self,
        args,
        script,
        num_exp=1,
    ):
        """
        :param script: (str) python script which will be executed eg. "main.py"
        :param cfg: (yacs.config.CfgNode) CfgNode object
        :param cfg_file: (str) path to yaml config file eg. config.yaml
        """
        self.script = script
        self.num_exp = num_exp
        self.cfg = args

    def submit(self):
        log_dir = self.cfg.log_dir
        submit_p = self.cfg.submit_p
        run_p = self.cfg.run_p
        repo_p = self.cfg.repo_p

        gpus = self._get_gpus(min_mem=self.cfg.gpu_min_mem, arch=self.cfg.gpu_arch)
        gpus = " || ".join([f"CUDADeviceName=={x}" for x in gpus])

        self._create_submission_file(run_p, gpus, submit_p)
        self._create_bash_file(run_p)

        logger.info(f"The logs for this experiments can be found under: {log_dir}")
        cmd = ["condor_submit_bid", f"{self.cfg.bid}", submit_p]
        logger.info("Executing " + " ".join(cmd))
        subprocess.call(cmd)

    def _create_requirements(self, gpus, skip_nodes):
        gpu_req = f"({gpus})"
        skip_nodes = [f'(UtsnameNodename =!= "{node}")' for node in skip_nodes]
        all_req = [gpu_req] + skip_nodes
        all_req = " && ".join(all_req)
        return all_req

    def _create_submission_file(self, run_script, gpus, submit_p):
        log_dir = "/".join(submit_p.split("/")[:-2])
        req_str = self._create_requirements(gpus, self.SKIP_NODES)
        submission = (
            f"executable = {run_script}\n"
            "arguments = $(Process) $(Cluster)\n"
            f"error = {log_dir}/condor/log_$(Cluster).$(Process).err\n"
            f"output = {log_dir}/condor/log_$(Cluster).$(Process).out\n"
            f"log = {log_dir}/condor/log_$(Cluster).$(Process).log\n"
            f"request_memory = {self.cfg.memory}\n"
            f"request_cpus={32}\n"
            # f"request_cpus={min(int(self.cfg.num_workers), 16)}\n"
            f"request_gpus={self.cfg.num_gpus}\n"
            f"requirements={req_str}\n"
            "+MaxRunningPrice = 500\n"
            '+RunningPriceExceededAction = "restart"\n'
            f"queue 1"
            # f"queue {self.num_exp}"
        )
        # f'next_job_start_delay=10\n' \

        with open(submit_p, "w") as f:
            f.write(submission)

    def _create_bash_file(self, run_p):
        exp_key = os.path.normpath(run_p).split("/")[-3]
        api_key = os.environ["COMET_API_KEY"]
        workspace = os.environ["COMET_WORKSPACE"]
        bash = "export PYTHONBUFFERED=1\n"
        bash += "export PATH=$PATH\n"
        bash += f'export COMET_API_KEY="{api_key}"\n'
        bash += f'export COMET_WORKSPACE="{workspace}"\n'
        bash += f'export MANO_MODEL_DIR_R="/home/fzicong/nether/common/body_models/mano/MANO_RIGHT.pkl"\n'
        bash += f'export MANO_MODEL_DIR_L="/home/fzicong/nether/common/body_models/mano/MANO_LEFT.pkl"\n'
        bash += (
            f'export MANO_MODEL_DIR="/home/fzicong/nether/common/body_models/mano"\n'
        )
        bash += 'export EMAIL_ACC="hijason78@gmail.com"\n'
        bash += 'export EMAIL_PASS="denote.foist.finnish.seasick"\n'
        bash += f"cd {self.cfg.repo_p}\n"
        bash += (
            f'/home/fzicong/anaconda3/envs/arctic_env/bin/python {self.script}'
        )
        bash += " --agent_id $1"
        bash += " --cluster_node $2.$1"
        bash += f" --exp_key {exp_key}"

        with open(run_p, "w") as f:
            f.write(bash)

        os.chmod(run_p, stat.S_IRWXU)

    def _get_gpus(self, min_mem, arch):
        if arch == "all":
            arch = ["tesla", "quadro", "rtx", "ampere"]

        gpu_names = []
        for k, (gpu_name, gpu_arch, gpu_mem) in self.GPUS.items():
            if gpu_mem >= min_mem and gpu_arch in arch:
                gpu_names.append(gpu_name)

        assert len(gpu_names) > 0, "Suitable GPU model could not be found"

        return gpu_names
