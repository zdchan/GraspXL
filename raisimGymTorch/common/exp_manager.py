from common.condor_cluster import CondorCluster


class ExpManager:
    def __init__(self, args):
        self.args = args

    def run_experiment(self, use_cluster, script, num_exp, logdir_format="LOG_DIR+"):
        if use_cluster:
            condor_cluster = CondorCluster(
                args=self.args, script=script, num_exp=num_exp
            )
            # create .sub
            condor_cluster.submit()
            exit()
