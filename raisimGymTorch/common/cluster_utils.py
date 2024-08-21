import multiprocessing.dummy as mp

from tqdm import tqdm

import common.list_utils as list_utils


class CPUCluster:
    def __init__(self, tasks, op, num_threads, num_nodes):
        super().__init__()
        self.chunks = list_utils.chunks(tasks, num_nodes)
        self.op = op
        self.num_threads = num_threads
        self.num_nodes = num_nodes

    def run(self, chunk_id):
        if self.num_threads > 1:
            p = mp.Pool(self.num_threads)
            p.map(self.op, self.chunks[chunk_id])
            p.close()
            p.join()
        else:
            for task in tqdm(self.chunks[chunk_id]):
                self.op(task)

    def __len__(self):
        return len(self.chunks)
