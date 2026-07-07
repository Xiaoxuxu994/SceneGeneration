import multiprocessing as mp
import os


def run_cmd(cmd):
    os.system(cmd)


NODE_START_PROC = int(os.environ["NODE_START_PROC"])
NODE_END_PROC   = int(os.environ["NODE_END_PROC"])
TOTAL_PROC      = int(os.environ["TOTAL_PROC"])
ROOT = os.environ["DATA_ROOT"]
cmd_list = []
for proc_id in range(NODE_START_PROC, NODE_END_PROC):
    cmd = "NPU_VISIBLE_DEVICES={} ASCEND_RT_VISIBLE_DEVICES={} python -m dataset.reliev3r_dataset.data_process.gen_pseudo_matching_romav2 --root {} --total_proc {} --proc_id {}".format(
        proc_id - NODE_START_PROC, proc_id - NODE_START_PROC, ROOT, TOTAL_PROC, proc_id
    )
    cmd_list.append(cmd)
processes = [mp.Process(target=run_cmd, args=(cmd, )) for cmd in cmd_list]
for p in processes:
    p.start()
for p in processes:
    p.join()
