import multiprocessing
import os
from argparse import ArgumentParser


def subproc(proc_id, total_proc):
    os.system("python sekai_annometa2clips.py --proc_id {} --total_proc {}".format(proc_id, total_proc))


def main(args):
    processes = []
    for proc_id in range(args.proc_start, args.proc_end):
        process = multiprocessing.Process(target=subproc, args=(proc_id, args.total_proc))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--proc_start", type=int, required=True)
    parser.add_argument("--proc_end", type=int, required=True)
    parser.add_argument("--total_proc", type=int, required=True)
    args = parser.parse_args()
    main(args)