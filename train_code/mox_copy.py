import moxing as mox
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "srt",
        type=str,
    )
    parser.add_argument(
        "dst",
        type=str,
    )
    args, unknown = parser.parse_known_args()
    mox.file.copy_parallel(args.srt, args.dst)