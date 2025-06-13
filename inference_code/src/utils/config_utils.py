import argparse
import importlib.util


def parse_args():
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("--config", default='./configs/train_configs.py', type=str, help="model config file path")

    return parser.parse_args()


def load_py_config(file_path):
    spec = importlib.util.spec_from_file_location("config_module", file_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def parse_configs():
    args = parse_args()
    cfg = load_py_config(args.config)
    return cfg
