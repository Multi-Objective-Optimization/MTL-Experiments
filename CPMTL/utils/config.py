import argparse
import yaml


def load_config(config_path, overrides=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for item in (overrides or []):
        keys, val = item.split("=", 1)
        parts = keys.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        for cast in (int, float):
            try:
                val = cast(val)
                break
            except ValueError:
                pass
        d[parts[-1]] = val
    return cfg


def parse_args(default_config):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=default_config, help="path to yaml config")
    p.add_argument("overrides", nargs="*", help="key.subkey=value overrides")
    return p.parse_args()
