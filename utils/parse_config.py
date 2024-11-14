import yaml
import argparse
import copy


def dict2namespace(config):
    namespace = argparse.ArgumentParser()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse():
    parser = argparse.ArgumentParser("training models")
    parser.add_argument("--config", type=str, help="config path")
    # for testing
    parser.add_argument("--model_name", type=str, help="testing model")
    parser.add_argument("--testset", type=str, help="testing set")
    parser.add_argument("--save_path", type=str, help="save file path")
    parser.add_argument("--checkpoint", type=str, help="checkpoint path")
    parser.add_argument("--tile", type=int, default=512, help="test size")
    parser.add_argument("--tile_overlap", type=int, default=0, help="test overlap")
    opt = parser.parse_args()
    opt_test = copy.deepcopy(opt)
    with open(opt.config, "r") as f:
        config = yaml.safe_load(f)
    opt = dict2namespace(config)
    setattr(opt, "model_name", opt_test.model_name)
    setattr(opt, "testset", opt_test.testset)
    setattr(opt, "save_path", opt_test.save_path)
    setattr(opt, "checkpoint", opt_test.checkpoint)
    setattr(opt, "tile", opt_test.tile)
    setattr(opt, "tile_overlap", opt_test.tile_overlap)
    return opt