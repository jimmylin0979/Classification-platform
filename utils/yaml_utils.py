from typing import Optional
import argparse
import yaml

from utils import logger

def flatten_dict_to_leaf(yaml: dict, prefix: Optional[str] = "", sep: Optional[str] = "."):
    """
    """

    res = []
    for k, v in yaml.items():
        if isinstance(v, dict):
            # 
            if prefix == "":
                res.extend(flatten_dict_to_leaf(v, f"{k}"))
            else:
                res.extend(flatten_dict_to_leaf(v, f"{prefix}{sep}{k}"))
        else:
            # 
            res.append((f"{prefix}{sep}{k}", v))

    return res


def load_config_file(opts):
    """
    """

    config_file_name = getattr(opts, "config", None)
    # If config file is None, then just return opts
    if config_file_name is None:
        return opts

    # 
    with open(config_file_name, "r") as yaml_file:
        try :
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
            
            # 
            leaves = flatten_dict_to_leaf(cfg)
            for k, v in leaves:
                setattr(opts, k, v)

        except Exception as ex:
            logger.warning(f"Error while loading config file: {config_file_name}")
            logger.warning(f"Error Message: {str(ex)}")
    
    return opts

if __name__ == "__main__":

    # Argparse connect with yaml
    # Constructure argparse insatnce, and get yaml file location from input command
    parser = argparse.ArgumentParser(description='Test yaml_utils.py')
    parser.add_argument('--common.config_file', type=str, required=True,
                                help='The folder to store the training stats of current model')
    opts = parser.parse_args()

    # Before 
    print("=" * 80)
    print(dir(opts))

    # After
    opts = load_config_file(opts)
    print("=" * 80)
    print(dir(opts))

    # Utilize getattr function to access yaml file
    print(getattr(opts, "common.run_label", -1))
