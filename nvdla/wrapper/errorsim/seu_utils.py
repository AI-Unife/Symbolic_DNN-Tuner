
import yaml
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config',      type=str, required=True, help='Input layer profiler .yaml file')
    parser.add_argument('--output',      type=str, required=True, help='Output units exposure log')
    parser.add_argument('--inject_data', action='store_true',     help='Dump flatten .yaml tile log')
    parser.add_argument('--inject_ctrl', action='store_true',     help='Dump flatten .yaml tile log')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    config['inject']['datapath'] = False
    config['inject']['ctrlpath'] = False

    if args.inject_data:
        config['inject']['datapath'] = True

    if args.inject_ctrl:
        config['inject']['ctrlpath'] = True

    with open(args.output, 'w') as f:
        yaml.dump(config, f)