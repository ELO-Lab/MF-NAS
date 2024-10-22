import yaml
import argparse

def run(kwargs):
    f = open(kwargs.config_file, 'r')
    all_configs = yaml.safe_load(f)

    all_configs[kwargs.algo][kwargs.key] = kwargs.value
    f.close()

    f = open(kwargs.config_file, 'w')
    yaml.dump(all_configs, f, default_flow_style=False, sort_keys=False)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--config_file', type=str, default='./configs/algo_201.yaml', help='the configuration file')
    parser.add_argument('--algo', type=str, default='MF-NAS')
    parser.add_argument('--key', type=str, default='metric_stage1')
    parser.add_argument('--value', type=str, default='synflow')

    args = parser.parse_args()

    run(args)