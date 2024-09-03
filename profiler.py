import os
import time
import argparse
import yaml
import logging
from munch import munchify
import torch
import pandas as pd
from libs.profiler.ModelProfiler import *
from libs.models.ModelFactory import ModelFactory
from libs.pruning.prune_utils import melt



if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Check if the log file already exists and remove it if it does
    log_fn = 'profiler_log.txt'
    if os.path.exists(log_fn):
        os.remove(log_fn)
    # Create a file handler
    file_handler = logging.FileHandler(log_fn)
    file_handler.setLevel(logging.INFO)

    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", default="configs/config-profiler.yaml",
                            help="profile the model")
    arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")

    args = arg_parser.parse_args()

    # check if the config files exists
    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit


    # Munchify the dict to access entries with both dot notation and ['name']
    logging.info(f'Loading the config file...')
    config = yaml.load(open(args.config, "r", encoding='utf-8'), yaml.FullLoader, )
    config = munchify(config)

    # Tell the task
    logging.info('Profiling the model...')
    report = []
    try:

        devices =['cpu', 'cuda']
        for device in devices:
            logging.info(f'Starting the task: {device}')
            model = ModelFactory(config.net_model, config.num_classes, config.in_ch).get().to(device=device)

            for m in model.modules():
                for child in m.children():
                    if type(child) == torch.nn.BatchNorm3d:
                        m.eval()

            state_dict = torch.load(config.torch_model_path)

            if config.is_pruned:
                hard_vector = state_dict['hard_vector']
                melt(model_name=config.net_model, model=model, hard_vector=hard_vector)

            model.load_state_dict(state_dict['state_dict'], strict=True)
            
            
            total_memory, average_time = monitoring(model, (config.batch_size, config.in_ch, *config.input_shape), device=device, repeats=10)
            report.append({'device': device, 'total_memory': total_memory, 'average_time': average_time})
        
        df = pd.DataFrame(report)
        logging.info(df)
        df.to_csv(f'profiling_report_{os.path.basename(config.torch_model_path)}.csv', index=False)
    except Exception as e:
        logging.error(f'Message in the task: {e}')
    




