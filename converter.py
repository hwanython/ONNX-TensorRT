import os
import time
import argparse
import yaml
import logging
from munch import munchify
from utils.TaskFactory import TaskFactory



if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Check if the log file already exists and remove it if it does
    log_fn = 'converting_log.txt'
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
    arg_parser.add_argument("-c", "--config", default="configs/config.yaml",
                            help="the preprocessing config file to be used to run the preprocessing")
    arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")

    args = arg_parser.parse_args()

    # check if the config files exists
    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit

    # announcement the goal of this script
    project_task = os.path.basename(__file__).split('.')[0]
    logging.info(f'Project Task: {project_task}')

    # Munchify the dict to access entries with both dot notation and ['name']
    logging.info(f'Loading the config file...')
    config = yaml.load(open(args.config, "r", encoding='utf-8'), yaml.FullLoader, )
    config = munchify(config)

    # Tell the task
    task = config.task
    logging.info(f'Task: {task}')
    try:
        s = time.time()
        logging.info(f'Starting the task: {task}')
        TaskFactory(config).run()
        e = time.time()
        logging.info(f'Finished the task: {task}')
        logging.info(f'Time elapsed: {e-s} seconds')
    except Exception as e:
        logging.error(f'Message in the task: {task}, {e}')
    



