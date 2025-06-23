"""
Module to run pipeline from command line
"""

import argparse
import logging
import shutil
import sys
import warnings
from pathlib import Path

import yaml

sys.path.append("../../../")

from tools.decoding.lstm import run_lstm_experiment


def setup_logger():
    """
    Setup logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers:  # Clear existing handlers
        logger.removeHandler(handler)

    # Create a new stream handler for the console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)
    return logger


class PrintToLogger:
    """
    Class to catch prints and log them
    """

    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        # Ignore empty messages
        if message.strip():
            self.logger.info(message.strip())

    def flush(self):
        pass  # No operation for flush in this case


def parse_args():
    """
    Parse command-line arguments to accept the path to the configuration file.
    """
    parser = argparse.ArgumentParser(
        description="Run LSTM experiments using the provided configuration file."
    )

    # Adding the argument for the config file path
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to the YAML configuration file"
    )

    parser.add_argument(
        "--exps",
        type=str,
        nargs="+",
        help="List of experiments (can be a single item or multiple items)",
    )

    # Parse arguments
    args = parser.parse_args()
    return args


def load_config(config_path):
    """
    Load the configuration file (YAML format).
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_experiments(exps: dict, exp_names: str | None, logger: logging.Logger):
    """Main experiment running routing

    Args:
        exps (dict): general yaml config
        exp_names (str | None): specific experiment to run
        logger (logging.Logger): logger
    """

    if exp_names is not None:
        exps = {key: exps[key] for key in exp_names if key in exps}
        if not exps:
            print(f"Experiments: {exp_names} dont match any experiment in provided config")

    # if
    for exp_cfg in exps.values():

        # Create results directory
        folder_path = Path(exp_cfg["results"]["results_dir"])

        if folder_path.exists():
            shutil.rmtree(folder_path)

        folder_path.mkdir(parents=True, exist_ok=True)

        # Remove existing file handler (if any) and add a new one for the current log file
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

        file_handler = logging.FileHandler(
            f'{exp_cfg["results"]["results_dir"]}/{exp_cfg['name']}.log', mode="w"
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        # Capture warnings and redirect them to the logger
        def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
            logger.warning(f"{category.__name__}: {message}")

        warnings.showwarning = custom_warning_handler

        # Logger printing
        sys.stdout = PrintToLogger(logger)

        logger.info(f"Running experiments: {exp_cfg['name']}, {exp_cfg['description']}")
        run_lstm_experiment(exp_cfg)

    return


def main():
    # Parse arguments
    args = parse_args()

    # Load the configuration file
    config = load_config(args.cfg)

    # Set up the logger
    logger = setup_logger()

    run_experiments(config, args.exps, logger)

    return


if __name__ == "__main__":
    main()
