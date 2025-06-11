"""
Module to run pipeline from command line
"""

import argparse
import logging
import os
import sys
import warnings

import yaml

sys.path.append("../../../")


from tools.decoding.lstm.lstm import run_experiment


# Set up logging
def setup_logger(log_file="application.log"):
    # Create a custom logger
    logger = logging.getLogger("my_logger")

    # Set the log level to DEBUG to capture all levels of logs (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.DEBUG)

    # Create handlers (one for console output, one for file)
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_file, mode="w")

    # Set the logging format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Redirect print statements to the logger
class PrintToLogger:
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
        "--config", type=str, required=True, help="Path to the YAML configuration file"
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


def run_experiments(cfg: dict):
    # breakpoint()

    for exp_cfg in cfg["experiments"]:

        # Create results directory
        os.makedirs(exp_cfg["results"]["results_dir"], exist_ok=True)

        # Setup logger
        logger = setup_logger(exp_cfg["logging"]["log_dir"])

        # Capture warnings and redirect them to the logger
        def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
            logger.warning(f"{category.__name__}: {message}")

        warnings.showwarning = custom_warning_handler

        sys.stdout = PrintToLogger(logger)

        logger.info(f"Running experiments: {exp_cfg['name']}, {exp_cfg['description']}")
        run_experiment(exp_cfg)
        # for handler in logger.handlers[:]:
        #     logger.removeHandler(handler)
        #     handler.close()  # Optionally close the handler if needed
        # del logger

    return


def main():
    # Parse arguments
    args = parse_args()

    # Load the configuration file
    config = load_config(args.config)

    run_experiments(config)

    return


if __name__ == "__main__":
    main()
