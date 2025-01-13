import logging
import sys
import warnings


class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():  # Ignore empty messages
            self.level(message.strip())

    def flush(self):
        pass


# Create a logger for the package
def set_logging(
    file_name: str,
) -> logging.Logger:
    """
    Set project-wide logging

    Parameters
    ----------
    file_name: str
        Name of the module being logged

    Returns
    -------
    logger: logging.Logger
        logger object
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.captureWarnings(True)

    logger = logging.getLogger(file_name)

    stdout_handler = logging.StreamHandler(sys.stdout)

    # Avoid duplicate handlers
    if not logger.hasHandlers():
        logger.addHandler(stdout_handler)

    def custom_warning_handler(
        message, category, filename, lineno, file=None, line=None
    ):
        logger.warning(f"{category.__name__}: {message}")

    # Set the custom handler
    warnings.showwarning = custom_warning_handler

    # Redirect print to logger automatically
    sys.stdout = LoggerWriter(logger, logger.info)
    # sys.stderr = LoggerWriter(logger, logger.error)

    return logger
