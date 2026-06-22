import os
import logging


def getLOGGER(
    name: str = "root",
    terminal=True,
    log_on_file=False,
    save_path="./",
    append=False,
    use_formatter=True,
    console_level=logging.INFO,  # Console shows INFO and above only
    file_level=logging.DEBUG,  # File shows DEBUG and above (includes detailed metrics)
):
    log_level = logging.INFO

    # create logger
    logger = logging.getLogger(name)
    # stop propagting to root logger
    logger.propagate = False
    logger.setLevel(log_level)

    # create formatter
    if use_formatter:
        formatter = logging.Formatter("%(asctime)s - %(message)s")
    else:
        formatter = logging.Formatter()

    # create console handler and set level to INFO (excludes DEBUG messages)
    if terminal:
        terminal_handler = logging.StreamHandler()
        terminal_handler.setLevel(console_level)  # Console only shows INFO and above
        terminal_handler.setFormatter(formatter)
        logger.addHandler(terminal_handler)

    if log_on_file:
        os.makedirs(save_path, exist_ok=True)

        filename = f"{save_path}{name}"
        i = ""
        while os.path.exists(f"{filename}{i}.log"):
            if i == "":
                i = 1
            else:
                i += 1

        # mode = 'a' if append else 'w+'
        file_handler = logging.FileHandler(
            filename=f"{filename}{i}.log", mode="w+", encoding="utf-8"
        )
        file_handler.setLevel(
            file_level
        )  # File shows DEBUG and above (includes detailed metrics)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_training_logger(name: str = "training", save_path: str = "./logs/"):
    """
    Convenience function to get a logger configured for training:
    - Console: Shows INFO and above (normal training messages)
    - File: Shows DEBUG and above (includes detailed JSON metrics)
    """
    return getLOGGER(
        name=name,
        terminal=True,
        log_on_file=True,
        save_path=save_path,
        console_level=logging.INFO,  # Console excludes DEBUG messages
        file_level=logging.DEBUG,  # File includes DEBUG messages
        use_formatter=True,
    )
