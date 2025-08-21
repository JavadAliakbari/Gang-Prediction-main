import os
import logging


def getLOGGER(
    name: str = "root",
    terminal=True,
    log_on_file=False,
    save_path="./",
    append=False,
    use_formatter=True,
):
    log_level = logging.DEBUG

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

    # create console handler and set level to debug
    if terminal:
        terminal_handler = logging.StreamHandler()
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
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
