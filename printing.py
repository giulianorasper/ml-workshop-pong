from enum import Enum


class LogLevel(Enum):
    INFO = 1
    ANALYSIS = 2
    DEBUG = 3


log_level = LogLevel.INFO


def log(level: LogLevel, message: str):
    if level.value <= log_level.value:
        print(f"[{level.name}] {message}")


def info(message: str):
    log(LogLevel.INFO, message)


def analysis(message: str):
    log(LogLevel.ANALYSIS, message)


def debug(message: str):
    log(LogLevel.DEBUG, message)


def run(level: LogLevel, function, *args):
    if level.value <= log_level.value:
        function(*args)


class SpecialPrint(Enum):
    REWARD = 1


special_prints = []


def special_print(print_type: SpecialPrint, message: str):
    exists = False
    for special_print in special_prints:
        if special_print == print_type:
            exists = True
    if exists:
        print(f"[{special_print.name}] {message}")
