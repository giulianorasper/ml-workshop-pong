from enum import Enum


class LogLevel(Enum):
    INFO = 1
    WARN = 2
    ANALYSIS = 3
    DEBUG = 4


log_level = LogLevel.ANALYSIS


def log(level: LogLevel, message: str):
    if level.value <= log_level.value:
        print(f"[{level.name}] {message}")


def info(message: str):
    log(LogLevel.INFO, message)


def warn(message: str):
    log(LogLevel.WARN, message)


def analysis(message: str):
    log(LogLevel.ANALYSIS, message)


def debug(message: str):
    log(LogLevel.DEBUG, message)


def run(level: LogLevel, function, *args):
    if level.value <= log_level.value:
        function(*args)


class PrintFlag(Enum):
    REWARD = 1


print_flags = []


def special_print(print_type: PrintFlag, message: str):
    exists = False
    for special_print in print_flags:
        if special_print == print_type:
            exists = True
    if exists:
        print(f"[{special_print.name}] {message}")
