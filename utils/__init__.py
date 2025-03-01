from datetime import datetime


def get_datetime_str() -> str:

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
