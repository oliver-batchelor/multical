from . import arguments

from .calibrate import calibrate
from .check_boards import show_boards
from .show_result import show_result


def main():
  args = arguments.parse_arguments()
