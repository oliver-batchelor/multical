from . import arguments

from .calibrate import calibrate
from .check_boards import show_boards
from .show_result import show_result


def main():
  args = arguments.parse_arguments()

  if args.which == "calibrate":
    calibrate(args)
  elif args.which == "show_boards":
    show_boards(args)
  elif args.which == "show_result":
    show_result(args)
  else:
    assert False, f"unknown command {args.which}"



if __name__ == "__main__":
  main()