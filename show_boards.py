
from multical import board
from multical.image.display import display_stacked

import argparse

def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument('--boards', required=True, help='configuration file (YAML) for calibration boards')
    args = parser.parse_args()
    
    board_file = args.boards
    boards = board.load_config(board_file)
    
    print("Using boards:")
    for name, b in boards.items():
      print(f"{name} {b}")

    images = [board.draw(100) for board in boards.values()]
    display_stacked(images)

if __name__ == '__main__':
    main()
