
from multical.display import show_detections
from multical import board
from multical.image.display import display, display_stacked

import argparse
from os import path

from multical.image.detect import load_image

def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument('boards',  help='configuration file (YAML) for calibration boards')
    parser.add_argument('--detect', default=None,  help='show detections from an image')

    args = parser.parse_args()
    
    board_file = args.boards
    boards = board.load_config(board_file)
    
    print("Using boards:")
    for name, b in boards.items():
      print(f"{name} {b}")

    if args.detect is not None:
      image = load_image(args.detect)
      detections = [board.detect(image) for board in boards.values()]

      for k, d in zip(boards.keys(), detections):
        print(f"Board {k}: detected {d.ids.size} points")

      image = show_detections(image, detections, radius=10)
      display(image)

    else:
      images = [board.draw(100) for board in boards.values()]
      display_stacked(images)

if __name__ == '__main__':
    main()
