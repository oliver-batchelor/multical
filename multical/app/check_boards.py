
from multical.app.arguments import add_boards_args, parse_with
from multical.display import show_detections
from multical import board
from multical.image.display import display, display_stacked

import argparse
from os import path

from multical.image.detect import load_image

def show_boards(args):
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



def main(): 
    args = parse_with(add_boards_args)
    show_boards(args)

if __name__ == '__main__':
    main()
