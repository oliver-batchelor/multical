
import pathlib

import cv2
import numpy as np
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

  image_size = None
  if args.image_size is not None:
    image_size = [int(x) for x in args.image_size.split('x')]
    assert len(image_size) == 2, "expected WxH image_size e.g. 800x600"


  def draw_board(board):
    board_image = board.draw(args.square_length, args.margin)

    if image_size is not None:
      image = np.full((image_size[1], image_size[0]), 255, dtype=np.uint8)
      dy, dx = [(a - b) // 2  for a, b in zip(image.shape, board_image.shape)]

      assert dx >= 0 and dy >= 0,\
        f"--image_size ({args.image_size}) must be larger than board image ({board_image.shape[1]}x{board_image.shape[0]})"

      image[dy:dy+board_image.shape[0], dx:dx + board_image.shape[1]] = board_image
      return image

    return board_image    

  images = {k:draw_board(board) for k, board in boards.items()}

  if args.detect is not None:
    image = load_image(args.detect)
    detections = [board.detect(image) for board in boards.values()]

    for k, d in zip(boards.keys(), detections):
      print(f"Board {k}: detected {d.ids.size} points")

    image = show_detections(image, detections, radius=10)
    display(image)

  elif args.write is not None:
    pathlib.Path(args.write).mkdir(parents=True, exist_ok=True)
    for k, board_image in images.items():
      filename = path.join(args.write, k + ".png")
      cv2.imwrite(filename, board_image)
      print(f"Wrote {filename}")
  else:
    display_stacked(list(images.values()))



def main(): 
    args = parse_with(add_boards_args)
    show_boards(args)

if __name__ == '__main__':
    main()
