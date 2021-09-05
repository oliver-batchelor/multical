
import pathlib
from typing import Optional

import cv2
import numpy as np
from simple_parsing import choice

from multical.display import show_detections
from multical import board
from multical.image.display import display, display_stacked

import argparse
from os import path

from multical.image.detect import load_image
from multical.config import *


standard_sizes = dict(
  A4 = (210, 297),
  A3 = (297, 420),
  A2 = (420, 594),
  A1 = (594, 841),
  A0 = (841, 1189)
)


@dataclass 
class Boards:
  """ Generate boards and show/detect for configuration file """

  boards : str # Configuration file (YAML) for calibration boards
  
  detect : Optional[str] = None # Show detections from an example image
  write : Optional[str] = None # Directory to write board images

  pixels_mm : int = 1   # Pixels per mm of pattern
  margin_mm : int = 20  # Border width in mm

  paper_size_mm : Optional[str] = None # Paper size in mm WxH 
  paper_size : Optional[str] = choice(*standard_sizes.keys(), default=None)

  def execute(self):
    show_boards(self)



def show_boards(args):
  boards = board.load_config(args.boards)

  print("Using boards:")
  for name, b in boards.items():
    print(f"{name} {b}")

  assert args.paper_size_mm is None or args.paper_size is None, "specify --paper_size_mm or --paper_size (not both)"

  paper_size_mm = None
  if args.paper_size is not None:
    paper_size_mm = standard_sizes[args.paper_size]

  elif args.paper_size_mm is not None:
    paper_size_mm = [int(x) for x in args.paper_size_mm.split('x')]   
    assert len(paper_size_mm) == 2, f"expected WxH paper_size_mm e.g. 420x594 or name, one of {list(standard_sizes.keys())}"

  if paper_size_mm is not None:
    args.margin = 0

  def draw_board(board):
    board_image = board.draw(args.pixels_mm, args.margin_mm)
    board_size = board.size_mm

    if paper_size_mm is not None:
      w, h = paper_size_mm[0] * args.pixels_mm, paper_size_mm[1] * args.pixels_mm

      image = np.full((h, w), 255, dtype=np.uint8)
      dy, dx = [(a - b) // 2  for a, b in zip(image.shape, board_image.shape)]

      assert dx >= 0 and dy >= 0,\
        f"--paper_size ({paper_size_mm[0]}x{paper_size_mm[1]}mm) must be larger than board size ({board_size[0]}x{board_size[1]}mm)"

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




if __name__ == '__main__':
    run_with(Boards)
