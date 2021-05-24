from os import path
from .charuco import CharucoBoard
from multical.io.logging import error


def read_pairs(filename):
  values = []
  with open(filename, 'rt') as file:
    for line in file:
      if len(line.strip()) > 0:
        line = [item.strip() for item in line.split()]
        assert len(line) == 2, f"expected form: key value on line {line}"

        values.append(line)
  return values


def take_keys(pairs, keys, dtype=int):
  values = []
  for expected in keys:
    k, v = pairs.pop(0)
    if k != expected:
      raise SyntaxError(f"expected {expected}, got {k}")
    values.append(dtype(v))

  return values


def load_mm_file(network_file, i):
  pairs = read_pairs(path.join(path.dirname(
      network_file), f"pattern_square_mm{i}.txt"))
  square_length, = take_keys(pairs, ["squareLength_mm"], dtype=float)
  return square_length


def load_calico(network_file):
  pairs = read_pairs(network_file)
  boards = {}
  offset = 0

  try:
    dict_id, number_boards = take_keys(pairs, ["aruco_dict", "number_boards"])
    for i in range(number_boards):

      fields = ["squaresX", "squaresY", "squareLength", "markerLength"]
      w, h, square_length_px, marker_length_px = take_keys(pairs, fields)
      square_length = load_mm_file(network_file, i)
      marker_length = square_length * (marker_length_px / square_length_px)

      board = CharucoBoard((w, h), square_length, marker_length,
                           aruco_dict=dict_id, aruco_offset=offset)

      boards[f"board{i}"] = board
      offset += len(board.board.ids)

  except (SyntaxError, IOError) as e:
    error(f"Failed to load calico network file {network_file}")
    error(e)

  return boards
