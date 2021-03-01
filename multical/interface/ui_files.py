from os import path, listdir
from qtpy import uic


ui_path = path.dirname(__file__) or "."

def load_ui(self, filename):
  return uic.loadUi(path.join(ui_path, filename), self)
