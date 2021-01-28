import pprint as pp
import numpy as np


class FormatPrinter(pp.PrettyPrinter):

  def __init__(self, formats={}, ):
    super(FormatPrinter, self).__init__()
    self.formats = formats

  def format(self, obj, ctx, maxlvl, lvl):
    for t, format in self.formats:
      if isinstance(obj, t):
        return format.format(obj), 1, 0
    return pp.PrettyPrinter.format(self, obj, ctx, maxlvl, lvl)


formatter = FormatPrinter(
  formats=[
    ((np.floating, float), "{:.4f}")
  ]) 


def pprint(x, *args, **kwargs):
  return formatter.pprint(x, *args, **kwargs)


def pformat(x, *args, **kwargs):
  return formatter.pformat(x, *args, **kwargs)
