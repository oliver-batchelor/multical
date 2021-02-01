import logging
import textwrap

from os import path
from sys import stdout
from copy import copy

from structs.struct import struct

logger = logging.getLogger("calibration")

def info(msg, *args, **kwargs):
  return logger.info(msg, *args, **kwargs)

def debug(msg, *args, **kwargs):
  return logger.debug(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
  return logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
  return logger.error(msg, *args, **kwargs)


class MemoryHandler(logging.Handler):
  def __init__(self, level=logging.DEBUG):
    super().__init__(level=level)
    self.records = []
    self.setFormatter(logging.Formatter('%(message)s'))

  def get_records(self):
    return self.records

  def emit(self, record):
    try:
      msg = self.format(record)
      entry = struct(level=record.levelname, time=record.created, message=msg)
      self.records.append(entry)
    except RecursionError:  # See issue 36272
        raise
    except Exception:
        self.handleError(record)

  def __getstate__(self):
    return struct(records = self.records)



class IndentFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super(IndentFormatter, self).__init__(*args, **kwargs)

    def format(self, record):
      message = str(record.msg)
      record = copy(record)
      record.msg = ''

      header = super(IndentFormatter, self).format(record)
      msg = textwrap.indent(message, ' ' * len(header)).strip()
      return header + msg


def setup_logging(log_file, console_level='INFO', handlers=[]):


    stream_handler = logging.StreamHandler(stream=stdout)
    stream_handler.setLevel(getattr(logging, console_level))
    stream_handler.setFormatter(IndentFormatter('%(levelname)s - %(message)s'))

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(IndentFormatter('%(levelname)s - %(message)s'))
    

    handlers = handlers + [stream_handler, file_handler]

    for handler in handlers:
      logger.addHandler(handler)

    logger.setLevel(logging.DEBUG)
    info(f"Logging to {log_file}")

 
