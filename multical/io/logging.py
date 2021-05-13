import logging
import textwrap

from os import path
from sys import stdout
from copy import copy

import numpy as np
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
    return struct(records = self.records, level=self.level)

  def __setstate__(self, state):
    self.__init__(level=state.get('level', logging.DEBUG))
    self.records = state.records


class LogWriter():
  def __init__(self, level=logging.INFO, ignore_newline=True):
    self.level = level
    self.ignore_newline = ignore_newline

  def write(self, message, *args, **kwargs):
    if message != '\n' or not self.ignore_newline:
      logger._log(self.level, message, args, **kwargs)

  @staticmethod
  def info(ignore_newline=True):
    return LogWriter(level=logging.INFO, ignore_newline=ignore_newline)
  
  @staticmethod
  def debug(ignore_newline=True):
    return LogWriter(level=logging.DEBUG, ignore_newline=ignore_newline)

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


def setup_logging(console_level='INFO', handlers=[], log_file=None):
  np.set_printoptions(precision=4, suppress=True)

  for handler in handlers:
    logger.addHandler(handler)

  stream_handler = logging.StreamHandler(stream=stdout)
  stream_handler.setLevel(getattr(logging, console_level))
  stream_handler.setFormatter(IndentFormatter('%(levelname)s - %(message)s'))
  logger.addHandler(stream_handler)    

  if log_file is not None:
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(IndentFormatter('%(levelname)s - %(message)s'))

    info(f"Logging to {log_file}")
    logger.addHandler(file_handler)


  logger.setLevel(logging.DEBUG)
  logger.propagate = False

 
