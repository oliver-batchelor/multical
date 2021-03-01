from qtpy import QtWidgets


def stretch():
  return QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)


def h_stretch():
  return QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)


def v_stretch():
  return QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)


def widget(layout, parent=None):
  w = QtWidgets.QWidget(parent)
  w.setLayout(layout)
  return w


def layout(create=QtWidgets.QHBoxLayout):
  def f(*widgets, margin=None, spacing=5):
    layout = create()

    def add_item(item):
      if isinstance(item, int):
        layout.addSpacing(item)
      elif isinstance(item, QtWidgets.QLayoutItem):
        layout.addItem(item)
      elif isinstance(item, QtWidgets.QWidget):
        layout.addWidget(item)

    for item in widgets:
      add_item(item)

    if margin is not None:
      layout.setContentsMargins(margin, margin, margin, margin)

    if spacing is not None:
      layout.setSpacing(spacing)

    return layout
  return f

h_layout = layout(QtWidgets.QHBoxLayout)
v_layout = layout(QtWidgets.QVBoxLayout)