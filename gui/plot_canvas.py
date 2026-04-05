"""Matplotlib canvas widget for embedding plots in PyQt5."""

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    """A matplotlib canvas that can be embedded in a PyQt5 widget.

    Args:
        parent: Parent QWidget.
        width: Figure width in inches.
        height: Figure height in inches.
        dpi: Figure resolution.
    """

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout(pad=3.0)

    def clear_and_draw(self):
        """Clear the axes and redraw the canvas."""
        self.fig.tight_layout(pad=3.0)
        self.draw()
