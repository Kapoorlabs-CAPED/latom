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
        single_axes: If True (default), create one axes at 111. Set False
            for canvases that manage their own subplot layout.
    """

    def __init__(
        self,
        parent=None,
        width=8,
        height=6,
        dpi=100,
        single_axes=True,
        constrained=False,
    ):
        self.fig = Figure(
            figsize=(width, height), dpi=dpi, constrained_layout=constrained
        )
        if single_axes:
            self.axes = self.fig.add_subplot(111)
        else:
            self.axes = None
        super().__init__(self.fig)
        self.setParent(parent)
        self._constrained = constrained
        if single_axes and not constrained:
            self.fig.tight_layout(pad=3.0)

    def clear_and_draw(self):
        """Redraw the canvas."""
        if not self._constrained:
            try:
                self.fig.tight_layout(pad=3.0)
            except Exception:
                pass
        self.draw()
