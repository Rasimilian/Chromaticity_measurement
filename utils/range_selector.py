from matplotlib.widgets import SpanSelector
import matplotlib.patches as patches


class RangeSelectorGraph:
    def __init__(self, ax, orientation='horizontal', color='green'):
        self.ax = ax
        self.orientation = orientation
        self.color = color
        self.min_val = 0
        self.max_val = 0.5
        self.span_selector = None
        self.rect_patch = None
        self.alpha = 0.2
        self._init_selector()

    def _init_selector(self):
        if self.orientation == 'horizontal':
            self.span_selector = SpanSelector(
                self.ax,
                self.onselect,
                'horizontal',
                useblit=True,
                props=dict(alpha=self.alpha, facecolor=self.color),
                interactive=True,
                drag_from_anywhere=True
            )
        else:
            self.span_selector = SpanSelector(
                self.ax,
                self.onselect,
                'vertical',
                useblit=True,
                props=dict(alpha=self.alpha, facecolor=self.color),
                interactive=True,
                drag_from_anywhere=True
            )

    def onselect(self, vmin, vmax):
        self.min_val = vmin
        self.max_val = vmax
        self.update_visual()
        self.regionChanged()

    def update_visual(self):
        if self.rect_patch:
            self.rect_patch.remove()

        if self.orientation == 'horizontal':
            ymin, ymax = self.ax.get_ylim()
            self.rect_patch = patches.Rectangle(
                (self.min_val, ymin),
                self.max_val - self.min_val,
                ymax - ymin,
                alpha=self.alpha,
                facecolor=self.color,
                edgecolor=self.color,
                linewidth=2
            )
        else:
            xmin, xmax = self.ax.get_xlim()
            self.rect_patch = patches.Rectangle(
                (xmin, self.min_val),
                xmax - xmin,
                self.max_val - self.min_val,
                alpha=self.alpha,
                facecolor=self.color,
                edgecolor=self.color,
                linewidth=2
            )

        self.ax.add_patch(self.rect_patch)
        self.ax.figure.canvas.draw_idle()

    def regionChanged(self):
        pass

    def regionChangeFinished(self):
        pass

    def getRangeCoordinates(self):
        self.min_val, self.max_val = self.span_selector.extents if self.span_selector else (0, 0.5)
        return self.min_val, self.max_val

    def setBounds(self, min_val, max_val):
        if self.span_selector:
            self.span_selector.extents = (min_val, max_val)
            self.min_val = min_val
            self.max_val = max_val
            self.update_visual()

