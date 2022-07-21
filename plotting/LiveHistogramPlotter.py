import functools
from matplotlib import pyplot as plt

def plotlive(func):
    plt.ion()

    @functools.wraps(func)
    def new_func(*args, **kwargs):

        # Clear all axes in the current figure.
        axes = plt.gcf().get_axes()
        for axis in axes:
            axis.cla()

        # Call func to plot something
        result = func(*args, **kwargs)

        # Draw the plot
        plt.draw()
        plt.pause(0.01)

        return result

    return new_func 
    
class LiveHistogramPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1,1)

    @plotlive
    def update(self, new_hist):
        return self.ax.hist(new_hist, density=True)