import time

import numpy as np
from jupyterplot import ProgressPlot

class TrainPlotter:
    def __init__(self, moving_average_size, labels) -> None:
        self.moving_average_size = moving_average_size
        self.labels = labels + ["Epoch Time"]
        self.progress_plot = ProgressPlot(plot_names = self.labels,line_names=["value", "MA"])
        self.num_labels = len(self.labels)
        self.data = dict(zip(self.labels,[{"plot":[],"batch":[]} for l in self.labels]))
    
    def start_epoch(self):
        self.epoch_start = time.time()
        for label in self.labels:
            self.data[label]["batch"] = []
                
    def batch_update(self,update_arr):
        assert len(update_arr) == len(self.labels)-1, "values must have same length as labels"
        for i,val in enumerate(update_arr):
            self.data[self.labels[i]]["batch"].append(val)

    def log_epoch(self):
        epoch_time = time.time()-self.epoch_start
        self.data["Epoch Time"]["batch"].append(epoch_time)
        output = []
        for label in self.labels:
            batch_mean = np.mean(self.data[label]["batch"])
            self.data[label]["plot"].append(batch_mean)
            plot_ma = np.mean(self.data[label]["plot"][-self.moving_average_size:])
            output.append([batch_mean,plot_ma])
        self.progress_plot.update(output)
