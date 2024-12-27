from matplotlib.ticker import PercentFormatter, MaxNLocator
from matplotlib.figure import Figure
from matplotlib.artist import Artist
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from grid_agent.data_structs import TrainData
from typing import Callable, Any
import json

class TrainDataView:

    def __init__(self) -> None:
        self.__iteration_indices: list[int] = []
        self.__mean_values: list[float] = []
        self.__changed_actions: list[int] = []
        self.__changed_actions_percentages: list[float] = []

    def get_callback(self) -> Callable[[TrainData], None]:
        return lambda train_data: self.__add_traindata(train_data)

    def __add_traindata(self, traindata: TrainData) -> None:
        self.__iteration_indices.append(traindata.iteration_number)
        self.__mean_values.append(traindata.mean_value)
        self.__changed_actions.append(traindata.changed_actions_number)
        self.__changed_actions_percentages.append(traindata.changed_actions_percentage)

    def display(self) -> None:
        plt.style.use("dark_background")
        fig: Figure
        mean_values_axes: Axes
        changed_actions_axes: Axes
        fig, (mean_values_axes, changed_actions_axes) = plt.subplots(1, 2, figsize=(9, 4.5))
        fig.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.8, wspace=0.4, hspace=0.06)
        fig.suptitle("Training Info", y=0.95, fontweight="bold", fontsize="x-large")

        label_size: float = 11

        mean_values_axes.plot(self.__iteration_indices, self.__mean_values, label="Mean value", color="b")
        mean_values_axes.set_xlabel("Iteration", size=label_size)
        mean_values_axes.set_ylabel("Mean value", size=label_size)
        mean_values_axes.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
        mean_values_axes.legend()

        changed_actions_axes.bar(self.__iteration_indices, self.__changed_actions, label="Changed actions", fill=True, log=True, color=("g", 0.6))
        changed_actions_axes.set_xlabel("Iteration", size=label_size)
        changed_actions_axes.set_ylabel("Changed actions", size=label_size)
        changed_actions_axes.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
        changed_actions_percentage_axes: Axes = changed_actions_axes.twinx()
        changed_actions_percentage_axes.plot(self.__iteration_indices, self.__changed_actions_percentages, label="Changed actions percentage", color="r")
        changed_actions_percentage_axes.set_ylabel("Percentage of changed actions", size=label_size)
        changed_actions_percentage_axes.set_ylim(bottom=0.0, top=1.0)
        changed_actions_percentage_axes.yaxis.set_major_formatter(PercentFormatter(1.0))
        handle1: list[Artist]
        legend1: list[Any]
        handle2: list[Artist]
        legend2: list[Any]
        handle1, legend1 = changed_actions_axes.get_legend_handles_labels()
        handle2, legend2 = changed_actions_percentage_axes.get_legend_handles_labels()
        changed_actions_axes.legend(handle1 + handle2, legend1 + legend2)
        
        plt.show()

    def read_from_file(self, file_path: str) -> None:
        with open(file_path, "rt") as f:
            data: Any = json.load(f)
            if ("iteration_indices" not in data or
                "mean_values" not in data or
                "changed_actions" not in data or
                "changed_actions_percentages" not in data):
                raise ValueError("TrainView: the file to read did not have all the necessary data.")
            self.__iteration_indices = data["iteration_indices"]
            self.__mean_values = data["mean_values"]
            self.__changed_actions = data["changed_actions"]
            self.__changed_actions_percentages = data["changed_actions_percentages"]

    def write_to_file(self, file_path: str) -> None:
        with open(file_path, "wt") as f:
            json.dump({ "iteration_indices" : self.__iteration_indices,
                        "mean_values" : self.__mean_values,
                        "changed_actions" : self.__changed_actions,
                        "changed_actions_percentages" : self.__changed_actions_percentages},
                        f)