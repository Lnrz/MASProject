from grid_agent.data_structs import TrainData
from typing import Callable
from matplotlib.container import BarContainer
import matplotlib.pyplot as plt
import json

class TrainDataView:

    def __init__(self) -> None:
        self.__mean_values: list[float] = list[float]()

    def get_callback(self) -> Callable[[TrainData], None]:
        return lambda train_data: self.__add_mean_value(train_data.mean_value)

    def __add_mean_value(self, value: float) -> None:
        self.__mean_values.append(value)

    def display(self) -> None:
        bar: BarContainer = plt.bar(x=[i for i in range(len(self.__mean_values))], height=self.__mean_values)
        plt.show()

    def load_from_file(self, file_path: str) -> None:
        with open(file_path, "rt") as f:
            self.__mean_values = json.load(f)

    def write_to_file(self, file_path: str) -> None:
        with open(file_path, "wt") as f:
            json.dump(self.__mean_values, f)