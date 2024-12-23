from grid_agent.base_configs import BaseConfigs, ConfigArgument, RewardFunction, SimpleRewardFunction, Callable

class TrainConfigs(BaseConfigs):

    def __init__(self) -> None:
        super().__init__()
        self._reward_factory: ConfigArgument[Callable[[TrainConfigs], RewardFunction]] = ConfigArgument(lambda c: SimpleRewardFunction())
        self._max_iter: ConfigArgument[int] = ConfigArgument(100)
        self._value_function_tolerance: ConfigArgument[float] = ConfigArgument(0.0)
        self._changed_actions_tolerance: ConfigArgument[int] = ConfigArgument(0)
        self._changed_actions_percentage_tolerance: ConfigArgument[float] = ConfigArgument(0.0)

    @property
    def reward_factory(self) -> Callable[["TrainConfigs"], RewardFunction]:
        return self._reward_factory.value
    
    @reward_factory.setter
    def reward_factory(self, factory: Callable[["TrainConfigs"], RewardFunction]) -> None:
        self._reward_factory.value = factory
        self._reward_factory.frozen = True

    @property
    def max_iter(self) -> int:
        return self._max_iter.value

    @max_iter.setter
    def max_iter(self, value: int) -> None:
        self._max_iter.value = value
        self._max_iter.frozen = True

    @property
    def value_function_tolerance(self) -> float:
        return self._value_function_tolerance.value

    @value_function_tolerance.setter
    def value_function_tolerance(self, value: float) -> None:
        self._value_function_tolerance.value = value
        self._value_function_tolerance.frozen = True

    @property
    def changed_actions_tolerance(self) -> int:
        return self._changed_actions_tolerance.value

    @changed_actions_tolerance.setter
    def changed_actions_tolerance(self, value: int) -> None:
        self._changed_actions_tolerance.value = value
        self._changed_actions_tolerance.frozen = True

    @property
    def changed_actions_percentage_tolerance(self) -> float:
        return self._changed_actions_percentage_tolerance.value

    @changed_actions_percentage_tolerance.setter
    def changed_actions_percentage_tolerance(self, value: float) -> None:
        self._changed_actions_percentage_tolerance.value = value
        self._changed_actions_percentage_tolerance.frozen = True

    def _process_line_helper(self, splitted_line: list[str]) -> bool:
        match splitted_line:
            case ["maxiter", max_iter]:
                if not self._max_iter.frozen:
                    self._max_iter.value = int(max_iter)
            case ["valuetolerance", value_tol]:
                if not self._value_function_tolerance.frozen:
                    self._value_function_tolerance.value = float(value_tol)
            case ["actiontolerance", action_tol]:
                if not  self._changed_actions_tolerance.frozen:
                    self._changed_actions_tolerance.value = int(action_tol)
            case ["actionperctolerance", action_perc_tol]:
                if not self._changed_actions_percentage_tolerance.frozen:
                    self._changed_actions_percentage_tolerance.value = float(action_perc_tol)
            case _:
                return False
        return True

    def _check_helper(self) -> None:
        self.__check_positivity(self.max_iter, "Maximum number of iterations")
        self.__check_non_negativity(self.value_function_tolerance, "Value function tolerance")
        self.__check_non_negativity(self.changed_actions_tolerance, "Change actions tolerance")
        self.__check_non_negativity(self.changed_actions_percentage_tolerance, "Changed actions percentage tolerance")

    def __check_positivity(self, value: int | float, name: str) -> None:
        if value <= 0:
            raise ValueError(f"{name} should be > 0.\n" +
                             f"It was {value}")

    def __check_non_negativity(self, value: float | int, name: str) -> None:
        if value < 0:
            raise ValueError(f"{name} should be >= 0" +
                             f"It was {value}")

    def _create_helper(self) -> None:
        self.reward: RewardFunction = self.reward_factory(self)