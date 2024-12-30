from grid_agent.data_structs.value_functions_container import  ValueFunctionsContainer, ValueFunctionsContainerSequential, ValueFunctionsContainerParallel
from grid_agent.data_structs.valid_state_space import ValidStateSpace, ValidStateSpaceSequential, ValidStateSpaceParallel 
from grid_agent.functors.reward import RewardFunction, DenseRewardFunction, SparseRewardFunction
from grid_agent.data_structs.policy import Policy, PolicySequential, PolicyParallel
from grid_agent.configs.base_configs import BaseConfigs, ConfigArgument

from collections.abc import Callable
from typing import override

class TrainConfigs(BaseConfigs):
    """``BaseConfigs`` specialized for train sessions."""

    def __init__(self) -> None:
        super().__init__()
        self.__reward_factory: ConfigArgument[Callable[[TrainConfigs], RewardFunction]] = ConfigArgument(lambda c: DenseRewardFunction())
        self.__processes_number: ConfigArgument[int] = ConfigArgument(1)
        self.__discount_factor: ConfigArgument[float] = ConfigArgument(0.5)
        self.__use_float: ConfigArgument[bool] = ConfigArgument(False)
        self.__is_dry_run: ConfigArgument[bool] = ConfigArgument(False)
        self.__max_iter: ConfigArgument[int] = ConfigArgument(100)
        self.__value_function_tolerance: ConfigArgument[float] = ConfigArgument(0.0)
        self.__changed_actions_tolerance: ConfigArgument[int] = ConfigArgument(0)
        self.__changed_actions_percentage_tolerance: ConfigArgument[float] = ConfigArgument(0.0)

    @property
    def reward_factory(self) -> Callable[["TrainConfigs"], RewardFunction]:
        return self.__reward_factory.value
    
    @reward_factory.setter
    def reward_factory(self, factory: Callable[["TrainConfigs"], RewardFunction]) -> None:
        self.__reward_factory.set_and_freeze(factory)

    @property
    def processes_number(self) -> int:
        return self.__processes_number.value
    
    @processes_number.setter
    def processes_number(self, number: int) -> None:
        self.__processes_number.set_and_freeze(number)

    @property
    def discount_factor(self) -> float:
        return self.__discount_factor.value

    @discount_factor.setter
    def discount_factor(self, factor: float) -> None:
        self.__discount_factor.set_and_freeze(factor)

    @property
    def use_float(self) -> bool:
        return self.__use_float.value

    @use_float.setter
    def use_float(self, use_float: bool) -> None:
        self.__use_float.set_and_freeze(use_float)

    @property
    def is_dry_run(self) -> bool:
        return self.__is_dry_run.value

    @is_dry_run.setter
    def is_dry_run(self, is_dry_run: bool) -> None:
        self.__is_dry_run.set_and_freeze(is_dry_run)

    @property
    def max_iter(self) -> int:
        return self.__max_iter.value

    @max_iter.setter
    def max_iter(self, value: int) -> None:
        self.__max_iter.set_and_freeze(value)

    @property
    def value_function_tolerance(self) -> float:
        return self.__value_function_tolerance.value

    @value_function_tolerance.setter
    def value_function_tolerance(self, value: float) -> None:
        self.__value_function_tolerance.set_and_freeze(value)

    @property
    def changed_actions_tolerance(self) -> int:
        return self.__changed_actions_tolerance.value

    @changed_actions_tolerance.setter
    def changed_actions_tolerance(self, value: int) -> None:
        self.__changed_actions_tolerance.set_and_freeze(value)

    @property
    def changed_actions_percentage_tolerance(self) -> float:
        return self.__changed_actions_percentage_tolerance.value

    @changed_actions_percentage_tolerance.setter
    def changed_actions_percentage_tolerance(self, value: float) -> None:
        self.__changed_actions_percentage_tolerance.set_and_freeze(value)

    @override
    def _process_line_helper(self, splitted_line: list[str]) -> bool:
        match splitted_line:
            case ["maxiter", max_iter]:
                self.__max_iter.set_if_not_frozen(int(max_iter))
            case ["valuetolerance", value_tol]:
                self.__value_function_tolerance.set_if_not_frozen(float(value_tol))
            case ["actiontolerance", action_tol]:
                self.__changed_actions_tolerance.set_if_not_frozen(int(action_tol))
            case ["actionperctolerance", action_perc_tol]:
                self.__changed_actions_percentage_tolerance.set_if_not_frozen(float(action_perc_tol))
            case ["discount", discount_factor]:
                self.__discount_factor.set_if_not_frozen(float(discount_factor))
            case ["processes", processes_number]:
                self.__processes_number.set_if_not_frozen(int(processes_number))
            case ["usefloat"]:
                self.__use_float.set_if_not_frozen(True)
            case ["usedouble"]:
                self.__use_float.set_if_not_frozen(False)
            case ["densereward"]:
                self.__reward_factory.set_if_not_frozen(lambda c: DenseRewardFunction())
            case ["sparsereward"]:
                self.__reward_factory.set_if_not_frozen(lambda c: SparseRewardFunction())
            case _:
                return False
        return True

    @override
    def _check_helper(self) -> None:
        self.__check_positivity(self.max_iter, "Maximum number of iterations")
        self.__check_positivity(self.processes_number, "Number of processes")
        self.__check_between_zero_and_one(self.discount_factor, "Discount factor")
        self.__check_non_negativity(self.value_function_tolerance, "Value function tolerance")
        self.__check_non_negativity(self.changed_actions_tolerance, "Change actions tolerance")
        self.__check_non_negativity(self.changed_actions_percentage_tolerance, "Changed actions percentage tolerance")

    def __check_between_zero_and_one(self, value: float, name: str) -> None:
        """Check that ``value`` is between zero and one, otherwise raise ``ValueError``."""
        if value < 0 or value > 1:
            raise ValueError(f"{name} should be between 0 and 1\n" +
                             f"It was {value}")

    def __check_positivity(self, value: int | float, name: str) -> None:
        """Check that ``value`` is positive, otherwise raise ``ValueError``."""
        if value <= 0:
            raise ValueError(f"{name} should be > 0.\n" +
                             f"It was {value}")

    def __check_non_negativity(self, value: float | int, name: str) -> None:
        """Check that ``value`` is not negative, otherwise raise ``ValueError``."""
        if value < 0:
            raise ValueError(f"{name} should be >= 0" +
                             f"It was {value}")

    @override
    def _create_helper(self) -> None:
        self.reward: RewardFunction = self.reward_factory(self)
        self.policy: Policy
        self.valid_state_space: ValidStateSpace
        self.value_functions_container: ValueFunctionsContainer
        if self.processes_number == 1:
            self.valid_state_space = ValidStateSpaceSequential(self.map_size, self.obstacles)
            self.policy = PolicySequential.from_action(self.valid_state_space.space_size)
            self.value_functions_container = ValueFunctionsContainerSequential(self.valid_state_space.space_size, use_double=not self.use_float)
        else:
            self.valid_state_space = ValidStateSpaceParallel(self.map_size, self.obstacles)
            self.policy = PolicyParallel.from_action(self.valid_state_space.space_size)
            self.value_functions_container = ValueFunctionsContainerParallel(self.valid_state_space.space_size, use_double=not self.use_float)