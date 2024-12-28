from grid_agent.entities.train_manager import TrainManager
from grid_agent.configs.train_configs import TrainConfigs

from argparse import ArgumentParser, Namespace

def main() -> None:
    arguments: Namespace = get_command_line_arguments()
    train_configuration: TrainConfigs =  get_train_configuration(arguments)
    train_manager: TrainManager = TrainManager(train_configuration)
    train_manager.start()


def get_command_line_arguments() ->  Namespace:
  parser: ArgumentParser = ArgumentParser(prog="Grid Agent - Train", description="TODO")
  
  parser.add_argument("configs", type=str, help="Path to the configuration file")
  parser.add_argument("-p", "--policy", type=str, help="Path where to save the policy file")
  parser.add_argument("-proc", "--processes", type=int, default=0, help="The number of processes that will be executed")
  parser.add_argument("-f", "--use_float", action="store_true", help="Use float instead of double to store value function values")
  parser.add_argument("-d", "--dry_run", action="store_true", help="Do not write policy to disk")
  parser.add_argument("-mi", "--max_iter", type=int, help="Maximum number of iterations")
  parser.add_argument("-vt", "--value_function_tolerance", type=float, help="Value function tolerance")
  parser.add_argument("-cat", "--changed_actions_tolerance", type=int, help="Changed actions tolerance")
  parser.add_argument("-capt", "--changed_actions_percentage_tolerance", type=float, help="Changed actions percentage tolerance")
  
  return parser.parse_args()


def get_train_configuration(arguments: Namespace) -> TrainConfigs:
  train_configuration: TrainConfigs = TrainConfigs()
  
  train_configuration.configs_file_path = arguments.configs
  if arguments.policy is not None:
    train_configuration.policy_file_path = arguments.policy
  if arguments.max_iter is not None:
    train_configuration.max_iter = arguments.max_iter
  if arguments.value_function_tolerance is not None:
    train_configuration.value_function_tolerance = arguments.value_function_tolerance
  if arguments.changed_actions_tolerance is not None:
    train_configuration.changed_actions_tolerance = arguments.changed_actions_tolerance
  if arguments.changed_actions_percentage_tolerance is not None:
    train_configuration.changed_actions_percentage_tolerance = arguments.changed_actions_percentage_tolerance
  if arguments.processes > 0:
    train_configuration.processes_number = arguments.processes
  train_configuration.use_float = arguments.use_float
  train_configuration.is_dry_run = arguments.dry_run
  
  return train_configuration


if __name__ == "__main__":
    main()