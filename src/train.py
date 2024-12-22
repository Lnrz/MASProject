from argparse import ArgumentParser, Namespace
from grid_agent.settings import TrainConfigs
from grid_agent.entities import TrainManager
from grid_agent.train_view import TrainDataView

def main() -> None:
    arguments: Namespace = get_command_line_arguments()
    train_configuration: TrainConfigs =  get_train_configuration(arguments)
    train_manager: TrainManager = TrainManager(train_configuration)
    train_view: TrainDataView = get_train_view(train_manager)
    train_manager.start()
    train_view.display()
    if arguments.train_data_path is not None:
        train_view.write_to_file(arguments.train_data_path)


def get_command_line_arguments() ->  Namespace:
  parser: ArgumentParser = ArgumentParser(prog="Grid Agent - Train", description="TODO")
  
  parser.add_argument("configs", type=str, help="Path to the configuration file")
  parser.add_argument("-p", "--policy", type=str, help="Path where to save the policy file")
  parser.add_argument("-mi", "--max_iter", type=int, help="Maximum number of iterations")
  parser.add_argument("-vt", "--value_function_tolerance", type=float, help="Value function tolerance")
  parser.add_argument("-cat", "--changed_actions_tolerance", type=int, help="Changed actions tolerance")
  parser.add_argument("-capt", "--changed_actions_percentage_tolerance", type=float, help="Changed actions percentage tolerance")
  parser.add_argument("-tdp", "--train_data_path", type=str, help="Path where to write train data")
  
  return parser.parse_args()


def get_train_configuration(arguments: Namespace) -> TrainConfigs:
  train_configuration: TrainConfigs = TrainConfigs()
  
  train_configuration.configs_file_path = arguments.configs
  train_configuration.policy_file_path = arguments.policy
  train_configuration.max_iter = arguments.max_iter
  train_configuration.value_function_tolerance = arguments.value_function_tolerance
  train_configuration.changed_actions_tolerance = arguments.changed_actions_tolerance
  train_configuration.changed_actions_percentage_tolerance = arguments.changed_actions_percentage_tolerance
  
  return train_configuration


def get_train_view(train_manager: TrainManager) -> TrainDataView:
  train_viewer: TrainDataView = TrainDataView()
  
  train_manager.register_callback(train_viewer.get_callback())
  
  return train_viewer


if __name__ == "__main__":
    main()