from argparse import ArgumentParser, Namespace
from grid_agent.train_view import TrainDataView

def main() -> None:
    arguments: Namespace = get_command_line_arguments()
    traindata_view: TrainDataView  = TrainDataView()
    traindata_view.read_from_file(arguments.train_data)
    traindata_view.display()


def get_command_line_arguments() ->  Namespace:
  parser: ArgumentParser = ArgumentParser(prog="Grid Agent - View Train Data", description="TODO")
  
  parser.add_argument("train_data", type=str, help="Path to the train data")
  
  return parser.parse_args()


if __name__ == "__main__":
    main()