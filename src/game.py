from argparse import ArgumentParser, Namespace
from grid_agent.data_structs import Vec2D
from grid_agent.configuration import GameConfigs
from grid_agent.entities import GameManager
from grid_agent.ascii_view import ASCIIView
import re

def main() -> None:
  arguments: Namespace = get_command_line_arguments()
  game_configuration: GameConfigs = get_game_configuration(arguments)
  game_manager: GameManager = GameManager(game_configuration)
  game_viewer: ASCIIView = get_game_view(game_manager, game_configuration)
  game_manager.start()
  if arguments.manual:
    game_viewer.start_manual()
  else:
    game_viewer.start_auto(arguments.time_step)


def get_command_line_arguments() ->  Namespace:
  parser: ArgumentParser = ArgumentParser(prog="Grid Agent - Game", description="TODO")
  
  parser.add_argument("configs", type=str, help="Path to the configuration file")
  parser.add_argument("-p", "--policy", type=str, help="Path to the policy file")
  parser.add_argument("-as", "--agent_start", type=str, help="Agent start position (x,y)")
  parser.add_argument("-ts", "--target_start", type=str, help="Target start position (x,y)")
  parser.add_argument("-os", "--opponent_start", type=str, help="Opponent start position (x,y)")
  parser.add_argument("-t", "--time_step", type=float, default=1.5, help="How many seconds between each game step")
  parser.add_argument("-m", "--manual", action="store_true", help="Activate manual view")
  
  return parser.parse_args()


def string_to_vec2D(string: str) -> Vec2D:
  match: re.Match[str] | None = re.search(r"\((\d+),(\d+)\)", string)
  if match is None:
    raise ValueError(f"A str to convert to Vec2D was ill-formed.\n" + 
                     f"It should have been (x,y) but it was {string}")
  return Vec2D(int(match[1]), int(match[2]))

def get_game_configuration(arguments: Namespace) -> GameConfigs:
  game_configuration: GameConfigs = GameConfigs()
  
  game_configuration.configs_file_path = arguments.configs
  if arguments.policy is not None:
    game_configuration.policy_file_path = arguments.policy
  if arguments.agent_start is not None:
    game_configuration.agent_start = string_to_vec2D(arguments.agent_start)
  if arguments.target_start is not None:
    game_configuration.target_start = string_to_vec2D(arguments.target_start)
  if arguments.opponent_start is not None:
    game_configuration.opponent_start = string_to_vec2D(arguments.opponent_start)
  
  return game_configuration


def get_game_view(game_manager: GameManager, game_configuration: GameConfigs) -> ASCIIView:
  game_viewer: ASCIIView = ASCIIView(game_configuration.map_size, game_configuration.obstacles)
  
  game_manager.register_callback(game_viewer.get_callback())
  
  return game_viewer


if __name__ == "__main__":
    main()