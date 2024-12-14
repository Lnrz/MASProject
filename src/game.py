from grid_agent.entities import GameManager
from grid_agent.settings import GameSettings, CommandLineSettings
from grid_agent.data_structs import Result

def main():
  cmd_line_set: CommandLineSettings = CommandLineSettings()
  cmd_line_set.parse()
  game_set : GameSettings = GameSettings(cmd_line_set)
  game_manager: GameManager = GameManager(game_set)
  res: Result = game_manager.start()
  print(res.name)

if __name__ == "__main__":
    main()