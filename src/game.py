from grid_agent.entities import GameManager
from grid_agent.settings import GameSettings, CommandLineSettings
from grid_agent.data_structs import Result

def main():
  game_set : GameSettings = GameSettings(CommandLineSettings())
  game_manager: GameManager = GameManager(game_set)
  res: Result = game_manager.start()
  print(res.name)

if __name__ == "__main__":
    main()