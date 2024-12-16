from grid_agent.entities import GameManager
from grid_agent.settings import GameSettings, GameCommandLineArguments
from grid_agent.data_structs import Result

def main(): 
  game_manager: GameManager = GameManager(GameSettings(GameCommandLineArguments()))
  res: Result = game_manager.start()
  print(res.name)

if __name__ == "__main__":
    main()