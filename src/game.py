from grid_agent.entities import GameManager
from grid_agent.settings import GameSettings, GameCommandLineArguments
from grid_agent.ascii_view import ASCIIView

def main() -> None:
  game_settings: GameSettings = GameSettings(GameCommandLineArguments())
  game_manager: GameManager = GameManager(game_settings)
  viewer: ASCIIView = ASCIIView(game_settings.map_size, game_settings.obstacles)
  game_manager.register_callback(viewer.get_callback())
  game_manager.start()
  viewer.start_manual()

if __name__ == "__main__":
    main()