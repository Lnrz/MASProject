from grid_agent.entities import TrainManager
from grid_agent.settings import TrainSettings, TrainCommandLineArguments
from grid_agent.train_view import TrainDataView

def main() -> None:
    train_settings: TrainSettings = TrainSettings(TrainCommandLineArguments())
    train_manager: TrainManager = TrainManager(train_settings)
    train_view: TrainDataView = TrainDataView()
    train_manager.register_callback(train_view.get_callback())
    train_manager.start()
    train_view.display()
    train_view.write_to_file(r"..\train_logs\train.json")

if __name__ == "__main__":
    main()