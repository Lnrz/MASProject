from grid_agent.entities import TrainManager
from grid_agent.settings import TrainSettings, TrainCommandLineArguments

def main(): 
    train_manager: TrainManager = TrainManager(TrainSettings(TrainCommandLineArguments()))
    train_manager.start()

if __name__ == "__main__":
    main()