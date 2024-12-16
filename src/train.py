#from grid_agent.entities import 
from grid_agent.settings import TrainSettings, TrainCommandLineArguments

def main(): 
    train_settings: TrainSettings = TrainSettings(TrainCommandLineArguments())

if __name__ == "__main__":
    main()