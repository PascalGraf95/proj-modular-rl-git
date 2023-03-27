import argparse
from modules.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpf', '--games_per_fixture', help="Number of matches between each fixture in the tournament schedule.", type=int, default=10, required=False)
    parser.add_argument('-gr', '--game_results_path', help="Path to the history csv file containing all played matches used to update player ratings.", type=str, default="./tournament/game_results.csv", required=False)
    parser.add_argument('-rp', '--rating_history_path', help="Path to the history csv file containing all player ratings.", type=str, default="./tournament/rating_history.csv", required=False)
    parser.add_argument('-tt', '--tournament_type', help="Type of tournament to be played.", type=str, default="round_robin", required=False)
    parser.add_argument('-mp', '--model_path', help="Path to the directory containing the trained models.", type=str, default="./training/models", required=False)
    parser.add_argument('-cp', '--clone_path', help="Path to the directory containing the cloned models.", type=str, default="./training/clones", required=False)
    parser.add_argument('-ep', '--environment_path', help="Path to the environment file.", type=str, default="./environments/AirHockey.exe", required=False)
    args = parser.parse_args()

    # Instantiate the trainer
    trainer = Trainer()
    trainer.select_training_algorithm('SAC')
    # - Self-Play Tournament -
    trainer.games_per_fixture = args.games_per_fixture
    trainer.game_results_path = args.game_results_path
    trainer.rating_history_path = args.rating_history_path

    # Given a path with one or more trained models the following function creates a dictionary that contains a unique
    # key for each model along with the corresponding model paths, the number of steps it's been trained for, the reward
    # it reached and possibly a (elo) rating. The same is done for a clone of the original model in case of self-play.
    trainer.create_model_dictionaries(model_path=args.model_path, clone_path=args.clone_path)
    # With the model dictionaries create a tournament schedule can be created where each model plays against every
    # other model. This is only necessary for self-play environments in tournament mode.
    trainer.create_tournament_schedule(tournament_type=args.tournament_type)

    # Instantiate the agent(s)
    trainer.async_instantiate_agent(mode='tournament', preprocessing_algorithm='None', exploration_algorithm='None', environment_path=args.environment_path)

    # The tournament loop plays all matches in the respective schedule and writes their results into a csv file.
    trainer.async_tournament_loop()
    

if __name__ == '__main__':
    main()