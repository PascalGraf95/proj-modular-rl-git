import os
import pandas as pd
import shutil
import csv
import math
from misc.elo import Elo
from misc.glicko2 import Glicko2
import unittest

class TestElo(unittest.TestCase):
    def test_elo_calculation(self):
        # test 
        new_rating = Elo.calculate_standard_elo(1500, 1500, 1, 32)
        self.assertEqual(new_rating, 1516)
        new_rating = Elo.calculate_standard_elo(1500, 1500, 0, 32)
        self.assertEqual(new_rating, 1484)
        new_rating = Elo.calculate_standard_elo(1500, 1500, 0.5, 32)
        self.assertEqual(new_rating, 1500)

class HelperFunctions():
    def __init__(self):
        pass

    def get_rating_history(self, path, player_results):
        # rating histories dictionary
        rating_histories = {}        
        # get each directory in the tourney_results directory
        for agent in os.scandir(path):
            # check if it is a directory
            if agent.is_dir():
                # get path to rating history
                rating_history_path = os.path.join(agent.path, player_results)
                # check if csv exists
                if os.path.exists(rating_history_path):
                    # get rating history to pd.DataFrame
                    rating_history = pd.read_csv(rating_history_path)
                    # add rating history to dictionary
                    rating_histories[agent.name] = rating_history
        return rating_histories

    def truncate(f, n):
        #Truncates/pads a float f to n decimal places without rounding
        return math.floor(f * 10 ** n) / 10 ** n


class TestGlicko2(unittest.TestCase):
    test_tourney_path = 'modules/tests/rating_test'
    test_tourney_results_path = 'modules/tests/rating_test_results'
    game_results = 'game_results.csv'
    player_results_file = 'rating_history.csv'

    def test_glicko2_calculation(self):
        # read http://www.glicko.net/glicko/glicko2.pdf for more information about the calculations
        # delete test tourney results directory if it exists
        if os.path.exists(self.test_tourney_results_path):
            shutil.rmtree(self.test_tourney_results_path)
        # rating histories dictionary
        rating_histories = HelperFunctions.get_rating_history(self, self.test_tourney_path, self.player_results_file)        
        # get game results to pd.DataFrame
        tourney_results = pd.read_csv(self.test_tourney_path + '/' + self.game_results)
        # create empty dataframe for current rating period
        rating_period = pd.DataFrame(columns=['game_id', 'agent_id_a', 'agent_id_b', 'score_a', 'score_b', 'rating_period'])
        # start from last row and work backwards until a former rating period is reached
        for index, row in tourney_results.iloc[::-1].iterrows():
            if row['rating_period'] != tourney_results.iloc[-1]['rating_period']:
                # former rating period reached
                break
            else:
                # add to current rating period
                rating_period = rating_period.append(row, ignore_index=True)

        # create for each agent a game history dataframe for the current rating period
        agent_game_histories = {}
        agent_game_history_df = pd.DataFrame(columns=['game_id', 'opponent', 'score', 'opponent_score', 'rating_self', 'rating_deviation_self', 'volatility_self', 'rating_opponent', 'rating_deviation_opponent', 'volatility_opponent'])
        agent_game_history_df_a = agent_game_history_df.copy()
        agent_game_history_df_b = agent_game_history_df.copy()
        # calculate new ratings based on tourney results
        for index, game in rating_period.iterrows():
            # get agent names
            agent_a = game['agent_id_a']
            agent_b = game['agent_id_b']
            # get agent ratings
            rating_a = rating_histories[agent_a].iloc[-1]['rating']
            rating_b = rating_histories[agent_b].iloc[-1]['rating']
            # get agent rating deviations
            rating_deviation_a = rating_histories[agent_a].iloc[-1]['rating_deviation']
            rating_deviation_b = rating_histories[agent_b].iloc[-1]['rating_deviation']
            # get agent volatility
            volatility_a = rating_histories[agent_a].iloc[-1]['volatility']
            volatility_b = rating_histories[agent_b].iloc[-1]['volatility']
            # get agent scores
            score_a = game['score_a']
            score_b = game['score_b']
            # determine game id for agent - if agent has no game history, game id is 0, otherwise it is the last game id + 1
            if agent_a in agent_game_histories:
                game_id_a = agent_game_histories[agent_a].iloc[-1]['game_id'] + 1
            else:
                game_id_a = 0
            if agent_b in agent_game_histories:
                game_id_b = agent_game_histories[agent_b].iloc[-1]['game_id'] + 1
            else:
                game_id_b = 0
            # add game to agent game history
            agent_game_history_df_a = agent_game_history_df_a.append({'game_id': game_id_a, 'opponent': agent_b, 'score': score_a, 'opponent_score': score_b, 'rating_self': rating_a, 'rating_deviation_self': rating_deviation_a, 'volatility_self': volatility_a, 'rating_opponent': rating_b, 'rating_deviation_opponent': rating_deviation_b, 'volatility_opponent': volatility_b}, ignore_index=True)
            agent_game_history_df_b = agent_game_history_df_b.append({'game_id': game_id_b, 'opponent': agent_a, 'score': score_b, 'opponent_score': score_a, 'rating_self': rating_b, 'rating_deviation_self': rating_deviation_b, 'volatility_self': volatility_b, 'rating_opponent': rating_a, 'rating_deviation_opponent': rating_deviation_a, 'volatility_opponent': volatility_a}, ignore_index=True)
            # update agent game history
            agent_game_histories[agent_a] = agent_game_history_df_a
            agent_game_histories[agent_b] = agent_game_history_df_b

        # calculate new ratings based on game results for each agent
        for agent_name, game_history_df in agent_game_histories.items():
            # get current agent rating
            rating = rating_histories[agent_name].iloc[-1]['rating']
            # get current agent rating deviation
            rating_deviation = rating_histories[agent_name].iloc[-1]['rating_deviation']
            # get current agent volatility
            volatility = rating_histories[agent_name].iloc[-1]['volatility']
            # calculate new rating
            rating_updated, rating_deviation_updated, volatility_updated = Glicko2.calculate_standard_glicko2(Glicko2, rating=rating, rating_deviation=rating_deviation, volatility=volatility, oppenents_in_period=game_history_df)            
            # create test results directory and csv files if they don't exist
            if not os.path.exists(self.test_tourney_results_path + '/' + agent_name):
                os.makedirs(self.test_tourney_results_path + '/' + agent_name)
                with open(self.test_tourney_results_path + '/' + agent_name + '/' + self.player_results_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['rating', 'rating_deviation', 'volatility'])
            # open csv file to append new ratings for current agent
            with open(self.test_tourney_results_path + '/' + agent_name + '/' + self.player_results_file, 'a', newline='') as file:
                # csv writer
                writer = csv.writer(file)
                # write new ratings at the end of the csv file without overwriting existing ratings
                writer.writerow([rating_updated, rating_deviation_updated, volatility_updated])            
            
        # check if new ratings are correct
        # get agent0 rating history to pd.DataFrame
        agent0_rating_history = pd.read_csv(self.test_tourney_results_path + '/agent0/' + self.player_results_file)
        # get last row of agent0 rating history
        last_row = agent0_rating_history.iloc[-1]
        # check if new rating is correct
        self.assertEqual(round(last_row['rating'], 2), 1464.05)
        # check if new rating deviation is correct
        self.assertEqual(round(last_row['rating_deviation'], 2), 151.52)
        # check if new volatility is correct
        self.assertEqual(HelperFunctions.truncate(last_row['volatility'], 5), 0.05999)

if __name__ == '__main__':
    unittest.main()