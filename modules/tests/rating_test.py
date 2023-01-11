import os
import pandas as pd
import shutil
import csv
import math
from misc.elo import Elo
from training_algorithms.agent_blueprint import Actor as ab
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

    def truncate(f, n):
        #Truncates/pads a float f to n decimal places without rounding
        return math.floor(f * 10 ** n) / 10 ** n


class TestGlicko2(unittest.TestCase):
    test_tourney_path = 'modules/tests/rating_test'
    test_tourney_results_path = 'modules/tests/rating_test_results'
    game_results = 'game_results.csv'
    player_results_file = 'rating_history.csv'

    def test_player_rating_history(self):
        # delete test tourney results directory if it exists
        if os.path.exists(self.test_tourney_results_path):
            shutil.rmtree(self.test_tourney_results_path)
        # rating histories dictionary
        rating_histories = ab.get_agent_rating_history(self, self.test_tourney_path, self.player_results_file) 
        # check if rating history was created for each agent
        self.assertEqual(len(rating_histories), 4)
        # check if rating history has correct number of rows
        self.assertEqual(len(rating_histories['Agent0']), 1)
        # check if rating history has correct columns
        self.assertEqual(list(rating_histories['Agent0'].columns), ['rating', 'rating_deviation', 'volatility'])        
        # check if rating history has correct values
        self.assertEqual(rating_histories['Agent0'].iloc[0]['rating'], 1500)
        self.assertEqual(rating_histories['Agent0'].iloc[0]['rating_deviation'], 200)
        self.assertEqual(rating_histories['Agent0'].iloc[0]['volatility'], 0.06)

    def test_getting_current_rating_period(self):
        rating_period = ab.get_current_rating_period(self, self.test_tourney_path) 
        # check if rating period is zero
        self.assertEqual(rating_period, 0)

    def test_updating_game_results_history(self):
        # delete test tourney results directory if it exists
        if os.path.exists(self.test_tourney_results_path):
            shutil.rmtree(self.test_tourney_results_path)
        # create instance of Actor class
        actor = ab(0, 0, 'testing', 'MLAgentsV18', 'None', 'None', 'None', 'test', 'test')
        # add test game results to game results history        
        ab.update_game_results_history(actor, self.test_tourney_path, 1, 0, 3, 'agent_1', 'agent_2')              
        # check if last row of game results history is correct
        with open(self.test_tourney_path + '/' + self.game_results, 'r') as f:
            reader = csv.reader(f)
            last_row = list(reader)[-1]
            self.assertEqual(last_row, ['agent_1', 'agent_2', '3', '1', '0', '0'])
        # delete last row of game results history to return to initial state
        with open(self.test_tourney_path + '/' + self.game_results, 'r') as f:
            lines = f.readlines()
            lines = lines[:-1]
            with open(self.test_tourney_path + '/' + self.game_results, 'w') as f:
                f.writelines(lines)

    def test_glicko2_calculation(self):        
        # delete test tourney results directory if it exists
        if os.path.exists(self.test_tourney_results_path):
            shutil.rmtree(self.test_tourney_results_path)
        # create new test tourney results directory. Copy everything from rating test directory to rating test results directory
        shutil.copytree(self.test_tourney_path, self.test_tourney_results_path)        
        # create instance of Actor class
        actor = ab(0, 0, 'testing', 'MLAgentsV18', 'None', 'None', 'None', 'test', 'test')
        # run glicko2 calculation
        ab.update_ratings(actor, self.test_tourney_results_path)         
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