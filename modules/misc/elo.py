import math
from modules.misc.glicko2 import *
import csv
import os
import numpy as np


def calculate_updated_elo(rating_a, rating_b, score, k=32):
    """
    Function to calculate an updated elo rating for a player a given the score of a played match between player a and
    player b as well as their previous rating estimates.
    :param rating_a: Current estimated rating of player a
    :param rating_b: Current estimated rating of player b
    :param score: Score of a match between player a and b where 1 means a has won, 0 means b has one and 0.5 means draw
    :param k: ToDo: Specify
    :return: Updated rating of player a and player b
    """
    rating_a = float(rating_a)
    rating_b = float(rating_b)
    # Calculate the expected score given the ratings of player a and b
    # their probability of winning + half their probability of drawing
    expected_score_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_score_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))

    # Calculate the actual transformed score for a and b
    score_a = score
    score_b = np.abs(score - 1)

    # Calculate the updated rating based on the difference of the expected and the actual score.
    rating_a = rating_a + k * (score_a - expected_score_a)
    rating_b = rating_b + k * (score_b - expected_score_b)
    return min(max(rating_a, 0), 4000), min(max(rating_b, 0), 4000)


def main():
    # Calculate or update an elo rating based on list of played games between different agents.
    # 1. Open or create a yalm file with agents' keys and an elo rating and parse it into a dictionary.
    player_ratings = {}
    rating_path = r"../../training/player_ratings.csv"
    if os.path.isfile(rating_path):
        with open(rating_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                player_ratings[row[0]] = row[1]

    # 2. Open the game history and parse it into a list
    game_history = []
    history_path = r"../../training/rating_history.csv"
    if os.path.isfile(history_path):
        with open(history_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                game_history.append(row)
    else:
        print("ERROR: No game history found!")
        return
    # 3. For each game update the elo ratings of the respective players
    for game in game_history:
        game_id, player_key_a, player_key_b, score_a, score_b = game
        if player_key_a in player_ratings:
            elo_a = player_ratings[player_key_a]
        else:
            elo_a = 1500

        if player_key_b in player_ratings:
            elo_b = player_ratings[player_key_b]
        else:
            elo_b = 1500

        elo_a, elo_b = calculate_updated_elo(elo_a, elo_b, calculate_normalized_score(score_a, score_b))
        player_ratings[player_key_a] = elo_a
        player_ratings[player_key_b] = elo_b

    # 4. Store the results into the csv file
    player_ratings = {k: v for k, v in sorted(player_ratings.items(), key=lambda item: item[1], reverse=True)}
    with open(rating_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["player_key", "elo_score"])
        for key, val in player_ratings.items():
            writer.writerow([key, val])


if __name__ == '__main__':
    main()



