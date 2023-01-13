import math
from modules.misc.glicko2 import *


def calculate_updated_elo(rating_a, rating_b, score, k=32):
    """
    Function to calculate an updated elo rating for a player a given the score of a played match between player a and
    player b as well as their previous rating estimates.
    :param rating_a: Current estimated rating of player a
    :param rating_b: Current estimated rating of player b
    :param score: Score of a match between player a and b where 1 means a has won, 0 means b has one and 0.5 means draw
    :param k: ToDo: Specify
    :return: Updated rating of player a
    """
    # Calculate the expected score given the ratings of player a and b
    # their probability of winning + half their probability of drawing
    estimated = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    # Depending on the actual game result return an updated rating for player a.
    if score == 1:
        # Player a won
        return rating_a + k * (1 - estimated)
    elif score == 0:
        # Player b won
        return rating_a + k * (0 - estimated)
    elif score == 0.5:
        # Draw
        return rating_a + k * (0.5 - estimated)

    print("WARNING: Invalid score value of {:.3f} when trying to calculate elo rating!".format(score))
    return rating_a
