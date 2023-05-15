import math
import numpy as np
import pandas as pd

# calculate standard Glicko2
def calculate_updated_glicko2(rating, rating_deviation, volatility, opponents_in_period, tau=0.5, rd_threshold=30):
    """
    Function to calculate an updated glicko rating for a player a given the score of a played match between player a and
    player b as well as their previous rating estimates.
    :param rating: Rating of player
    :param rating_deviation: Rating deviation of player
    :param volatility: Volatility of player
    :param opponents_in_period: Dataframe containing the opponents of player a in the rating period
    :param tau: System constant which constrains the change in volatility over time
    :return:
    """
    # Step 1: Determine a rating and RD for each player at the onset of the rating period. The
    # system constant, τ , which constrains the change in volatility over time, needs to be
    # set prior to application of the system.
    # (a) If the player is unrated, set the rating to 1500 and the RD to 350. Set the volatility to 0.06.
    # (b) If the player is rated, use the rating, rating deviation and volatility from the previous rating period.

    # create to store the opponents glicko2 calculations
    opponents = pd.DataFrame(columns=['my', 'phi', 'g', 'E', 's'])
    # apply rating deviation threshold. If rating deviation is below threshold, set it to threshold
    rating_deviation = max(rating_deviation, rd_threshold)
    # Step 2: Convert ratings to Glicko-2 scale
    converted_rating, converted_rating_deviation = _convert_to_glicko2_scale(rating, rating_deviation)
    for index, opponent in opponents_in_period.iterrows():
        score = calculate_normalized_score(opponent['score'], opponent['opponent_score'])
        converted_rating_opponent, converted_rating_deviation_opponent = _convert_to_glicko2_scale(opponent['rating_opponent'], opponent['rating_deviation_opponent'])
        g = _calculate_g(converted_rating_deviation_opponent)
        e = _calculate_e(g, converted_rating, converted_rating_opponent)
        # add row to opponents dataframe
        opponents.loc[index] = [converted_rating_opponent, converted_rating_deviation_opponent, g, e, score]

    # Step 3: Compute the quantity v. This is the estimated variance of the team's/player's rating based only on game outcomes.
    v = _calculate_v(opponents['g'].to_list(), opponents['E'].to_list())

    # Step 4: Compute the quantity delta. This is the estimated improvement in rating capability based on the game outcomes and the value of v from the previous step.
    delta = _calculate_delta(v, opponents['g'].to_list(), opponents['s'].to_list(), opponents['E'].to_list())

    # Step 5: Compute the new values of the volatility variables sigma'.
    volatility_updated = _calculate_volatility(volatility, delta, v, tau, converted_rating_deviation)

    # Step 6: Compute the new pre rating period deviation phi*.
    rating_deviation_pre_rating_period = math.sqrt((converted_rating_deviation ** 2) + (volatility_updated ** 2))

    # Step 7: Update the rating and rating deviation to the new values
    rating_updated, rating_deviation_updated = _update_rating(converted_rating, rating_deviation_pre_rating_period, v, opponents['g'].to_list(), opponents['E'].to_list(), opponents['s'].to_list())

    # Step 8: Convert ratings back to original scale
    rating_updated, rating_deviation_updated = _convert_to_original_scale(rating_updated, rating_deviation_updated)
    return rating_updated, rating_deviation_updated, volatility_updated


# Step 2: Convert ratings to Glicko-2 scale
def _convert_to_glicko2_scale(rating, rating_deviation):
    converted_rating = (rating - 1500) / 173.7178
    converted_rating_deviation = rating_deviation / 173.7178
    return converted_rating, converted_rating_deviation


# calculate g
def _calculate_g(converted_rating_deviation):
    g = 1 / math.sqrt(1 + (3 * (converted_rating_deviation ** 2)) / (math.pi ** 2))
    return g


# calculate e
def _calculate_e(g, first_rating, second_rating):
    e = 1 / (1 + math.exp(-g * (first_rating - second_rating)))
    return e


# calculate v
def _calculate_v(g_list, e_list):
    v = 1 / sum([(g ** 2) * e * (1 - e) for g, e in zip(g_list, e_list)])
    return v


# calculate score without taking account of margin of victory
def calculate_normalized_score(score_a, score_b):
    """
    Given the score of a game return a normalized score between 0 and 1
    :param score_a: Score of player a
    :param score_b: Score of player b
    :return: Normalized score
    """
    if score_a > score_b:
        return 1
    elif score_b > score_a:
        return 0
    elif score_a == score_b:
        return 0.5


# calculate delta
def _calculate_delta(v, g_list, score_list, e_list):
    delta = v * sum([g * (score - e) for g, score, e in zip(g_list, score_list, e_list)])
    return delta


# calculate f
def _calculate_f(x, delta, v, tau, rating_deviation, a):
    f = math.exp(x) * ((delta ** 2) - (rating_deviation ** 2) - v - math.exp(x)) / (2 * ((rating_deviation ** 2) + v + math.exp(x)) ** 2) - ((x - a) / (tau ** 2))
    return f


# calculate volatility
def _calculate_volatility(volatility, delta, v, tau, rating_deviation, epsilon=0.000001, max_iterations=1000):
    # Step 5.2: Set the initial values of the iterative algorithm
    a = np.log(volatility ** 2)
    A = a
    if (delta ** 2) > ((rating_deviation ** 2) + v):
        B = np.log(delta ** 2 - (rating_deviation ** 2) - v)
    if (delta ** 2) <= ((rating_deviation ** 2) + v):
        k = 1
        while _calculate_f(a - k * tau, delta, v, tau, rating_deviation, a) < 0:
            k += 1
            B = a - (k * tau)
        B = a - (k * tau)
    # Step 5.3: Calculate f(a) and f(b)
    f_a = _calculate_f(A, delta, v, tau, rating_deviation, a)
    f_b = _calculate_f(B, delta, v, tau, rating_deviation, a)
    # Step 5.4: While |b-a| > e, carry out the following steps.
    iterations = 0
    while abs(B - A) > epsilon:
        iterations += 1
        if iterations > max_iterations:
            print('Max iterations reached')
            break
        # if abs(B - A) <= epsilon:
        #     break
        else:
            C = A + ((A - B) * f_a) / (f_b - f_a)
            f_c = _calculate_f(C, delta, v, tau, rating_deviation, a)
            if f_c * f_b <= 0:
                A = B
                f_a = f_b
            else:
                f_a = f_a / 2
            B = C
            f_b = f_c
    # Step 5.5: Update the volatility        
    volatility_updated = math.exp(A / 2)
    return volatility_updated


def _update_rating(rating, rating_deviation_pre_rating_period, v, g_list, e_list, score_list):
    rating_deviation_updated = 1 / math.sqrt((1 / (rating_deviation_pre_rating_period ** 2)) + (1 / v))
    rating_updated = rating + (rating_deviation_updated ** 2) * sum([g * (score - e) for g, score, e in zip(g_list, score_list, e_list)])
    return rating_updated, rating_deviation_updated


def _convert_to_original_scale(converted_rating, converted_rating_deviation):
    rating = converted_rating * 173.7178 + 1500
    rating_deviation = converted_rating_deviation * 173.7178
    return rating, rating_deviation
