import math
import pandas as pd

# calculate standard Glicko2
def calculate_updated_glicko2(rating, rating_deviation, volatility, oppenents_in_period, tau=0.5):
    """
    Function to calculate an updated glicko rating for a player a given the score of a played match between player a and
    player b as well as their previous rating estimates.
    :param rating:
    :param rating_deviation:
    :param volatility:
    :param oppenents_in_period:
    :param tau:
    :return:
    """
    # Step 1: Determine a rating and RD for each player at the onset of the rating period. The
    # system constant, τ , which constrains the change in volatility over time, needs to be
    # set prior to application of the system.
    # (a) If the player is unrated, set the rating to 1500 and the RD to 350. Set the volatility to 0.06.
    # (b) If the player is rated, use the rating, rating deviation and volatility from the previous rating period.
    # These values are stored in the corresponding csv files and should be used to initialize this function.

    # create to store the opponents glicko2 calculations
    oppenents = pd.DataFrame(columns=['my', 'phi', 'g', 'E', 's'])
    # Step 2: Convert ratings to Glicko-2 scale
    converted_rating, converted_rating_deviation = _convert_to_glicko2_scale(rating, rating_deviation)
    for index, opponent in oppenents_in_period.iterrows():
        score = _calculate_normalized_score(opponent['score'], opponent['opponent_score'])
        converted_rating_opponent, converted_rating_deviation_opponent = _convert_to_glicko2_scale(opponent['rating_opponent'], opponent['rating_deviation_opponent'])
        g = _calculate_g(converted_rating_deviation_opponent)
        e = _calculate_e(g, converted_rating, converted_rating_opponent)
        # add row to oppenents dataframe
        oppenents.loc[index] = [converted_rating_opponent, converted_rating_deviation_opponent, g, e, score]

    # Step 3: Compute the quantity v. This is the estimated variance of the team's/player's rating based only on game outcomes.
    v = _calculate_v(oppenents['g'].to_list(), oppenents['E'].to_list())

    # Step 4: Compute the quantity delta. This is the estimated improvement in rating capability based on the game outcomes and the value of v from the previous step.
    delta = _calculate_delta(v, oppenents['g'].to_list(), oppenents['s'].to_list(), oppenents['E'].to_list())

    # Step 5: Compute the new values of the volatility variables sigma'.
    volatility_updated = _calculate_volatility(volatility, delta, v, tau, converted_rating_deviation)

    # Step 6: Compute the new pre rating period deviation phi*.
    rating_deviation_pre_rating_period = math.sqrt(converted_rating_deviation ** 2 + volatility_updated ** 2)

    # Step 7: Update the rating and rating deviation to the new values
    rating_updated, rating_deviation_updated = _update_rating(converted_rating, rating_deviation_pre_rating_period, v, oppenents['g'].to_list(), oppenents['E'].to_list(), oppenents['s'].to_list())

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
def _calculate_normalized_score(score_a, score_b):
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
def _calculate_volatility(volatility, delta, v, tau, rating_deviation, epsilon=0.000001):
    a = math.log(volatility ** 2)
    if delta ** 2 > (rating_deviation ** 2) + v:
        b = math.log(delta ** 2 - (rating_deviation ** 2) - v)
    else:
        k = 1
        while True:
            if _calculate_f(a - k * tau, delta, v, tau, rating_deviation, a) < 0:
                k += 1
            else:
                break
        b = a - k * tau
    f_a = _calculate_f(a, delta, v, tau, rating_deviation, a)
    f_b = _calculate_f(b, delta, v, tau, rating_deviation, a)
    while True:
        if abs(b - a) > epsilon:
            c = a + ((a - b) * f_a) / (f_b - f_a)
            f_c = _calculate_f(c, delta, v, tau, rating_deviation, a)
            if f_c * f_b <= 0:
                a = b
                f_a = f_b
            else:
                f_a = f_a / 2
            b = c
            f_b = f_c
        else:
            break
    volatility_updated = math.exp(a / 2)
    return volatility_updated


def _update_rating(rating, rating_deviation_pre_rating_period, v, g_list, e_list, score_list):
    rating_deviation_updated = 1 / math.sqrt((1 / (rating_deviation_pre_rating_period ** 2)) + (1 / v))
    rating_updated = rating + rating_deviation_updated ** 2 * sum([g * (score - e) for g, score, e in zip(g_list, score_list, e_list)])
    return rating_updated, rating_deviation_updated


def _convert_to_original_scale(converted_rating, converted_rating_deviation):
    rating = converted_rating * 173.7178 + 1500
    rating_deviation = converted_rating_deviation * 173.7178
    return rating, rating_deviation


"""
def get_current_rating_period(self, model_path):
    # get the current rating period from the game results file using pandas
    rating_period = pd.read_csv(os.path.join(model_path, self.game_results))['rating_period'].iloc[-1]
    return rating_period

def update_rating_period(self, model_path):
    # get the current rating period from the game results file using pandas
    rating_period = self.get_current_rating_period(model_path)
    # get all the games in the current rating period
    current_rating_period_games = pd.read_csv(os.path.join(model_path, self.game_results))[pd.read_csv(os.path.join(model_path, self.game_results))['rating_period'] == rating_period]
    # calculate the average number of games played by each agent in the current rating period
    average_games_played = current_rating_period_games.groupby(['agent_id_a', 'agent_id_b']).count()['game_id'].mean()
    # check if the average number of games played is greater than 10
    if average_games_played > 10:
        # update the rating period
        return rating_period + 1
    else:
        # return the current rating period
        return rating_period

def update_game_id(self, model_path):
    # get the current game id from the game results file using pandas
    game_id = pd.read_csv(os.path.join(model_path, self.game_results))['game_id'].iloc[-1]
    return game_id + 1

def get_agent_rating_history(self, path, player_results):
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
            else:
                print("No rating history found for agent: ", agent.name)
                print("Creating new rating history for agent: ", agent.name)
                # create new rating history
                with open(rating_history_path + '/' + self.player_results_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['rating', 'rating_deviation', 'volatility'])
                    # write initial rating values
                    writer.writerow([1500, 350, 0.06])
    return rating_histories

def update_ratings(self, model_path):
    # rating histories dictionary
    rating_histories = self.get_agent_rating_history(model_path, self.player_results_file)
    # get game results to pd.DataFrame
    tourney_results = pd.read_csv(model_path + '/' + self.game_results)
    # create empty dataframe for current rating period
    rating_period = pd.DataFrame(columns=['game_id', 'agent_id_a', 'agent_id_b', 'score_a', 'score_b', 'rating_period'])
    # start from last row and work backwards until a former rating period is reached
    for _, row in tourney_results.iloc[::-1].iterrows():
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
    for _, game in rating_period.iterrows():
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
        # open csv file to append new ratings for current agent
        with open(model_path + '/' + agent_name + '/' + self.player_results_file, 'a', newline='') as file:
            # csv writer
            writer = csv.writer(file)
            # write new ratings at the end of the csv file without overwriting existing ratings
            writer.writerow([rating_updated, rating_deviation_updated, volatility_updated])

"""


    