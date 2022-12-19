import math
import pandas as pd

class Glicko2:
    # Step 2: Convert ratings to Glicko-2 scale
    def convert_to_glicko2_scale(rating, rating_deviation):
        converted_rating = (rating - 1500) / 173.7178
        converted_rating_deviation = rating_deviation / 173.7178
        return converted_rating, converted_rating_deviation

    # calculate g
    def calculate_g(converted_rating_deviation):
        g = 1 / math.sqrt(1 + (3 * (converted_rating_deviation ** 2)) / (math.pi ** 2))
        return g
    
    # calculate e
    def calculate_e(g, first_rating, second_rating):
        e = 1 / (1 + math.exp(-g * (first_rating - second_rating)))
        return e

    # calculate v
    def calculate_v(g_list, e_list):
        v = 1 / sum([(g ** 2) * e * (1 - e) for g, e in zip(g_list, e_list)])
        return v
    
    # calculate score without taking account of margin of victory
    def calculate_score(score_a, score_b):
        if score_a > score_b:
            score = 1
        elif score_b > score_a:
            score = 0
        elif score_a == score_b:
            score = 0.5
        return score

    # calculate delta
    def calculate_delta(v, g_list, score_list, e_list):
        delta = v * sum([g * (score - e) for g, score, e in zip(g_list, score_list, e_list)])
        return delta

    # calculate f
    def calculate_f(x, delta, v, tau, rating_deviation, a):
        f = math.exp(x) * ((delta ** 2) - (rating_deviation ** 2) - v - math.exp(x)) / (2 * ((rating_deviation ** 2) + v + math.exp(x)) ** 2) - ((x - a) / (tau ** 2))
        return f

    # calculate volatility
    def calculate_volatility(volatility, delta, v, tau, rating_deviation, epsilon=0.000001):
        a = math.log(volatility ** 2)
        if delta ** 2 > (rating_deviation ** 2) + v:
            b = math.log(delta ** 2 - (rating_deviation ** 2) - v)
        if delta ** 2 <= (rating_deviation ** 2) + v:
            k = 1
            while True:
                if Glicko2.calculate_f(a - k * tau, delta, v, tau, rating_deviation, a) < 0:
                    k += 1
                else:
                    break                                   
            b = a - k * tau        
        f_a = Glicko2.calculate_f(a, delta, v, tau, rating_deviation, a)
        f_b = Glicko2.calculate_f(b, delta, v, tau, rating_deviation, a)
        while True:
            if (abs(b - a) > epsilon):
                c = a + ((a - b) * f_a) / (f_b - f_a)
                f_c = Glicko2.calculate_f(c, delta, v, tau, rating_deviation, a)
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

    def update_rating(rating, rating_deviation_pre_rating_period, v, g_list, e_list, score_list):
        rating_deviation_updated = 1 / math.sqrt((1 / (rating_deviation_pre_rating_period ** 2)) + (1 / v))
        rating_updated = rating + rating_deviation_updated ** 2 * sum([g * (score - e) for g, score, e in zip(g_list, score_list, e_list)])
        return rating_updated, rating_deviation_updated

    
    def convert_to_original_scale(converted_rating, converted_rating_deviation):
        rating = converted_rating * 173.7178 + 1500
        rating_deviation = converted_rating_deviation * 173.7178
        return rating, rating_deviation

    # calculate standard Glicko2
    def calculate_standard_glicko2(self, rating, rating_deviation, volatility, oppenents_in_period, tau=0.5):
        # Step 1: Determine a rating and RD for each player at the onset of the rating period. The
        # system constant, τ , which constrains the change in volatility over time, needs to be
        # set prior to application of the system. 
        # (a) If the player is unrated, set the rating to 1500 and the RD to 350. Set the volatility to 0.06.
        # (b) If the player is rated, use the rating, rating deviation and volatility from the previous rating period.
        # These values are stored in the corresponding csv files and should be used to initialize this function.

        # create to store the opponents glicko2 calculations
        oppenents = pd.DataFrame(columns=['my', 'phi', 'g', 'E', 's'])
        # agent_game_history_df = pd.DataFrame(columns=['game_id', 'opponent', 'score', 'opponent_score', 'rating_self', 'rating_deviation_self', 'volatility_self', 'rating_opponent', 'rating_deviation_opponent', 'volatility_opponent'])
        # Step 2: Convert ratings to Glicko-2 scale
        converted_rating, converted_rating_deviation = self.convert_to_glicko2_scale(rating, rating_deviation)        
        for index, opponent in oppenents_in_period.iterrows():
            score = self.calculate_score(opponent['score'], opponent['opponent_score'])
            converted_rating_opponent, converted_rating_deviation_opponent = self.convert_to_glicko2_scale(opponent['rating_opponent'], opponent['rating_deviation_opponent'])
            g = self.calculate_g(converted_rating_deviation_opponent)
            e = self.calculate_e(g, converted_rating, converted_rating_opponent)         
            # add row to oppenents dataframe
            oppenents.loc[index] = [converted_rating_opponent, converted_rating_deviation_opponent, g, e, score]

        # Step 3: Compute the quantity v. This is the estimated variance of the team's/player's rating based only on game outcomes.
        v = self.calculate_v(oppenents['g'].to_list(), oppenents['E'].to_list())                    

        # Step 4: Compute the quantity delta. This is the estimated improvement in rating capability based on the game outcomes and the value of v from the previous step.
        delta = self.calculate_delta(v, oppenents['g'].to_list(), oppenents['s'].to_list(), oppenents['E'].to_list())

        # Step 5: Compute the new values of the volatility variables sigma'.
        volatility_updated = self.calculate_volatility(volatility, delta, v, tau, converted_rating_deviation)

        # Step 6: Compute the new pre rating period deviation phi*.
        rating_deviation_pre_rating_period = math.sqrt(converted_rating_deviation ** 2 + volatility_updated ** 2)

        # Step 7: Update the rating and rating deviation to the new values
        rating_updated, rating_deviation_updated = self.update_rating(converted_rating, rating_deviation_pre_rating_period, v, oppenents['g'].to_list(), oppenents['E'].to_list(), oppenents['s'].to_list())

        # Step 8: Convert ratings back to original scale
        rating_updated, rating_deviation_updated = self.convert_to_original_scale(rating_updated, rating_deviation_updated)
        return rating_updated, rating_deviation_updated, volatility_updated   
    