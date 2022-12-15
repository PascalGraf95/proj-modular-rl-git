import math

class Glicko2:
    # Step 2: Convert ratings to Glicko-2 scale
    def convert_to_glicko2_scale(self, rating, rating_deviation):
        converted_rating = (rating - 1500) / 173.7178
        converted_rating_deviation = rating_deviation / 173.7178
        return converted_rating, converted_rating_deviation

    # calculate g_phi
    def calculate_g_phi(self, converted_rating_deviation):
        g_phi = 1 / math.sqrt(1 + (3 * (converted_rating_deviation ** 2)) / (math.pi ** 2))
        return g_phi
    
    # calculate e_my
    def calculate_e_my(self, g_phi, first_rating, second_rating):
        e_my = 1 / (1 + math.exp(-g_phi * (first_rating - second_rating)))
        return e_my

    # calculate v
    def calculate_v(self, g_phi, e_my):
        v = 1 / (g_phi ** 2) * e_my * (1 - e_my)
        return v
    
    # calculate score without taking account of margin of victory
    def calculate_score(self, score_a, score_b):
        if score_a > score_b:
            score = 1
        elif score_b > score_a:
            score = 0
        elif score_a == score_b:
            score = 0.5
        return score

    # calculate delta
    def calculate_delta(self, v, g_phi, score, e_my):
        delta = v * g_phi * (score - e_my)
        return delta

    # calculate f
    def calculate_f(self, x, delta, v, tau, rating_deviation):
        f = (math.exp(x) * ((delta ** 2) - (rating_deviation ** 2) - v - math.exp(x))) / (2 * ((rating_deviation ** 2) + v + math.exp(x)) ** 2) - (x - math.log(tau ** 2)) / (tau ** 2)
        return f

    # calculate volatility
    def calculate_volatility(self, volatility, delta, v, tau, rating_deviation, epsilon=0.000001):
        a = math.log(volatility ** 2)
        if delta ** 2 > (rating_deviation ** 2) + v:
            b = math.log(delta ** 2 - (rating_deviation ** 2) - v)
        else:
            k = 1
            while True:
                b = a - k * tau
                if self.calculate_f(b, delta, v, tau, rating_deviation) < 0:
                    break
                k += 1
        f_a = self.calculate_f(a, delta, v, tau, rating_deviation)
        f_b = self.calculate_f(b, delta, v, tau, rating_deviation)
        while abs(b - a) > epsilon:
            c = a + ((a - b) * f_a) / (f_b - f_a)
            f_c = self.calculate_f(c, delta, v, tau, rating_deviation)
            if f_c * f_b < 0:
                a = b
                f_a = f_b
            else:
                f_a = f_a / 2
            b = c
            f_b = f_c
        volatility_updated = math.exp(a / 2)
        return volatility_updated

    # calculate standard Glicko2
    def calculate_standard_glicko2(self, rating_a, rating_b, score_a, score_b, rating_deviation_a, rating_deviation_b, volatility_a, volatility_b, tau):
        # Step 2: Convert ratings to Glicko-2 scale
        converted_rating_a, converted_rating_deviation_a = self.convert_to_glicko2_scale(rating_a, rating_deviation_a)
        converted_rating_b, converted_rating_deviation_b = self.convert_to_glicko2_scale(rating_b, rating_deviation_b)
        # Step 3: Compute the quantity v. This is the estimated variance of the team's/player's rating based only on game outcomes.
        g_phi_a = self.calculate_g_phi(converted_rating_deviation_a)
        g_phi_b = self.calculate_g_phi(converted_rating_deviation_b)
        e_my_a = self.calculate_e_my(g_phi_b, converted_rating_a, converted_rating_b)
        e_my_b = self.calculate_e_my(g_phi_a, converted_rating_b, converted_rating_a)
        v_a = self.calculate_v(g_phi_b, e_my_a)
        v_b = self.calculate_v(g_phi_a, e_my_b)
        # Step 4: Compute the quantity delta. This is the estimated improvement in rating capability based on the game outcomes and the value of v from the previous step.
        delta_a = self.calculate_delta(v_a, g_phi_b, self.calculate_score(score_a, score_b), e_my_a)
        delta_b = self.calculate_delta(v_b, g_phi_a, self.calculate_score(score_b, score_a), e_my_b)
        # Step 5: Compute the new values of the volatility variables sigma'.
        volatility_a_updated = self.calculate_volatility(volatility_a, delta_a, v_a, tau, converted_rating_deviation_a)
        volatility_b_updated = self.calculate_volatility(volatility_b, delta_b, v_b, tau, converted_rating_deviation_b)
        # Step 6: Compute the new rating deviation phi*.
        rating_deviation_a_updated = math.sqrt(converted_rating_a ** 2 + volatility_a_updated ** 2)
        rating_deviation_b_updated = math.sqrt(converted_rating_b ** 2 + volatility_b_updated ** 2)
        # Step 7: Update the rating and rating deviation to the new values
        rating_a_updated = 1 / (math.sqrt(1 / (rating_deviation_a_updated ** 2) + 1 / v_a))
        rating_b_updated = 1 / (math.sqrt(1 / (rating_deviation_b_updated ** 2) + 1 / v_b)) 
        return new_rating_a, new_rating_b, new_rating_deviation_a, new_rating_deviation_b, new_volatility_a, new_volatility_b
    
    
