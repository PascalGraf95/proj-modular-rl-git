import math

class Elo:
    # calculate standard Elo
    def calculate_standard_elo(rating_a, rating_b, score, k=32):
        # calculate players expected score:
        # their probability of winning + half their probability of drawing
        estimated = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

        # updating Elo-rating
        if score == 1:
            rating_updated = rating_a + k * (1 - estimated)
        elif score == 0:
            rating_updated = rating_a + k * (0 - estimated)
        elif score == 0.5:
            rating_updated = rating_a + k * (0.5 - estimated)            

        return rating_updated


    

# '''
# ######   DEBUG  ######
# '''
# from random import randrange

# a = 0
# b = 0
# for i in range(1000):
#     margin_check = True
#     while margin_check:
#         agent_score = randrange(10)
#         human_score = randrange(10)
#         if agent_score != human_score:
#             margin_check = False

#     if i == 1:
#         a, b = standard_elo(agent_elo, human_elo, agent_score, human_score)
#     else:
#         a, b = standard_elo(a, b, agent_score, human_score)
    
#     print(a, b)

# '''
# ######   DEBUG  ######
# '''
