"""
        2018-11-28 Molly O'Brien

        Compute the score from a given trajectory in needle_master

        score computation copied from needlemaster/app/src/main/java/edu/jhu/lcsr/needlemaster/ScoringActivity.java by Chris Paxton
"""


"""
    Stuff we need:
    * level
    * num_gates
    * passed_gates
    * failed_gates
    * path_length
    * time_remaining
    * passed
    * deep_tissue
    * deep_hit
    * damage

"""

""" compute gate score """
if(num_gates == 0):
    gate_score = 1000
else:
    gate_score = 1000 * float(passed_gates)/num_gates

""" compute time score """
if(time_remaining > 5000):
    time_score = 1000
else:
    time_score = 1000 * float(time_remaining)/5000

""" compute path score """
path_score = -50*path_length

""" compute damage score """
damage_score = -4*damage
# penalize hitting deep tissue
if(deep_tissue and deep_hit):
    damage_score = damage_score - 1000


score = gate_score + time_score + path_score + damage_score

return score
