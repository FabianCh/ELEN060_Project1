import math


def entropy(probability_distribution):
    h = 0
    for i in probability_distribution:
        h -= i * math.log2(i)
    return h


def joint_entropy(joint_distribution):
    h = 0
    for i in range(len(joint_distribution)):
        for j in range(len(joint_distribution[1])):
            p = joint_distribution[i][j]
            h -= p * math.log2(p)
    return h


def conditional_entropy(joint_distribution):
    pass


def mutual_information():
    pass


def cond_joint_entropy():
    pass


def cond_mutual_information():
    pass
