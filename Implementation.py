import math
import numpy as np


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
            if p != 0:
                h -= p * math.log2(p)
    return h


def conditional_entropy(joint_distribution):
    h = 0
    for i in range(len(joint_distribution)):
        pi = sum(joint_distribution[i])
        for j in range(len(joint_distribution[1])):
            p = joint_distribution[i][j]
            if p != 0:
                h -= p * math.log2(p/pi)
    return h


def mutual_information(joint_distribution):
    h = 0
    for i in range(len(joint_distribution)):
        pi = np.sum(joint_distribution, axis=1)[i]
        for j in range(len(joint_distribution[i])):
            pj = np.sum(joint_distribution, axis=0)[j]
            p = joint_distribution[i][j]
            if p != 0:
                h += p * math.log2(p / (pi * pj))
    return h


def cond_joint_entropy(joint_distribution_3d):
    h = 0
    for i in range(len(joint_distribution_3d)):
        for j in range(len(joint_distribution_3d[i])):
            for k in range(len(joint_distribution_3d[i][j])):
                p = joint_distribution_3d[i][j][k]
                if p != 0:
                    h -= p * math.log2(p)
    x_probability_distribution = np.sum(joint_distribution_3d, axis=(0, 1))
    h3 = entropy(x_probability_distribution)
    return h - h3


def cond_mutual_information():
    pass
