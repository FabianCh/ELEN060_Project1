from Implementation import *
import numpy as np

x_y_joint_distribution = np.array([[1/8, 1/16, 1/16, 1/4],
                                 [1/16, 1/8, 1/16, 0],
                                 [1/32, 1/32, 1/16, 0],
                                 [1/32, 1/32, 1/16, 0]])

x_probability_distribution = np.sum(x_y_joint_distribution, axis=0)
y_probability_distribution = np.sum(x_y_joint_distribution, axis=1)


print("1.   H(x) = " + str(entropy(x_probability_distribution)))
print("     H(y) = " + str(entropy(y_probability_distribution)))
print("     H(w) = " + str(None))
print("     H(z) = " + str(None))

print("2.   H(x,y) = " + str(joint_entropy(x_y_joint_distribution)))
print("     H(x,w) = " + str(None))
print("     H(y,w) = " + str(None))
print("     H(w,z) = " + str(None))

print("3.   H(x|y) = " + str(conditional_entropy(x_y_joint_distribution)))
print("     H(x|w) = " + str(None))
print("     H(y|w) = " + str(None))
print("     H(w|z) = " + str(None))

print("4.   H(x,y|w) = " + str(None))
print("     H(w,z|x) = " + str(None))

print("5.   I(x;y) = " + str(mutual_information(x_y_joint_distribution)))
print("     I(x;w) = " + str(None))
print("     I(y;z) = " + str(None))
print("     I(w;z) = " + str(None))

print("6.   I(x;y|w) = " + str(None))
print("     I(w;z|x) = " + str(None))

