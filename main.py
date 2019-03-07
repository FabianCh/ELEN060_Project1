from Implementation import *
import numpy as np

x_y_joint_distribution = np.array([[1/8, 1/16, 1/16, 1/4],
                                   [1/16, 1/8, 1/16, 0],
                                   [1/32, 1/32, 1/16, 0],
                                   [1/32, 1/32, 1/16, 0]])

w_given_x_y = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

z_given_x_y = np.array([[0, 1, 1, 1],
                        [1, 0, 1, 1],
                        [1, 1, 0, 1],
                        [1, 1, 1, 0]])

x_probability_distribution = np.sum(x_y_joint_distribution, axis=0)
y_probability_distribution = np.sum(x_y_joint_distribution, axis=1)
w_probability_distribution = [5/16, 11/16]
z_probability_distribution = [11/16, 5/16]

x_w_joint_distribution = np.array([[1/8, 1/8, 1/16, 0],
                                   [1/8, 1/8, 3/16, 1/4]])
y_w_joint_distribution = np.array([[1/8, 1/8, 1/16, 0],
                                   [3/8, 1/8, 1/16, 1/8]])
w_z_joint_distribution = np.array([[0, 5/16],
                                   [11/16, 0]])
y_z_joint_distribution = np.array([[1/8, 1/8, 1/16, 0],
                                   [1/8, 1/8, 3/16, 1/4]])

print("1.   H(X) = " + str(entropy(x_probability_distribution)))
print("     H(Y) = " + str(entropy(y_probability_distribution)))
print("     H(W) = " + str(entropy(w_probability_distribution)))
print("     H(Z) = " + str(entropy(z_probability_distribution)))

print("2.   H(X,Y) = " + str(joint_entropy(x_y_joint_distribution)))
print("     H(X,W) = " + str(joint_entropy(x_w_joint_distribution)))
print("     H(Y,W) = " + str(joint_entropy(y_w_joint_distribution)))
print("     H(W,Z) = " + str(joint_entropy(w_z_joint_distribution)))

print("3.   H(X|Y) = " + str(conditional_entropy(x_y_joint_distribution)))
print("     H(W|X) = " + str(conditional_entropy(x_w_joint_distribution)))
print("     H(Z|W) = " + str(conditional_entropy(w_z_joint_distribution)))
print("     H(W|Z) = " + str(conditional_entropy(np.transpose(w_z_joint_distribution))))

print("4.   H(X,Y|W) = " + str(None))
print("     H(W,Z|X) = " + str(None))

print("5.   I(X;Y) = " + str(mutual_information(x_y_joint_distribution)))
print("     I(X;W) = " + str(mutual_information(x_w_joint_distribution)))
print("     I(Y;Z) = " + str(mutual_information(y_z_joint_distribution)))
print("     I(W;Z) = " + str(mutual_information(w_z_joint_distribution)))

print("6.   I(X;Y|W) = " + str(None))
print("     I(W;Z|X) = " + str(None))
