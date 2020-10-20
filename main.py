import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from math import sin, cos, atan2, pi
from IPython.display import display, Math, Latex, Markdown, HTML

def get_correspondence_indices(P, Q):
    """For each point in P find closest one in Q."""
    p_size = P.shape[1]
    q_size = Q.shape[1]
    correspondences = []
    for i in range(p_size):
        if (i % 1000 == 0):
            print(i)
        p_point = P[:, i]
        min_dist = sys.maxsize
        chosen_idx = -1
        for j in range(q_size):
            q_point = Q[:, j]
            dist = np.linalg.norm(q_point - p_point)
            if dist < min_dist:
                min_dist = dist
                chosen_idx = j
        correspondences.append((i, chosen_idx))
    return correspondences


def center_data(data, exclude_indices=[]):
    reduced_data = np.delete(data, exclude_indices, axis=1)
    center = np.array([reduced_data.mean(axis=1)]).T
    return center, data - center

def compute_error(P, Q, correspondances):
    err = 0
    for i in range(len(P)):
        err += np.linalg.norm(P[:,i] - Q[:,correspondances[i][1]])
    return err

def compute_cross_covariance(P, Q, correspondences, kernel=lambda diff: 1.0):
    cov = np.zeros((3, 3))
    exclude_indices = []
    for i, j in correspondences:
        p_point = P[:, [i]]
        q_point = Q[:, [j]]
        weight = kernel(p_point - q_point)
        if weight < 0.01: exclude_indices.append(i)
        cov += weight * q_point.dot(p_point.T)
    return cov, exclude_indices


def save(P):
    f = open("transformation.txt", "w")
    f.write("Points_0,Points_1,Points_2\n")
    for i in range (len(P)):
        f.write(str(P[:,i][0])+","+str(P[:,i][1])+","+str(P[:,i][2])+"\n")
    f.close()


def icp_svd(P, Q, iterations=10, kernel=lambda diff: 1.0):
    """Perform ICP using SVD."""
    center_of_Q, Q_centered = center_data(Q)

    P_values = [P.copy()]
    P_copy = P.copy()

    print("center of Q : \n", center_of_Q)

    for i in range(iterations):
        print("iter", i)
        center_of_P, P_centered = center_data(P_copy)
        print("center of P : \n", center_of_P)

        print("find_correspondances")
        correspondences = get_correspondence_indices(P_centered, Q_centered)

        #norm_values.append(np.linalg.norm(P_centered - Q_centered))
        print("covariance")

        cov, exclude_indices = compute_cross_covariance(P_centered, Q_centered, correspondences, kernel)
        U, S, V_T = np.linalg.svd(cov)
        R = U.dot(V_T)
        t = center_of_Q - R.dot(center_of_P)
        P_copy = R.dot(P_copy) + t

        err = compute_error(P_copy, Q, correspondences)

        print("err : ", err)
        if err < 0.00005:
            print("stop")
            break

    save(P_copy)
    return P_values


def read_file(filepath):
    f = open(filepath, "r")
    f.readline()
    res = []
    for line in f:
        coords = line.split(',')
        res.append([coords[0], coords[1], coords[2]])
    return np.array(res, dtype=float).transpose()

if __name__ == '__main__':
    P = read_file(sys.argv[2])
    Q = read_file(sys.argv[1])

    print(P.shape)
    print(Q.shape)

    icp_svd(P, Q)

