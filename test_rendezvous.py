import numpy as np
import matplotlib.pyplot as plt
import self_networked_controllers as ncs

# Basic test to drive the agent towards the heavily weighted neighbor
def test_basic():
    w_nbrs = np.array([1.0, 0.5, 0.75])
    q_nbrs = np.array([[2.0, 2.0],
                    [1.0, 0.0],
                    [-0.5, 0.75]])
    q0_agent = np.array([1.1, -0.3])

    qs, ts, us = ncs.simulation(q0_agent, q_nbrs, w_nbrs, 0.05, 3.0)
    # fig0 = plt.figure(0, figsize=(6,6))
    plt.scatter(q_nbrs[:, 0], q_nbrs[:, 1], marker="*")
    plt.scatter(qs[0, :], qs[1, :], c=ts, cmap="summer")
    plt.show()

# Convergence to the centroid test
def test_converge_centroid():
    # Agents
    q0 = np.array([1.0, 2.0])
    q1 = np.array([-1.2, 0.3])
    q2 = np.array([0.4, -0.9])
    q3 = np.array([2.4, 1.0])
    q4 = np.array([1.4, -0.5])

    # 2D Array of all agents
    qs = np.array([q0, q1, q2, q3, q4])

    # Neighbors of each agent
    q_nbrs0 = np.array([q4, q1, q2, q3])
    q_nbrs1 = np.array([q4, q0, q2, q3])
    q_nbrs2 = np.array([q4, q0, q1, q3])
    q_nbrs3 = np.array([q4, q0, q1, q2])
    q_nbrs4 = np.array([q0, q1, q2, q3])

    # 2D Array containing neighbors of each agent
    q_nbrs_all = np.array([q_nbrs0, q_nbrs1, q_nbrs2, q_nbrs3, q_nbrs4])

    # Weights of neighbors
    w_nbrs0 = np.array([2.0, 2.0, 2.0, 2.0])
    w_nbrs1 = np.array([2.0, 2.0, 2.0, 2.0])
    w_nbrs2 = np.array([2.0, 2.0, 2.0, 2.0])
    w_nbrs3 = np.array([2.0, 2.0, 2.0, 2.0])
    w_nbrs4 = np.array([2.0, 2.0, 2.0, 2.0])

    # 2D Array containing weights of neighbors of each agent
    w_nbrs_all = np.array([w_nbrs0, w_nbrs1, w_nbrs2, w_nbrs3, w_nbrs4])

    # List to add the positions of each agent's motion after the defined time interval
    qs_s = []

    # Time at which the position of each agent's location was captured
    ts_s = []

    # Array of us
    us_s = []

    # Get the simulation data
    for q, q_nbrs, w_nbrs in zip(qs, q_nbrs_all, w_nbrs_all):
        qs, ts, us = ncs.simulation(q, q_nbrs, w_nbrs, 0.005, 0.2)
        qs_s.append(qs)
        ts_s.append(ts)
        us_s.append(us)
    
    # Plot the data
    for qs, ts, us in zip(qs_s, ts_s, us_s):
        plt.scatter(qs[0,:], qs[1,:], c=ts, cmap='spring', alpha=0.4)
    
    plt.show()

# Convergence to a point with two agents
def test_converge_point_two_agents(x, y):
    """
    x: x-coord
    y: y-coord
    """
    q0 = np.array([2, 2.5])
    q1 = np.array([-1.6, 0.8])
    q_ref = np.array([3*x - q0[0] - q1[0], 3*y - q0[1] - q1[1]])
    qs = np.array([q0, q1])

    q_nbrs0 = np.array([q1, q_ref])
    q_nbrs1 = np.array([q0, q_ref])
    q_nbrs_all = np.array([q_nbrs0, q_nbrs1])

    w_nbrs0 = np.array([0.75, 0.75])
    w_nbrs1 = np.array([0.75, 0.75])
    w_nbrs_all = np.array([w_nbrs0, w_nbrs1])

    qs_s = []
    ts_s = []
    us_s = []

    for q, q_nbrs, w_nbrs in zip(qs, q_nbrs_all, w_nbrs_all):
        qs, ts, us = ncs.simulation(q, q_nbrs, w_nbrs, 0.005, 0.72)
        qs_s.append(qs)
        ts_s.append(ts)
        us_s.append(us)
    
    for qs, ts, us in zip(qs_s, ts_s, us_s):
        plt.scatter(qs[0, :], qs[1, :], c=ts, cmap='winter')
        plt.scatter(us[0,:], us[1,:], c=ts, cmap='spring')

    plt.scatter(x, y, c='r', marker="*")
    plt.show()

# General convergence test
def test_converge(x, y):
    q0 = np.array([1.0, 3.0])
    q1 = np.array([-2.1, 0.8])
    q2 = np.array([-1.9, -2.5])
    q_ref = np.array([x, y])
    qs = np.array([q0, q1, q2])

    q_nbrs0 = np.array([q1, q2, q_ref])
    q_nbrs1 = np.array([q0, q2, q_ref])
    q_nbrs2 = np.array([q0, q1, q_ref])
    q_nbrs_all = np.array([q_nbrs0, q_nbrs1, q_nbrs2])

    w_nbrs0 = np.array([0.6, 0.4, 25.0])
    w_nbrs1 = np.array([0.6, 0.7, 25.0])
    w_nbrs2 = np.array([0.4, 0.7, 25.0])
    w_nbrs_all = np.array([w_nbrs0, w_nbrs1, w_nbrs2])

    qs_s = []
    ts_s = []
    us_s = []

    for q, q_nbrs, w_nbrs in zip(qs, q_nbrs_all, w_nbrs_all):
        qs, ts, us = ncs.simulation(q, q_nbrs, w_nbrs, 0.005, 0.19)
        qs_s.append(qs)
        ts_s.append(ts)
        us_s.append(us)
    
    for qs, ts, us in zip(qs_s, ts_s, us_s):
        plt.scatter(qs[0, :], qs[1, :], c=ts, cmap='spring')
        plt.scatter(us[0,:], us[1,:], c=ts, cmap='winter')
    
    plt.show()

if __name__ == "__main__":
    test_converge_centroid()