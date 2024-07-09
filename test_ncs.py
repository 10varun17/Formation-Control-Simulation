
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import formation_controller.networked_controllers as ncs
if __name__ == '__main__':
    w_nbrs = np.array([1.0, 0.5, 0.75])
    q_nbrs = np.array([[2.0, 2.0],
                      [0.0, 0.0],
                      [-0.5, 0.75]])
    q0_agent = np.array([1.8, -0.3])

    # Basic smoke test - should be driving agent toward the more heavily
    # weighted neighbor
    u1 = ncs.rendezvous(q0_agent, q_nbrs, w_nbrs)
    print(f"u: {u1}")

    # Simulation test
    dt = 0.0005
    t_f = 3.0
    q_agent_k = np.array(q0_agent)
    ts = np.arange(0, t_f, step=dt)
    qs = np.zeros((q_agent_k.shape[0], len(ts)))
    for i, _ in enumerate(ts):
        qs[:,i] = q_agent_k
        u = ncs.rendezvous(q_agent_k, q_nbrs, w_nbrs)
        q_agent_k1 = q_agent_k + np.array(u) * dt
        q_agent_k = q_agent_k1
    fig0 = plt.figure(0, (6,6))
    plt.scatter(q_nbrs[0][0], q_nbrs[0][1], marker='*')
    plt.scatter(q_nbrs[1][0], q_nbrs[1][1], marker='^')
    plt.scatter(q_nbrs[2][0], q_nbrs[2][1], marker='o')
    plt.scatter(qs[0,:], qs[1,:], c=ts, cmap='summer')
    plt.xlim(-0.6, 2.1)
    plt.ylim(-0.6, 2.1)
    plt.show()
    
    # Triangle formation smoke test
    q0_a1 = [0.0, 0.0]
    q0_a2 = [2.0, 2.0]
    q0_a3 = [2.0, -0.5]

    # Simulation test
    ndim = len(q0_a1)
    q0_agents = [q0_a1, q0_a2, q0_a3]
    # Assuming 2d agents
    qs_a1 = np.zeros((ndim, len(ts)))
    qs_a2 = np.zeros((ndim, len(ts)))
    qs_a3 = np.zeros((ndim, len(ts)))
    ds = np.array([np.array([0., 0.]), np.array([0.5, 0.866]), np.array([1.0, 0.0])])
    q_a1_k = np.array(q0_agents[0])
    q_a2_k = np.array(q0_agents[1])
    q_a3_k = np.array(q0_agents[2])
    for i, _ in enumerate(ts):
        # Compute the control for each agent
        qs_a1[:,i] = q_a1_k
        u1 = ncs.triangle_formation(q_a1_k, [q_a2_k, q_a3_k], ds[0], [ds[1], ds[2]])
        qs_a2[:,i] = q_a2_k
        u2 = ncs.triangle_formation(q_a2_k, [q_a1_k, q_a3_k], ds[1], [ds[0], ds[2]])
        qs_a3[:,i] = q_a3_k
        u3 = ncs.triangle_formation(q_a3_k, [q_a1_k, q_a2_k], ds[2], [ds[0], ds[1]])
        # Move them!
        q_a1_k1 = q_a1_k + np.array(u1) * dt
        q_a1_k = q_a1_k1
        q_a2_k1 = q_a2_k + np.array(u2) * dt
        q_a2_k = q_a2_k1
        q_a3_k1 = q_a3_k + np.array(u3) * dt
        q_a3_k = q_a3_k1

    fig1 = plt.figure(1, (6,6))
    plt.scatter(qs_a1[0,:], qs_a1[1,:], c=ts, cmap='summer')
    plt.scatter(qs_a2[0,:], qs_a2[1,:], c=ts, cmap='spring')
    plt.scatter(qs_a3[0,:], qs_a3[1,:], c=ts, cmap='winter')
    plt.xlim(-2, 5)
    plt.ylim(-2, 5)
    plt.show()
    print("Done!")