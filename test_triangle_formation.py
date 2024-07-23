import numpy as np
import matplotlib.pyplot as plt
import self_networked_controllers as ncs
import cmath

q0 = np.array([-1.0, -1.5])
q1 = np.array([4.0, 2.0])
q2 = np.array([5.0, -0.5])

z0 = 2 * cmath.exp((cmath.pi) * (1j)/2)
z1 = 2 * cmath.exp((7 * cmath.pi) * (1j)/6)
z2 = 2 * cmath.exp((11 * cmath.pi) * (1j)/6)

def test_triangle_formation(q0, q1, q2, dt, t_f):
    ts = np.arange(0, t_f, step=dt)
    q_agents = np.array([q0, q1, q2])
    pos_dim = len(q0)

    qs0 = np.zeros((pos_dim, len(ts)))
    qs1 = np.zeros((pos_dim, len(ts)))
    qs2 = np.zeros((pos_dim, len(ts)))

    q0_k = np.array([q_agents[0]])
    q1_k = np.array([q_agents[1]])
    q2_k = np.array([q_agents[2]])

    ds = np.array([
        np.array([z0.real, z0.imag]),
        np.array([z1.real, z1.imag]),
        np.array([z2.real, z2.imag]),
        # np.array([z3.real, z3.imag])
    ])

    for i, _ in enumerate(ts):
        qs0[:, i] = q0_k
        u0 = ncs.triangle_formation(q0_k, [q1_k, q2_k], ds[0], [ds[1], ds[2]])
        q0_k1 = q0_k + np.array(u0) * dt
        q0_k = q0_k1

        qs1[:, i] = q1_k
        u1 = ncs.triangle_formation(q1_k, [q0_k, q2_k], ds[1],  [ds[0], ds[2]])
        q1_k1 = q1_k + np.array(u1) * dt
        q1_k = q1_k1

        qs2[:, i] = q2_k
        u2 = ncs.triangle_formation(q2_k, [q0_k, q1_k], ds[2], [ds[0], ds[1]])
        q2_k1 = q2_k + np.array(u2) * dt
        q2_k = q2_k1

    fig = plt.figure(figsize=(6,6))
    plt.scatter(qs0[0, :], qs0[1, :], c = ts, cmap='summer')
    plt.scatter(qs1[0, :], qs1[1, :], c = ts, cmap='spring')
    plt.scatter(qs2[0, :], qs2[1, :], c = ts, cmap='winter')
    plt.xlim(-3, 6)
    plt.ylim(-3, 6)
    plt.show()

if __name__ == "__main__":
    test_triangle_formation(q0, q1, q2, 0.0005, 3.0)
