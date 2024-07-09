from typing import Sequence
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def rendezvous(q_agent: NDArray, q_nbrs: NDArray, w_nbrs: NDArray):
    pos_dim = len(q_agent)
    q_agent_dot = np.zeros((pos_dim,))

    for q_nbr, w_nbr in zip(q_nbrs, w_nbrs):
        q_agent_dot += w_nbr * (q_agent - q_nbr)

    return -q_agent_dot

def triangle_formation(q_agent : NDArray, q_nbrs : NDArray, z_ref : NDArray, z_nbrs : NDArray):
    """
    Implements a triangle formation controller using relative distance
    specifications.

    q_agent : Sequence
        Agents are represented in a real plane or volume (R^2, R^3).
    q_nbrs : Sequence[Sequence]
        All neighbors have the same dimension as q_agent.
    """
    agent_dim = len(q_agent)
    # Init the control inputs
    u = np.zeros((agent_dim,))
    for q_nbr, z_nbr in zip(q_nbrs, z_nbrs):
        diff = (q_agent - q_nbr) - (z_ref - z_nbr)
        u = u + diff
    return -u

def simulation(q_agent, q_nbrs, w_nbrs, dt, t_f):
    q_agent_k = np.array(q_agent)
    ts = np.arange(0, t_f, step = dt)
    qs = np.zeros((q_agent_k.shape[0], len(ts)))
    us = np.zeros((q_agent_k.shape[0], len(ts)))
    for i, _ in enumerate(ts):
        qs[:, i] = q_agent_k
        u = rendezvous(q_agent_k, q_nbrs, w_nbrs)
        us[:, i] = u
        q_agent_k1 = q_agent_k + np.array(u) * dt
        q_agent_k = q_agent_k1
    # fig0 = plt.figure(0, (6,6))
    plt.scatter(q_agent[0], q_agent[1], marker = 's')
    return qs, ts, us
