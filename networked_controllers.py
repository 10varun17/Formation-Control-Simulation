from typing import Sequence
import operator
import numpy as np
from numpy.typing import NDArray

def rendezvous(q_agent : NDArray, q_nbrs : NDArray, w_nbrs : NDArray):
    """
    Implements the basic rendezvous equation for one agent assuming linear dynamics.

    q_agent : NDArray
        Agents are represented in a real plane or volume (R^2, R^3).
    q_nbrs : NDArray
        All neighbors have the same dimension as q_agent.
    w_nbrs : NDArray
        The weight between q_agent and each ith neighbor, e.g. w_nbrs[0]
        is the weight on this edge: q_agent---w[0]---q_nbrs[0].
    """
    agent_dim = len(q_agent)
    # Init the control inputs
    u = np.zeros((agent_dim,))
    for q_nbr, w_nbr in zip(q_nbrs, w_nbrs):
        diff = w_nbr * (q_agent - q_nbr)
        u = u + diff
    return -u

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