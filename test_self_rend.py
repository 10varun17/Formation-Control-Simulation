import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import formation_controller.self_networked_controllers as ncs

q_agent = np.array([10, 12])
q_nbrs = np.array([
    [5,6],
    [15,6]
])
w_nbrs = np.array([0.5, 0.7])
print(ncs.rendezvous(q_agent, q_nbrs, w_nbrs))

