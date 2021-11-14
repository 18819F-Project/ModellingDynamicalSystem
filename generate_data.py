"""
Double pendulum motion simulation

"""
from __future__ import print_function

import sys
sys.path.append("./model")
sys.path.append("./dynamics")

from visualize import *
from dynamics.double_pendulum_ground_truth import *
from dynamics.ode_solver import *

import numpy as np

if __name__ == "__main__":
    m1 = 1  # mass of pendulum 1 (kg)
    m2 = 1  # mass of pendulum 2 (kg)
    L1 = 1  # length of pendulum 1 (m)
    L2 = 1  # length of pendulum 2 (m)
    g = 9.81  # gravity (m/s^2)

    # set up bounds for initial conditions
    # bounds are interpretted as +/-
    theta_bound = np.pi  # rad
    omega_bound = np.pi / 12  # rad/s

    # state_0 = np.array([-np.pi / 2.2, np.pi / 1.8, 0, 0])
    # state_0 = np.array([np.random.uniform(low=-theta_bound, high=theta_bound),
    #                     np.random.uniform(low=-theta_bound, high=theta_bound),
    #                     np.random.uniform(low=-omega_bound, high=omega_bound),
    #                     np.random.uniform(low=-omega_bound, high=omega_bound)])

    m_mat = np.array([m1, m2])
    L_mat = np.array([L1, L2])

    # tfinal = 1000.0  # Final time. Simulation time = 0 to tfinal.
    # dt = 0.05
    # t = np.arange(0, tfinal, dt)

    doublePendulumGroundTruth = GroundTruthDoublePendulum(m_mat, L_mat)
    # E_start = doublePendulumGroundTruth.lagrangian(state_0)

    # curr_state = state_0
    num_samples = 500000
    count = 0
    for i in range(num_samples):
        count += 1
        if count % 100 == 0:
            print(count)
        state = np.array([np.random.uniform(low=-theta_bound, high=theta_bound),
                          np.random.uniform(low=-theta_bound, high=theta_bound),
                          np.random.uniform(low=-omega_bound, high=omega_bound),
                          np.random.uniform(low=-omega_bound, high=omega_bound)])
        doublePendulumGroundTruth.dynamics(state)
        # E_curr = doublePendulumGroundTruth.lagrangian(curr_state)
        # if abs(E_curr / E_start) / E_start > 0.02:
        #     print("Energy unstable!")

    # coords = doublePendulumGroundTruth.polar_to_cartesian(doublePendulumGroundTruth.state_hist[:, 0],
    #                                                       doublePendulumGroundTruth.state_hist[:, 1])
    np.savez("dataset", input=doublePendulumGroundTruth.state_hist,
                        labels=doublePendulumGroundTruth.state_dot_hist)

    # gifMaker = GifMaker()
    # gifMaker.make_gif_dp(coords, dt)


