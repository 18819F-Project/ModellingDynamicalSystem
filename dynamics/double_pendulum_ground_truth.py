
import numpy as np


class GroundTruthDoublePendulum:
    """
    A class to represent a ground truth representation of a double pendulum
    and all of its system properties
    """

    def __init__(self, m_matrix, l_vec):
        self.m = m_matrix
        self.lengths = l_vec
        self.g = 9.81
        self.state_hist = np.empty((0, 4))  # input
        self.state_dot_hist = np.empty((0, 4))  # label

    def dynamics(self, state):
        """
        :param state: [[theta1],
                       [theta2],
                       [theta1_dot],
                       [theta2_dot]]
               t: time_value (s)
        :return state_dot:
        """
        self.state_hist = np.vstack((self.state_hist, state))

        state_dot = np.zeros(4)

        c = np.cos(state[0] - state[1])  # intermediate variables
        s = np.sin(state[0] - state[1])  # intermediate variables

        state_dot[0] = state[2]  # d(theta 1)
        state_dot[1] = state[3]  # d(theta 2)
        state_dot[2] = (self.m[1] * self.g * np.sin(state[1]) * c - self.m[1] * s * (self.lengths[0] * c * state[2] ** 2 + self.lengths[1] * state[3] ** 2)
                        - (self.m[0] + self.m[1]) * self.g * np.sin(state[0])) / (self.lengths[0] * (self.m[0] + self.m[1] * s ** 2))
        state_dot[3] = ((self.m[0] + self.m[1]) * (self.lengths[0] * state[2] ** 2 * s - self.g * np.sin(state[1]) + self.g * np.sin(state[0]) * c)
                        + self.m[1] * self.lengths[1] * state[3] ** 2 * s * c) / (self.lengths[1] * (self.m[0] + self.m[1] * s ** 2))

        self.state_dot_hist = np.vstack((self.state_dot_hist, state_dot))

        return state_dot

    def polar_to_cartesian(self, theta1, theta2):
        # Mapping from polar to Cartesian
        x1 = self.lengths[0] * np.sin(theta1)  # First Pendulum
        y1 = -self.lengths[0] * np.cos(theta1)

        x2 = x1 + self.lengths[1] * np.sin(theta2)  # Second Pendulum
        y2 = y1 - self.lengths[1] * np.cos(theta2)

        return zip(x1, y1, x2, y2)

    def lagrangian(self, state):
        """
        :param state: [[theta1],
                       [theta2],
                       [theta1_dot],
                       [theta2_dot]]
        :return lagrangian:
        """

        link1_vel = (self.lengths[0]**2)*(state[2]**2)
        T_kinetic_link1 = 1/2*self.m[0]*link1_vel
        link2_vel = link1_vel + (self.lengths[1]**2)*(state[3]**2) \
                    + 2*self.lengths[0]*self.lengths[1]*state[2]*state[3]*np.cos(state[0]-state[1])
        T_kinetic_link2 = 1/2*self.m[1]*link2_vel

        T_kinetic = T_kinetic_link1 + T_kinetic_link2

        V_potential = -(self.m[0] + self.m[1])*self.g*self.lengths[0]*np.cos(state[0]) \
                      - self.m[1]*self.g*self.lengths[1]*np.cos(state[1])

        lagrangian = T_kinetic + V_potential

        return lagrangian

