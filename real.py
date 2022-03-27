import numpy as np
import pandas as pd


def m_hat(S):
    N = len(S) - 1
    return (S[N] - S[0])/N


def S_squared_hat(S):
    m_h = m_hat(S)
    N = len(S) - 1
    num = 0
    for t in range(N):
        num = num + (S[t + 1] - S[t])**2

    num = num + 2 * m_h * (S[N] - S[0]) + N * (m_h**2)
    return num / N


def p_hat(X):
    N = len(X) - 1
    p1 = 1/(N * np.sum(X[0:-1]**2) - np.sum(X[0:-1])**2)
    p2 = (N * np.dot(X[1:], X[:-1]) - (X[-1] - X[0])
          * np.sum(X[0:-1]) - np.sum(X[0:-1])**2)
    return p1 * p2


def q_hat(X):
    p_h = p_hat(X)
    N = len(X) - 1
    num = X[N] - X[0] + np.sum(X[:-1]) - p_h * np.sum(X[:-1])
    return num / N


def V_squared_hat(X):
    p_h = p_hat(X)
    q_h = q_hat(X)
    N = len(X) - 1
    num = X[N]**2 - X[0]**2 + (1 + p_h**2) * np.sum(X[:-1]**2) - \
        2 * p_h * np.dot(X[1:], X[:-1]) - N * q_h
    return num / N


def C_hat(X, S):
    N = len(X) - 1
    p_h = p_hat(X)
    m_h = m_hat(S)
    V_h = np.sqrt(V_squared_hat(X))
    S_h = np.sqrt(S_squared_hat(S))
    num1 = np.dot(X[1:], S[1:] - S[:-1])
    num2 = p_h * np.dot(X[:-1], S[1:] - S[:-1])
    num3 = m_h * (X[N] - X[0])
    num4 = m_h * (1 - p_h) * np.sum(X[:-1])
    den = N * V_h * S_h
    return (num1 - num2 - num3 - num4) / den


def sigma_hat(S, dt):
    S_h = np.sqrt(S_squared_hat(S))
    return np.sqrt(S_h / dt)


def mu_hat(S, dt):
    m_h = m_hat(S)
    sigma_h = sigma_hat(S, dt)
    return m_h/dt + (sigma_h**2) / 2


def k_hat(X, dt):
    p_h = p_hat(X)
    return -(np.log(p_h)/dt)


def theta_hat(X):
    q_h = q_hat(X)
    p_h = p_hat(X)
    return q_h / (1 - p_h)


def eta_hat(X, dt):
    k_h = k_hat(X, dt)
    V_sq_h = V_squared_hat(X)
    p_h = p_hat(X)
    return np.sqrt((2 * k_h * V_sq_h)/(1 - p_h**2))


def rho_hat(X, S, dt):
    k_h = k_hat(X, dt)
    C_h = C_hat(X, S)
    V_h = np.sqrt(V_squared_hat(X))
    S_h = np.sqrt(S_squared_hat(S))
    eta_h = eta_hat(X, dt)
    sigma_h = sigma_hat(S, dt)
    p_h = p_hat(X)
    num = k_h * C_h * V_h * S_h
    den = eta_h * sigma_h * (1 - p_h)
    return num/den
