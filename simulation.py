import numpy as np
import pandas as pd


def eular_sim(B0, mu, sigma, X0, k, theta, eta, rho, T, N=251):
    dt = T/N

    B = np.ones(N+1) * B0
    X = np.ones(N+1) * X0
    A = np.ones(N+1) * B0 * np.exp(X0)
    cov = np.array([[1, rho], [rho, 1]])
    for n in range(1, N+1):
        # generate correlated standard normal with correlation rho
        Z, W = np.random.multivariate_normal([0, 0], cov)
        B[n] = B[n-1] * np.exp((mu-(sigma**2)/2) * dt + sigma*np.sqrt(dt)*Z)
        X[n] = theta * (1 - np.exp(-k*dt)) + np.exp(-k*dt) * X[n-1] + \
            np.sqrt(((eta**2)/(2*k))*(1 - np.exp(-2*k*dt)))*W
        A[n] = B[n] * np.exp(X[n])

    return (A, B, X)


def alpha_t(T, k, gamma, eta):
    def a(t):
        a1 = (k*(1-np.sqrt(1 - gamma)))/(2*(eta**2))
        a2 = 2*np.sqrt(1 - gamma)
        a3 = 1 - np.sqrt(1 - gamma) - (1 + np.sqrt(1 - gamma)) * \
            np.exp((2*k*(T - t))/(np.sqrt(1 - gamma)))
        return a1 * (1 + a2/a3)
    return a


def beta_t(T, k, gamma, eta, sigma, rho, theta):
    def b(t):
        b0 = np.exp((2*k*(T - t))/(np.sqrt(1-gamma)))
        b1 = 1/(2*(eta**2)*((1 - np.sqrt(1-gamma)) - (1 + np.sqrt(1-gamma))
                * b0))
        b2 = gamma * np.sqrt(1 - gamma) * ((eta**2) + 2 *
                                           rho * sigma * eta)*((1-b0)**2)
        b3 = gamma * ((eta**2) + 2 * rho * sigma *
                      eta + 2 * k * theta) * (1 - b0)
        return b1 * (b2 - b3)
    return b


def h_t(T, k, gamma, eta, sigma, rho, theta):
    a = alpha_t(T, k, gamma, eta)
    b = beta_t(T, k, gamma, eta, sigma, rho, theta)

    def h(t, x):
        h1 = 1/(1-gamma)
        h2 = (b(t) + 2 * x * a(t) - (k*(x - theta) /
              (eta**2)) + (rho * sigma)/(eta) + 1/2)
        return h1*h2
    return h


def portfolio(V0, B0, X0, r, T, k, gamma, eta, mu, sigma, rho, theta):
    h = h_t(T, k, gamma, eta, sigma, rho, theta)
    # V(t) daily rebalancing
    V_n = V0
    # Use 251 for yearly trading days, N is the total number of trading days in T years
    N = 251 * T
    dt = T / N
    A, B, X = eular_sim(B0, mu, sigma, X0, k, theta, eta, rho, T, N)
    for n in range(N):
        t = n * dt
        V_new = V_n + V_n * (h(t, X[n])/A[n]) * (A[n+1] - A[n]) - V_n * \
            (h(t, X[n])/B[n]) * (B[n+1] - B[n]) + V_n * r * dt
        V_n = V_new
    return V_n
