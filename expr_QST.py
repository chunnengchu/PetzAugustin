import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import qutip
# Helper functions
def runif_in_simplex(d):
    k = np.random.exponential(scale=1.0, size=d)
    return k / sum(k)

def generate_psd(D):
    qobj_density_matrix = qutip.rand_dm(D)
    return qobj_density_matrix.full()

def f(sigma, rhos, P, a):
    ans = 0
    for i in range(len(P)):
        ans += P[i] * np.log(np.trace(rhos[i] @ linalg.fractional_matrix_power(sigma, 1-a)))
    return ans / (a - 1)

def h(x, y, a):
    if x == y:
        return (1 - a) * x**(-a)
    return (x**(1 - a) - y**(1 - a)) / (x - y)

# def gradf(X, As, P, a):
#     D = X.shape[0]
#     tmp = step(X, As, P, a)
#     tmp = linalg.fractional_matrix_power(tmp, a)
#     ans = np.zeros((D, D))
#     eig_vals, eig_vecs = linalg.eigh(X)
#     rX = np.zeros((D, D))

#     for j in range(len(P)):
#         for i in range(D):
#             for k in range(D):
#                 ei, ek = eig_vals[i], eig_vals[k]
#                 ui, uk = np.reshape(eig_vecs[:, i], (D, 1)), np.reshape(eig_vecs[:, k], (D, 1))
#                 ui, uk = np.matrix(ui), np.matrix(uk)
#                 ans += (P[j] * h(ei, ek, a) / (a - 1)) * ((ui @ ui.H) @ tmp @ (uk @ uk.H))
#             rX += P[j] * ei * (ui @ ui.H)
#     return ans

def step(sigma, rhos, P, a):
    D = sigma.shape[0]
    ans = np.zeros((D, D))
    for i in range(len(P)):
        ans = ans + P[i] * rhos[i] / np.trace(rhos[i] @ linalg.fractional_matrix_power(sigma, 1-a))
    ans = linalg.fractional_matrix_power(ans, 1/a)
    return ans

# Experiment parameters
D = 2**7
m = 2**5
alphas = [0.8, 1.1, 1.4, 1.7, 2]


global_rhos = []
for i in range(m):
    rho = generate_psd(D)
    rho /= np.trace(rho)
    global_rhos.append(rho)

alphas = [0.8,1.1,1.4,1.7,2]
P = runif_in_simplex(m)
# Loop over different values of alpha
for alpha in alphas:
    rhos = []
    for i in range(m):
        rhos.append(linalg.fractional_matrix_power(global_rhos[i],alpha)) # Taking matrix power of supported states beforehand
    
    sigma = np.identity(D) / D
    # Arrays to store iteration results
    iterations = [0]
    f_values = [f(sigma / np.trace(sigma), rhos, P, alpha)]

    # Perform the iterative process
    T = 10
    for i in range(T):
        sigma = step(sigma, rhos, P, alpha)
        iterations.append(i+1)
        f_val = f(sigma / np.trace(sigma), rhos, P, alpha)
        f_values.append(f_val)
        print(alpha,f_val)

    min_f_val = min(f_values)
    for i in range(len(f_values)):
        f_values[i] -= min_f_val
    # f_values = f_values[:T//10]
    # norm_logs = norm_logs[:T//10]
    # iterations = iterations[:T//10]

    # Plotting the results with log-scale y-axis
    plt.figure(figsize=(6, 5))

    # f(X) vs iteration with log-scale y-axis
    # plt.subplot(1, 2, 1)
    plt.plot(iterations, f_values, label=f'f(X), alpha={alpha}')
    plt.xlabel('Iteration',fontsize=20)
    plt.ylabel('f(X/tr(X))-f*',fontsize=20)
    plt.yscale('log')  # Set log scale for y-axis
    plt.ylim(1e-11,1e-3)
    plt.tick_params(axis='both',which='major',labelsize=12)
    plt.grid(True)

    # # Log of norm vs iteration with log-scale y-axis
    # plt.subplot(1, 2, 2)
    # plt.plot(iterations, norm_logs, label=f'log(norm), alpha={alpha}', color='red')
    # plt.xlabel('Iteration')
    # plt.ylabel('log(norm)')
    # plt.yscale('log')  # Set log scale for y-axis
    # plt.title(f'Log of log(op-norm(-grad)) vs Iteration for alpha={alpha}')
    # plt.grid(True)

    plt.tight_layout()

    # Save the figure with the name based on alpha
    plt.savefig(f'figure_alpha_{alpha}.png')

    # Show the plot (optional, you can comment this out if you don't want to display the figures)
    plt.show()









