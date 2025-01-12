import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import qutip
import random

# Fix random seeds
np.random.seed(1)
random.seed(1)
qutip.settings.rand_seed = 1


# Helper functions
def runif_in_simplex(d):
    k = np.random.exponential(scale=1.0, size=d)
    return k / sum(k)

def generate_psd(D):
    qobj_density_matrix = qutip.rand_dm(D)
    return qobj_density_matrix.full()

def f(sigma, rhos, P, a, Diagonal=False): # fpetz
    ans = 0
    if(Diagonal): # If all matrices are diagonal (classical case), we can compute faster
        sigma_power = np.diag(sigma)**(1-a)
        for i in range(len(P)):
            ans += P[i] * np.log(np.dot(np.diag(rhos[i]),sigma_power))
    else:  
        sigma_power = linalg.fractional_matrix_power(sigma,1-a) # O(D^3) Precompute power of sigma
        for i in range(len(P)):
            ans += P[i] * np.log(np.tensordot(rhos[i].T,sigma_power))
    return ans / (a - 1)

def h(x, y, a):
   if x == y:
       return (1 - a) * x**(-a)
   return (x**(1 - a) - y**(1 - a)) / (x - y)

def gradf(X, rhos, P, a, Diagonal=False): # In case you want to compute gradient of f
    D = X.shape[0]
    if(Diagonal): # If all matrices are diagonal (classical case), we can compute faster
        tmp = SimpleIteration(X, rhos, P, a, Diagonal)
        tmp = np.diag(tmp)**a
        ans = tmp * (np.diag(X)**(-a))
        ans = -np.diag(ans)
    else:
        tmp = SimpleIteration(X, rhos, P, a)
        tmp = linalg.fractional_matrix_power(tmp, a)
        ans = np.zeros((D, D))
        eig_vals, eig_vecs = linalg.eigh(X)

        for i in range(D):
            for k in range(D):
                ei, ek = eig_vals[i], eig_vals[k]
                ui, uk = np.reshape(eig_vecs[:, i], (D, 1)), np.reshape(eig_vecs[:, k], (D, 1))
                ui, uk = np.matrix(ui), np.matrix(uk)
                ans += (h(ei, ek, a) / (a - 1)) * ((ui @ ((ui.H @ tmp) @ uk) @ uk.H))
    return ans
def Polyak(rhos,P,a, Diagonal=False):
    delta_t, delta, gamma, beta, c = 2.5, 1e-11, 1.25, 0.75, 0.1
    T = 30
    D = rhos[0].shape[0]
    sigma_t = np.identity(D) / D
    min_f = float('inf')
    if(Diagonal): # If all matrices are diagonal (classical case), we can compute faster
        f_val = f(sigma_t,rhos,P,a,Diagonal)
        for t in range(T):
            
            min_f = min(f_val,min_f)
            ft_tilde = min_f - delta_t
            grad =  gradf(sigma_t,rhos,P,a,Diagonal)
            if(t%(T//10)==0):
                print("polyak",f_val,np.diag(sigma_t))
            grad_norm_square = np.tensordot(grad.T,grad)
            eta_t = (f_val-ft_tilde)/(c*grad_norm_square)
            sigma_t = np.diag(sigma_t)*np.exp(-eta_t*np.diag(grad))
            sigma_t = np.diag(sigma_t)
            sigma_t /= np.trace(sigma_t)
            f_val = f(sigma_t,rhos,P,a,Diagonal=True)
            if(f_val<=ft_tilde):
                delta_t *= gamma
            else:
                delta_t = max(beta*delta_t,delta)
    else:
        f_val = f(sigma_t,rhos,P,a)
        for t in range(T):
            min_f = min(f_val,min_f)
            ft_tilde = min_f - delta_t
            grad = gradf(sigma_t,rhos,P,a)
            grad_norm_square = np.tensordot(grad.T,grad)
            eta_t = (f_val-min_f)/(c*grad_norm_square)
            sigma_t = linalg.expm(linalg.logm(sigma_t)-eta_t*grad)
            sigma_t /= np.trace(sigma_t)
            f_val = f(sigma_t,rhos,P,a)
            if(f_val<=ft_tilde):
                delta_t *= gamma
            else:
                delta_t = max(beta*delta_t,delta)
    return min_f


def SimpleIteration(sigma, rhos, P, a, Diagonal=False):
    D = sigma.shape[0]
    if(Diagonal): # If all matrices are diagonal (classical case), we can compute faster
        ans = np.zeros(D)
        sigma_power=np.diag(sigma)**(1-a)
        for i in range(len(P)):
            ans += P[i] * np.diag(rhos[i])/np.dot(np.diag(rhos[i]),sigma_power)
        ans = ans**(1/a)
        ans = np.diag(ans)

            
    else:   
        ans = np.zeros((D, D))
        sigma_power = linalg.fractional_matrix_power(sigma,1-a) # O(D^3) precompute sigma's power
        for i in range(len(P)): # O(N)
            ans = ans + P[i] * rhos[i] / np.tensordot(rhos[i].T,sigma_power) # O(D^2)
        ans = linalg.fractional_matrix_power(ans, 1/a) #O(D^3)
    return ans

# Experiment parameters
D = 2**7 # Quantum state dimension
N = 2**5 # Cardinality of alphabet


global_rhos = []
for i in range(N):
    rho = generate_psd(D)
    rho /= np.trace(rho)
    global_rhos.append(rho)

alphas = [0.8,1.5,3,5]
P = runif_in_simplex(N)
# Loop over different values of alpha
for alpha in alphas:
    rhos = []
    for i in range(N): # Total O(N * D^3) time for initialization
        rhos.append(linalg.fractional_matrix_power(global_rhos[i],alpha)) # Taking matrix power of supported states beforehand
    
    sigma = np.identity(D) / D
    # Arrays to store iteration results
    iterations = [0]
    f_values = [f(sigma / np.trace(sigma), rhos, P, alpha)]

    # Perform the iterative process
    T = 30
    for i in range(T):
        sigma = SimpleIteration(sigma, rhos, P, alpha)
        iterations.append(i+1)
        f_val = f(sigma / np.trace(sigma), rhos, P, alpha)
        f_values.append(f_val)
        print(alpha,f_val)

    last_f_val = f_values[-1] # Last iterate is guaranteed to converge.
    for i in range(len(f_values)):
        f_values[i] = abs(f_values[i]-last_f_val) # Calculate the distance instead.

    # Plotting the results with log-scale y-axis
    plt.figure(figsize=(6, 5))

    # f(X) vs iteration with log-scale y-axis
    # plt.subplot(1, 2, 1)
    plt.plot(iterations, f_values, label=f'f(X), alpha={alpha}')
    plt.xlabel('Number of iterations',fontsize=20)
    plt.ylabel('Approx. optimization error',fontsize=20)
    plt.yscale('log')  # Set log scale for y-axis
    plt.ylim(1e-11,1e-1)
    plt.xlim(0, 13)
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



# Outside the range guaranteed to converge...
N=3
D=3
# An instance that cause our proposed algorithm to fail when alpha < 0.5
global_rhos = [
    [
        [0.6,0,0],
        [0,0.36,0],
        [0,0,0.04]
    ],
    [
        [0.21,0,0],
        [0,0.7,0],
        [0,0,0.09]
    ],
    [
        [0.04,0,0],
        [0,0.16,0],
        [0,0,0.8]
    ]

]
for i in range(N):
    #rho = generate_psd(D)
    #rho /= np.trace(rho)
    global_rhos[i]=np.array(global_rhos[i])
    global_rhos[i]/=np.trace(global_rhos[i])

alphas = [0.2,0.4]
P = np.array([1/3,1/3,1/3])
# Loop over different values of alpha
for alpha in alphas:
    rhos = []
    for i in range(N): # Total O(N * D^3) time for initialization
        rhos.append(np.diag(np.diag(global_rhos[i])**alpha)) # Taking matrix power of supported states beforehand
    
    sigma = np.identity(D) / D
    # Arrays to store iteration results
    iterations = [0]
    f_values = [f(sigma, rhos, P, alpha)]

    # Perform the iterative process
    T = 30
    for i in range(T):
        sigma = SimpleIteration(sigma, rhos, P, alpha,Diagonal=True)
        iterations.append(i+1)
        f_val = f(sigma, rhos, P, alpha,Diagonal=True)
        f_values.append(f_val)
        print(alpha,f_val,np.diag(sigma))

    #min_f_val = min(f_values)
    min_f_val = Polyak(rhos, P, alpha,Diagonal=True) # Use mirror descent with Polyak stepsize to compute
                                       # minimum value instead. As our algorithm is not 
                                       # proved to converge for alpha < 0.5
    print("Minimum computed by Polyak",min_f_val)
    for i in range(len(f_values)):
        f_values[i] = abs(f_values[i]-min_f_val) # Compute the distance instead.
    # f_values = f_values[:T//10]
    # norm_logs = norm_logs[:T//10]
    # iterations = iterations[:T//10]

    # Plotting the results with log-scale y-axis
    plt.figure(figsize=(6, 5))

    # f(X) vs iteration with log-scale y-axis
    # plt.subplot(1, 2, 1)
    plt.plot(iterations, f_values, label=f'f(X), alpha={alpha}')
    plt.xlabel('Number of iterations',fontsize=20)
    plt.ylabel('Approx. optimization error',fontsize=20)
    plt.yscale('log')  # Set log scale for y-axis
    plt.ylim(1e-11,1e-1)
    plt.xlim(0, 13)
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









