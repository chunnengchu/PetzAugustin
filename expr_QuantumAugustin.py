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

def ThompSonMetric(U,V):
    U_inv_sqrt = linalg.fractional_matrix_power(U, -0.5)
    V_inv_sqrt = linalg.fractional_matrix_power(V, -0.5)
    val1 = linalg.norm(U_inv_sqrt @ V @ U_inv_sqrt, 2)
    val2 = linalg.norm(V_inv_sqrt @ U @ V_inv_sqrt, 2)
    return np.log(max(val1, val2))

def F(Q, As, w, a, Diagonal=False): # fpetz
    ans = 0
    if(Diagonal): # If all matrices are diagonal (classical case), we can compute faster
        Q_power = np.diag(Q)**(1-a)
        for i in range(len(w)):
            ans += w[i] * np.log(np.dot(np.diag(As[i]),Q_power))
    else:  
        Q_power = linalg.fractional_matrix_power(Q,1-a) # O(D^3) Precompute power of Q
        for i in range(len(w)):
            ans += w[i] * np.log(np.tensordot(As[i].T,Q_power))
    return ans / (a - 1)

def h(x, y, a):
   if x == y:
       return (1 - a) * x**(-a)
   return (x**(1 - a) - y**(1 - a)) / (x - y)

def gradF(Q, As, w, a, Diagonal=False): # In case you want to compute gradient of f
    D = Q.shape[0]
    if(Diagonal): # If all matrices are diagonal (classical case), we can compute faster
        tmp = SimpleIteration(Q, As, w, a, Diagonal)
        tmp = np.diag(tmp)**a
        ans = tmp * (np.diag(Q)**(-a))
        ans = -np.diag(ans)
    else:
        tmp = SimpleIteration(Q, As, w, a)
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
def Polyak(As,w,a, Diagonal=False):
    delta_t, delta, gamma, beta, c = 2.5, 1e-11, 1.25, 0.75, 0.05
    T = 1000
    D = As[0].shape[0]
    Q_t = np.identity(D) / D
    min_F = float('inf')
    Q_star = Q_t
    if(Diagonal): # If all matrices are diagonal (classical case), we can compute faster
        F_val = F(Q_t,As,w,a,Diagonal)
        for t in range(T):
            if(F_val < min_F):
                min_F = F_val
                Q_star = Q_t
            Ft_tilde = min_F - delta_t
            grad =  gradF(Q_t,As,w,a,Diagonal)
            if(t%(T//10)==0):
                print("polyak",F_val,np.diag(Q_t))
            grad_norm_square = np.tensordot(grad.T,grad)
            eta_t = (F_val-Ft_tilde)/(c*grad_norm_square)
            Q_t = np.diag(Q_t)*np.exp(-eta_t*np.diag(grad))
            Q_t = np.diag(Q_t)
            Q_t /= np.trace(Q_t)
            F_val = F(Q_t,As,w,a,Diagonal=True)
            if(F_val<=Ft_tilde):
                delta_t *= gamma
            else:
                delta_t = max(beta*delta_t,delta)
    else:
        F_val = F(Q_t,As,w,a)
        for t in range(T):
            if(F_val < min_F):
                min_F = F_val
                Q_star = Q_t
            Ft_tilde = min_F - delta_t
            grad = gradF(Q_t,As,w,a)
            grad_norm_square = np.tensordot(grad.T,grad)
            eta_t = (F_val-min_F)/(c*grad_norm_square)
            Q_t = linalg.expm(linalg.logm(Q_t)-eta_t*grad)
            Q_t /= np.trace(Q_t)
            F_val = F(Q_t,As,w,a)
            if(F_val<=Ft_tilde):
                delta_t *= gamma
            else:
                delta_t = max(beta*delta_t,delta)
    return min_F, Q_star

def SimpleIteration(Q, As, w, a, Diagonal=False):
    D = Q.shape[0]
    if(Diagonal): # If all matrices are diagonal (classical case), we can compute faster
        ans = np.zeros(D)
        Q_power=np.diag(Q)**(1-a)
        for i in range(len(w)):
            ans += w[i] * np.diag(As[i])/np.dot(np.diag(As[i]),Q_power)
        ans = ans**(1/a)
        ans = np.diag(ans)

            
    else:   
        ans = np.zeros((D, D))
        Q_power = linalg.fractional_matrix_power(Q,1-a) # O(D^3) precompute sigma's power
        for i in range(len(w)): # O(N)
            ans = ans + w[i] * As[i] / np.tensordot(As[i].T,Q_power) # O(D^2)
        ans = linalg.fractional_matrix_power(ans, 1/a) #O(D^3)
    return ans

# Experiment parameters
D = 2**7 # Quantum state dimension
N = 2**5 # Cardinality of alphabet


global_As = []
for i in range(N):
    Ai = generate_psd(D)
    Ai /= np.trace(Ai)
    global_As.append(Ai)

alphas = [0.8,1.5,3,5]
w = runif_in_simplex(N)
# Loop over different values of alpha
for alpha in alphas:
    As = []
    for i in range(N): # Total O(N * D^3) time for initialization
        As.append(linalg.fractional_matrix_power(global_As[i],alpha)) # Taking matrix power of supported states beforehand
    
    Q = np.identity(D) / D
    # Arrays to store iteration results
    iterations = [0]
    iterates = [Q]
    F_values = [F(Q / np.trace(Q), As, w, alpha)]

    # Perform the iterative process
    T = 60
    for i in range(T):
        Q = SimpleIteration(Q, As, w, alpha)
        iterations.append(i+1)
        F_val = F(Q / np.trace(Q), As, w, alpha)
        F_values.append(F_val)
        iterates.append(Q / np.trace(Q))
        print(alpha,F_val)

    last_F_val = F_values[-1] # Last iterate is guaranteed to converge.
    distances_to_opt = []
    last_iterate_power = linalg.fractional_matrix_power(iterates[-1],1-alpha)
    for i in range(len(F_values)):
        F_values[i] = F_values[i]-last_F_val 
        distances_to_opt.append(ThompSonMetric(linalg.fractional_matrix_power(iterates[i],1-alpha),last_iterate_power))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Optimization error
    axs[0].plot(iterations, F_values, label='Optimization error')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Number of iterations', fontsize=14)
    axs[0].set_ylabel('Approx. optimization error', fontsize=14)
    axs[0].set_title(f'α = {alpha}', fontsize=14)
    axs[0].set_xlim(0, 30)
    axs[0].set_ylim(1e-11, 1e1)
    axs[0].grid(True)
    axs[0].tick_params(labelsize=12)

    # Right: Distance to optimality
    axs[1].plot(iterations, distances_to_opt, label='Distance to $Q_\star$')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Number of iterations', fontsize=14)
    axs[1].set_ylabel('Approx. iter. error', fontsize=14)
    axs[1].set_title(f'α = {alpha}', fontsize=14)
    axs[1].set_xlim(0, 30)
    axs[1].set_ylim(1e-11, 1e1)
    axs[1].grid(True)
    axs[1].tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f'figure_alpha_{alpha}.png')
    plt.show()



# Outside the range guaranteed to converge...
N=3
D=3
# An instance that cause our proposed algorithm to fail when alpha < 0.5
global_As = [
    [
        [0.9,0,0],
        [0,0.09,0],
        [0,0,0.001]
    ],
    [
        [0.009,0,0],
        [0,0.99,0],
        [0,0,0.001]
    ],
    [
        [0.0001,0,0],
        [0,0.0009,0],
        [0,0,0.999]
    ]

]
for i in range(N):
    #rho = generate_psd(D)
    #rho /= np.trace(rho)
    global_As[i]=np.array(global_As[i])
    global_As[i]/=np.trace(global_As[i])

alphas = [0.2,0.4]
w = np.array([1/3,1/3,1/3])
# Loop over different values of alpha
for alpha in alphas:
    As = []
    for i in range(N): # Total O(N * D^3) time for initialization
        As.append(np.diag(np.diag(global_As[i])**alpha)) # Taking matrix power of supported states beforehand
    
    Q = np.identity(D) / D
    # Arrays to store iteration results
    iterations = [0]
    iterates = [Q]
    F_values = [F(Q / np.trace(Q), As, w, alpha, Diagonal=True)]

    # Perform the iterative process
    T = 60
    for i in range(T):
        Q = SimpleIteration(Q, As, w, alpha, Diagonal = True)
        Q /= np.trace(Q) # We deliberately take normalization here for the sake of numerical stability. It is easy to verify that it won't affect the output of the algorithm.
        iterations.append(i+1)
        F_val = F(Q, As, w, alpha, Diagonal = True)
        F_values.append(F_val)
        iterates.append(Q)
        print(alpha,F_val,np.diag(Q))


    #min_f_val = min(f_values)
    min_F_val, Q_star = Polyak(As, w, alpha, Diagonal=True) # Use mirror descent with Polyak stepsize to compute
                                       # minimum value instead. As our algorithm is not 
                                       # proved to converge for alpha < 0.5
    print("Minimum computed by Polyak",min_F_val)
    # for i in range(len(F_values)):
    #     F_values[i] = F_values[i]-min_F_val
    distances_to_opt = []
    Q_star_power = linalg.fractional_matrix_power(Q_star,1-alpha)
    for i in range(len(F_values)):
        F_values[i] = F_values[i]-min_F_val 
        distances_to_opt.append(ThompSonMetric(linalg.fractional_matrix_power(iterates[i],1-alpha),Q_star_power))
        print("Distance",distances_to_opt[-1])

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Optimization error
    axs[0].plot(iterations, F_values, label='Optimization error')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Number of iterations', fontsize=14)
    axs[0].set_ylabel('Approx. optimization error', fontsize=14)
    axs[0].set_title(f'α = {alpha}', fontsize=14)
    axs[0].set_xlim(0, 30)
    axs[0].set_ylim(1e-11, 1e1)
    axs[0].grid(True)
    axs[0].tick_params(labelsize=12)

    # Right: Distance to optimality
    axs[1].plot(iterations, distances_to_opt, label='Distance to $Q_\star$')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Number of iterations', fontsize=14)
    axs[1].set_ylabel('Approx. iter. error', fontsize=14)
    axs[1].set_title(f'α = {alpha}', fontsize=14)
    axs[1].set_xlim(0, 30)
    axs[1].set_ylim(1e-11, 1e1)
    axs[1].grid(True)
    axs[1].tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f'figure_alpha_{alpha}.png')
    plt.show()








