# PetzAugustin
A demonstration of our fix point iteration for computing Petz-Augustin information.
## Run 
```
$ python expr_QuantumAugustin.py
```
## Challenging Instance
input alphabet : {1,2,3}\
classical-quantum channel outputs :
    $$\rho_B^1=\begin{bmatrix}
                    0.9 & 0 & 0\\
                    0 & 0.09 & 0\\
                    0 & 0 &0.01
                \end{bmatrix}$$
    $$\rho_B^2=\begin{bmatrix}
                    0.009 & 0 & 0\\
                    0 & 0.99 & 0\\
                    0 & 0 &0.001
                \end{bmatrix}$$
    $$\rho_B^3=\begin{bmatrix}
                    0.0001 & 0 & 0\\
                    0 & 0.0009 & 0\\
                    0 & 0 &0.999
                \end{bmatrix}$$
probability distribution : $$P_X(1)=P_X(2)=P_X(3)=\frac{1}{3}$$
## References
We use the QuTip package.\
J.R. Johansson, P.D. Nation and F. Nori, QuTiP: An open-source Python framework for the dynamics of open quantum systems, *Comp. Phys. Comm.*, 2012 ([link](https://doi.org/10.1016/j.cpc.2012.02.021))
