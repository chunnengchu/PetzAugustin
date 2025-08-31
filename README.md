# PetzAugustin
A demonstration of our fix point iteration for computing Petz-Augustin mean.
## Run 
```
$ python expr_QuantumAugustin.py
```
## Challenging Instance
input alphabet : {1,2,3}\
channel outputs : $$A_1=\mathrm{Diag}\left(\left(0.9,0.09,0.01\right)\right),A_2=\mathrm{Diag}\left(\left(0.009,0.99,0.001\right)\right),A_3=\mathrm{Diag}\left(\left(0.0001,0.0009,0.999\right)\right)$$\
probability distribution : $$w[1]=w[2]=w[3]=\frac{1}{3}$$
## References
We use the QuTip package.\
J.R. Johansson, P.D. Nation and F. Nori, QuTiP: An open-source Python framework for the dynamics of open quantum systems, *Comp. Phys. Comm.*, 2012 ([link](https://doi.org/10.1016/j.cpc.2012.02.021))\
We approximate optimal point for orders smaller than 0.5 by entropic mirror descent with Polyak step size.\
J.K. You, H.C. Cheng, and Y.H. Li. Minimizing quantum Rényi divergences via mirror descent with Polyak step size. In IEEE Int. Symp. Information Theory, pages 252–257, 2022. 
