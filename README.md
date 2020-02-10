# Markov-Random-Fields

## Ising model
We aim to sample the following distribution on state space {0,1}^n where

<img src="img/gibbs_eq.png" width="60%">

<img src="img/ising_eq.png" width="30%">

We achieved sampling Ising model by using 3 different algorithms: Gibbs sampling, Metropolis sampling, ICM.

  Example with beta = 1.  :

<img src="img/ising_gibbs_1_.png" width="60%">

  Example with beta = 3.  :

<img src="img/ising_gibbs_3_.png" width="60%">

While MCMC algorithms Gibbs and Metropolis have theorical guarantee to converge to the designed distribution, ICM converge quickly to a local minimum. On the following graph we compare convergence of these algorithms by measuring how fast they minimize the global energy of the distribution.

<img src="img/all_curves.png" width="60%">

If ICM is faster it requires to start from a suitable initial solution and if not it may not converge at all. On the following plots we compare the ability of respectively ICM, Gibbs and Metropolis algorithms to converge to Ising model distribution starting from a all white image.

<img src="img/ising_icm_from_zeros_.png" width="60%">
<img src="img/ising_gibbs_from_zeros_.png" width="60%">
<img src="img/ising_metro_from_zeros_.png" width="60%">

## Potts model

We also sample the Potts model, a more generalized version of Ising model as the state space {0,1,2,...,q}^n is no more binary.

In the following we sampled:

<img src="img/potts_simple_eq.png" width="20%">

<img src="img/potts_no_mix_.png" width="60%">

In the following we sampled:

<img src="img/potts_hierarch_eq.png" width="20%">

<img src="img/potts_with_hierarch_mix_.png" width="60%">

## Image denoising 

<img src="img/duo_test.png" width="60%">
<img src="img/stats_patches.png" width="40%">
<img src="img/test_without_.png" width="90%">
<img src="img/test_with_ising_.png" width="90%">
<img src="img/recovery_curves.png" width="60%">
