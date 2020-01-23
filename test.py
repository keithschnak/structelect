import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import quantecon as qe
import scipy.sparse as sparse
from scipy.stats import norm
from scipy.stats import binom
from quantecon import compute_fixed_point
from quantecon.markov import DiscreteDP
import scipy.special as sps
import itertools as it
import time
import multiprocessing as mp
from functools import partial
import pandas as pd
import scipy.optimize as opt
import math
import importlib
import eq

#importlib.reload(eq)

## some sample parameters for testing functions
pol_br_sample = np.repeat(.5, 20)
pm_sample = np.array([[.9, .025, .025, .025, .025],
                      [.025, .9, .025, .025, .025],
                      [.025, .025, .9, .025, .025],
                      [.025, .025, .025, .9, .025],
                      [.025, .025, .025, .025, .9]])
etah_sample = 1
etal_sample = -1
tauy_sample = 1
pipar_sample = np.asarray([.5, .55])
prg_sample = .8
prb_sample = .2
etam_sample = [-.1, -.05, 0, .05, .1] #sample values for etam_sample
beta_sample = .05 #sample value for beta
delta_sample = .85
my_ygrid = np.linspace(-4, 4, 10)

## setting up a model and generating test data
M = eq.Model(my_ygrid, 5) #this sets up an instance of my mathematical model
#the next line computes an equilibrium to my model given the parameters

#note -- I have to compute the equilibrium to the model every time I evaluate
#the likelihood function, so even though I am asking you to optimize my_ll (below)
#most of the speedup will come from speeding up eq_compute and its supporting functions
E = M.eq_compute(pol_br_sample,
                 etah_sample,
                 etal_sample,
                 tauy_sample,
                 pipar_sample,
                 prg_sample,
                 prb_sample,
                 pm_sample,
                 etam_sample,
                 beta_sample,
                 delta_sample)
#this simulates some data from my model and stores in a Structmodel object
# which includes a dataset as well as some of the information used to set
#up the model
D = eq.Structmodel(E.eq_simulate(1000, 100000, my_ygrid, M.statelist, 5), my_ygrid, 5)
#next line generates some random  for parameters
my_pars = np.random.normal(0, 1, 13)

#next lines are one evaluation of my log likelihood function (actually -2*ll).
#this is what I want you to optimize
t0 = time.time()
T = D.my_ll(my_pars, pm_sample, pol_br_sample, delta_sample)
t1 = time.time()
print(t1 - t0)

#line below WOULD compute the maximum likelihood estimator of the Model
#but currently this would take weeks or months
#T2 = D.my_mle(my_pars, pm_sample, pol_br_sample, delta_sample)
