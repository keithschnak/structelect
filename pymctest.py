import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pymc3 as pm
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
