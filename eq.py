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

#the functions in this file are used to compute an equilibrium to my game theoretic
#model and then to compute the log likelihood of a statistical model
# the players are voters and candidates

#this is the voter's per-period utility
def reward(v, y, m, zd, zr, pp, term, etam, beta):
    return y + v*etam[m] + (1-term)*beta*(v*pp + (1-v)*(1-pp))

#the model has continuous states but they are discretized for this program
# the discretized states space is given by ygrid which always has equally spaced
# intervals representing the midpoints of each ''bin''
# the next three functions are different versions of a discretized normal
# distribution defined using these cutpoints.
def discretized_normal(yvals, mean, sd):
    s = yvals[1] - yvals[0] #distance between points in ygrid
    bot_dist = norm.cdf(x=yvals - s/2.0, loc=mean, scale=sd) #cdf at midpoint between point in grid and previous point
    top_dist = norm.cdf(x=yvals + s/2.0, loc=mean, scale=sd) #cdf at midpoint between point in grid and next point
    bot_dist[0] = 0 #there is no previous point for the first value
    top_dist[-1] = 1 #there is no next point for the last value
    pr_nearest = top_dist - bot_dist #this gives probability of all nearest neighbors to a point
    return pr_nearest

def discretized_normal_log(yvals, mean, sd):
    s = yvals[1] - yvals[0] #distance between points in ygrid
    bot_dist = norm.logcdf(x=yvals - s/2.0, loc=mean, scale=sd) #LOG cdf at midpoint between point in grid and previous point
    top_dist = norm.logcdf(x=yvals + s/2.0, loc=mean, scale=sd) #LOG cdf at midpoint between point in grid and next point
    diff = top_dist + np.log(-np.expm1(bot_dist - top_dist)) #log1p difference formula
    diff[0] = top_dist[0] #first value should integrate from -inf
    diff[-1] = norm.logsf(x=ygrid[-1] - s/2.0, loc=mean, scale=sd) #last value should integrate to inf
    return diff

def discretized_normal_log2(yvals, mean, sd):
    s = yvals[1] - yvals[0] #distance between points in ygrid
    return norm.logpdf(x=yvals, loc=mean, scale=sd) + np.log(s)

#DPPsol is a class that holds the results of solving a dynamic programming problem
# as well as some of the inputs to the function
class DPPsol:
    def __init__(self, vsol, R, Q):
        self.vsol = vsol
        self.R = R #reward matrix
        self.Q = Q #state transition matrix

#Equilibrium is similar but for equilibria to the model. the voter strategy
# is a result of solving a DPP but the politician strategy pol_strat is simipler
class Equilibrium:
    def __init__(self, pol_strat, voter_sol, R, Q):
        self.pol_strat = pol_strat
        self.voter_sol = voter_sol
        self.R = R
        self.Q = Q
#eq_simulate simulates data from an equilibrium to the model. used to generate
#test data.
    def eq_simulate(self, nperiods, nvoters, ygrid, statelist, nm):
        s = ygrid[1] - ygrid[0]
        nstates = len(self.voter_sol.v)
        #construct choice-specific value functions
        v_dem = self.Q[range(nstates), :] @ self.voter_sol.v + self.R[range(nstates)]
        v_rep = self.Q[range(nstates, 2*nstates), :] @ self.voter_sol.v + self.R[range(nstates, 2*nstates)]
        v_inc = np.concatenate((v_dem[range(200)], v_rep[range(400, 600)]))
        v_cha = np.concatenate((v_rep[range(200)], v_dem[range(400, 600)]))
        eincshare = np.exp(v_inc)/(np.exp(v_inc) + np.exp(v_cha))
        yvals = np.tile(ygrid, nm*16)
        yvals = np.concatenate((yvals[range(200)], yvals[range(400, 600)]))
        #simulate data from equilibrium
        nvars = statelist.shape[1]+4
        sim_states = self.voter_sol.mc.simulate(ts_length=nperiods, init=0) #simulate states from markov chain
        my_data = np.empty([nperiods, nvars])
        for i in range(nperiods):
            for j in range(statelist.shape[1]):
                my_data[i,j] = statelist[sim_states[i], j]
            my_data[i, 5] = np.random.uniform(ygrid[int(my_data[i, 5])]- s/2.0, ygrid[int(my_data[i, 5])]+ s/2.0, 1)
            my_data[i, nvars-4] = self.voter_sol.sigma[sim_states[i]]
            my_data[i, nvars-3] = np.exp(v_rep[sim_states[i]])/(np.exp(v_dem[sim_states[i]]) + np.exp(v_rep[sim_states[i]]))
            my_data[i, nvars-2] = round(my_data[i, nvars-3]*nvoters)
            my_data[i, nvars-1] = nvoters - my_data[i, nvars-2]
        my_data = pd.DataFrame(my_data, columns=["party", "term", "zd", "zr", "m", "y", "rwin", "rvoteshare", "rvotes", "dvotes"])

        my_govid = np.zeros(my_data.shape[0])
        k=0
        for i in range(nperiods):
            if my_data['term'][i] == 0:
                my_govid[i] = int(k)
                k = k+1
            elif my_data['term'][i] == 1:
                my_govid[i] = my_govid[i-1]
        my_govid = my_govid.astype(int)
        my_data = my_data.assign(govid = my_govid)

        #reshape from long to wide
        i = my_data.groupby('govid').cumcount() + 1
        d1 = my_data.set_index(['govid', i]).unstack().sort_index(1, 1)
        d1.columns = d1.columns.to_series().map('{0[0]}_{0[1]}'.format)
        mdw = d1
        twoterm = mdw['party_1']*mdw['rwin_1'] + (1-mdw['party_1'])*(1-mdw['rwin_1'])

        mdw = mdw.assign(twoterm = twoterm)
        mdw.drop(mdw.tail(1).index,inplace=True)
        return(mdw)

#most of the action happens in methods for this Model class. I am using a class
#because I compute several attributes of the model
class Model:
    def __init__(self, ygrid, nm):
        self.ygrid = ygrid
        self.nm = nm
        self.ygl = len(ygrid)
        self.nstates = self.ygl*nm*16
        self.nsa=self.nstates*2
        salist = np.empty([self.nsa, 7]) #empty list for states
        statelist = np.empty([self.nstates, 6])

        # in the code that solves the voter's dynamic programming problem
        # i need to set up a list of states (which involve several variables)
        # and a list of state-action pairs (the actions are always just Republican
        # or Democrat).
        #I define these below as attributes of the model so I only have to do it once

        #generate state-action pair list
        #v (0) then party (1) then term (2) then zd (3) then zr (4) then m (5) then y (6)
        k=0
        for i in it.product(range(2), range(2), range(2), range(2), range(2), range(self.nm), range(self.ygl)):
            salist[k] = [i[0], i[1], i[2], i[3],i[4], i[5], i[6]]
            k = k+1
        salist = salist.astype(int)

        #generate state  list
        #party (0) then term (1) then zd (2) then zr (3) then m (4) then y (5)
        k=0
        for i in it.product(range(2), range(2), range(2), range(2), range(self.nm), range(self.ygl)):
            statelist[k] = [i[0], i[1], i[2], i[3],i[4], i[5]]
            k = k+1
        statelist = statelist.astype(int)

        #state and action indices
        action_indices = [i for i in it.repeat(0, self.nstates)]
        for j in it.repeat(1, self.nstates):
            action_indices.append(j)

        state_indices = [i for i in range(self.nstates)]
        for j in range(self.nstates):
            state_indices.append(j)
        self.salist = salist
        self.statelist=statelist
        self.action_indices = action_indices
        self.state_indices = state_indices

        # next few lines generate lists of indices from the state-action pair lists
        #with particular attributes. again, doing this once here so I don't have
        # to do it repeatedly later

        #index to look up probability incumbent gives high effort in each state-action-pair
        self.pind0 = salist[:,1]*2*nm +  (salist[:,1]*salist[:,4] + \
                    (1-salist[:,1])*salist[:,3])*nm + salist[:,5]
        #list of indices in which an incumbent is reelected
        self.increelect = np.where((1-salist[:,2])*(salist[:,0]*salist[:,1] + \
                                    (1 - salist[:,0])*(1 - salist[:,1]))==1)
        self.pind1 = statelist[:,0]*2*nm + (statelist[:,0]*statelist[:,3] +
                    (1 - statelist[:,0])*statelist[:,2])*nm + statelist[:,4]
        self.lameduck = statelist[:,1]

        #next several lines compute various matrices that are useful in some
        #of the functions. these lines are also only done once when i define
        #the model so not worth optimizing too much here
        traitblockD = np.transpose(np.tile(np.vstack((np.tile(np.repeat(0, self.nstates), (self.ygl*self.nm*2, 1)),
                                                    np.tile(np.repeat(1, self.nstates), (self.ygl*self.nm*2, 1)))),
                                                    (4, 1)))
        ZDp0 = np.vstack((np.zeros((self.nstates, self.nstates)), 1-traitblockD))
        ZDp1 = np.vstack((np.zeros((self.nstates, self.nstates)), traitblockD))
        blk1 = np.tile(np.vstack((np.tile([1, 0], 4), np.tile([0, 1], 4))), (4, 1))
        ZD1 = np.vstack((np.kron(blk1, np.ones((self.ygl*self.nm*2, self.ygl*self.nm*2))), np.zeros((self.nstates, self.nstates))))
        blk0 = np.tile(np.vstack((np.tile([0, 1], 4), np.tile([1, 0], 4))), (4, 1))
        ZD0 = np.vstack((np.kron(blk0, np.ones((self.ygl*self.nm*2, self.ygl*self.nm*2))), np.zeros((self.nstates, self.nstates))))
        traitblockR = np.transpose(np.tile(np.vstack((np.tile(np.repeat(0, self.nstates), (self.ygl*self.nm, 1)),
                                                    np.tile(np.repeat(1, self.nstates), (self.ygl*self.nm, 1)))),
                                                    (8, 1)))
        ZRp0 = np.vstack((1-traitblockR, np.zeros((self.nstates, self.nstates))))
        ZRp1 = np.vstack((traitblockR, np.zeros((self.nstates, self.nstates))))
        blk1R = np.tile(np.vstack((np.tile([1, 0], 8), np.tile([0, 1], 8))), (8, 1))
        ZR1 = np.vstack((np.zeros((self.nstates, self.nstates)), np.kron(blk1R, np.ones((self.ygl*self.nm, self.ygl*self.nm)))))
        blk0R = np.tile(np.vstack((np.tile([0, 1], 8), np.tile([1, 0], 8))), (8, 1))
        ZR0 = np.vstack((np.zeros((self.nstates, self.nstates)), np.kron(blk0R, np.ones((self.ygl*self.nm, self.ygl*self.nm)))))
        no_term_one_trans = (salist[:,0]*salist[:,1] + (1-salist[:,0])*(1-salist[:,1]))*(1-salist[:,2])
        no_term_two_trans = ((1-salist[:,0])*salist[:,1] + salist[:,0]*(1-salist[:,1]))*(1-salist[:,2]) + salist[:,2]
        term_poss =  1 - (np.outer(no_term_one_trans, 1-statelist[:,1]) + np.outer(no_term_two_trans, statelist[:,1]))
        term_poss = sparse.coo_matrix(term_poss)
        p_poss = np.outer(salist[:,0], statelist[:,0]) + np.outer(1-salist[:,0], 1-statelist[:,0])
        p_poss = sparse.coo_matrix(p_poss)
        poss_mat = sparse.coo_matrix.multiply(term_poss, p_poss)
        self.poss_mat = poss_mat
        self.ZDp0 = ZDp0
        self.ZDp1 = ZDp1
        self.ZD1 = ZD1
        self.ZD0 = ZD0
        self.ZRp0 = ZRp0
        self.ZRp1 = ZRp1
        self.ZR1 = ZR1
        self.ZR0 = ZR0
        ptvec = np.concatenate((np.repeat(0, 2*self.nm),
                                np.repeat(8*self.nm*self.ygl, 2*self.nm)))
        zincvec = np.concatenate((np.repeat(0, self.nm),
                                np.repeat(2*self.nm*self.ygl, self.nm),
                                np.repeat(0, self.nm),
                                np.repeat(self.nm*self.ygl, self.nm)))
        zoppvec = np.concatenate((np.repeat(self.nm*self.ygl, 10),
                                np.repeat(2*self.nm*self.ygl, 10)))
        mvec = np.tile([m*self.ygl for m in range(self.nm)], 4)
        incindmat = np.vstack((ptvec + zincvec + mvec,
                            ptvec + zincvec + mvec + zoppvec))
        self.incindmat = incindmat

    #the next few methods help me solve the voter's dynamic programming problem
    #this is the slowest part of the code and has to be done repeatedly

    #computes, given parameters, what is the probability that a high type is in office
    #next period
    def winner_type_prob(self, etah, etal, tauy, pipar, prg, prb, pol_br):
        nm = self.nm
        pirat = (1-pipar)/pipar
        prrat = prb/prg
        prcomprat = (1-prb)/prb
        peff = pol_br[self.pind0]
        muc = self.salist[:,0]*(1 + pirat[1]*prrat**self.salist[:,4]*prcomprat**(1- self.salist[:,4]))**(-1) + \
                        (1-self.salist[:,0])*(1 + pirat[0]*prrat**self.salist[:,3]*prcomprat**(1- self.salist[:,3]))**(-1)
        fH0 = np.tile(discretized_normal_log2(self.ygrid, etah, tauy**(-2)), nm*32)
        fL0 = np.tile(discretized_normal_log2(self.ygrid, etal, tauy**(-2)), nm*32)
        mui = (1 + (1-muc)*peff/muc + \
        np.exp(fL0 - fH0)*(1-muc)*(1-peff)/muc)**(-1)
        mu = muc
        mu[self.increelect] = mui[self.increelect]
        return(mu)

    #this function provides an intermediate step toward computing transition
    #probabilities -- specifically giving us the probability associated with
    #two elements of the state -- y and m -- with y given by ygrid and m generated
    #from probabilities in parameter Pm
    def fym(self, etah, etal, tauy, pipar, prg, prb, Pm, pol_br):
        nm = self.nm
        #repeating discretized normal over ygrid a few tiems to populate
        #state-action matrix
        fH1 = np.tile(discretized_normal(self.ygrid, etah, tauy**(-2)), nm*16)
        fL1 = np.tile(discretized_normal(self.ygrid, etal, tauy**(-2)), nm*16)
        #probability of high effort in each state
        peff1 = pol_br[self.pind1]
        peff1 = peff1*(1 - self.lameduck)
        #compute winner type probabilities using previous functions
        mu = self.winner_type_prob(etah, etal, tauy, pipar, prg, prb, pol_br)
        my_mult = fH1 + peff1*(fL1 - fH1) - fL1
        my_add = peff1*(fH1 - fL1) + fL1
        fy = np.outer(mu, my_mult) + my_add
        #matrix Pm provides transition probabilities for a discrete variables with 5
        #values. next few lines repeat this matrix in the appropriate parts of
        #the state-action matrix and then multiplies those probabilities by fy
        PMM = np.empty([self.nsa, self.nstates])
        for i in range(self.nsa):
            PMM[i,:] = np.tile(np.repeat(Pm[self.salist[i,5]], self.ygl), 16)
        return np.multiply(fy, PMM)
    #next three functions give another component of transition probabilities
    # a lot of the probabilities are zero so they are sparse matrices
    def zdfun(self, pipar, prg, prb):
        p0 = pipar[0]*(1-prg) + (1-pipar[0])*(1-prb)
        p1 = pipar[0]*prg + (1-pipar[0])*prb
        return sparse.coo_matrix(np.zeros(self.ZD1.shape) + self.ZD1 + p0*self.ZDp0 + p1*self.ZDp1)

    def zrfun(self, pipar, prg, prb):
        p0 = pipar[1]*(1-prg) + (1-pipar[1])*(1-prb)
        p1 = pipar[1]*prg + (1-pipar[1])*prb
        return sparse.coo_matrix(np.zeros(self.ZR1.shape) + self.ZR1 + p0*self.ZRp0 + p1*self.ZRp1)

    def zfun(self, pipar, prg, prb):
        ZD = self.zdfun(pipar, prg, prb)
        ZR = self.zrfun(pipar, prg, prb)
        return sparse.coo_matrix.multiply(ZD, ZR)

    #this function populates the state transition probability matrix
    def populate_Q(self, etah, etal, tauy, pipar, prg, prb, Pm, pol_br):
        my_fym = sparse.coo_matrix(self.fym(etah, etal, tauy, pipar, prg, prb, Pm, pol_br))
        my_z = self.zfun(pipar, prg, prb)
        my_fym_z = sparse.coo_matrix.multiply(my_fym, my_z)
        my_Q = sparse.coo_matrix.multiply(my_fym_z, self.poss_mat)
        return(my_Q)
    # this function populates the reward matrix
    def rewardv(self, etam, beta):
        R = np.empty(self.nsa)
        statelist = self.statelist
        state_indices = self.state_indices
        for i in range(self.nsa):
            R[i] = reward(v=self.action_indices[i],
                          y=self.ygrid[int(statelist[state_indices[i], 5])],
                          m=int(statelist[state_indices[i], 4]),
                          zd=statelist[state_indices[i], 2],
                          zr=statelist[state_indices[i], 3],
                          pp=statelist[state_indices[i], 0],
                          term=statelist[state_indices[i], 1],
                          etam=etam,
                          beta=beta)
        return R
    #now we solve the voter's dynamic programming problem using these matrices
    # I am relying here on DiscreteDP in the quantecon library
    def voter_dpp(self, etah, etal, tauy, pipar, prg, prb, Pm, pol_br, etam, beta, delta):
        R = self.rewardv(etam, beta)
        Q = self.populate_Q(etah, etal, tauy, pipar, prg, prb, Pm, pol_br)
        dpp = DiscreteDP(R=R, Q=Q, beta=delta, s_indices=self.state_indices, a_indices=self.action_indices)
        results = dpp.solve(method='policy_iteration')
        sol = DPPsol(results, R, Q)
        return(sol)
    #this function computes the politician's best response given the voter stategy
    def pol_response_fun(self, etah, etal, tauy, pipar, prg, prb, incwin):
        nm = self.nm
        fH = np.tile(discretized_normal(self.ygrid, etah, tauy**(-2)), nm*16) #y density given high effort
        fL = np.tile(discretized_normal(self.ygrid, etal, tauy**(-2)), nm*16) #y density given low effort
        pH = fH*incwin #multiply y density by win/loss for high effort
        pL = fL*incwin #...same for low effort
        #pr zopp=1 for each incumbent
        przopp1 = np.concatenate((np.repeat(pipar[1]*prg + (1-pipar[1])*prb, 2*nm),
                                np.repeat(pipar[0]*prg + (1-pipar[0])*prb, 2*nm)))
        resultH = np.empty([2, 4*nm])
        resultL = np.empty([2, 4*nm])
        for i in range(4*nm):
            resultH[0, i] = np.sum(pH[range(self.incindmat[0, i], self.incindmat[0, i]+self.ygl)])
            resultL[0, i] = np.sum(pL[range(self.incindmat[0, i], self.incindmat[0, i]+self.ygl)])
            resultH[1, i] = np.sum(pH[range(self.incindmat[1, i], self.incindmat[1, i]+self.ygl)])
            resultL[1, i] = np.sum(pL[range(self.incindmat[1, i], self.incindmat[1, i]+self.ygl)])
            result = (przopp1*resultH[1,:] + (1-przopp1)*resultH[0,:]) -  (przopp1*resultL[1,:] + (1-przopp1)*resultL[0,:])
        return result
    #eq_iter takes an initial value of the politician's strategy, computes the
    #voter's best response, then computes the politician's best response to the
    #the resulting voter strategy
    def eq_iter(self, init, etah, etal, tauy, pipar, prg, prb, Pm, etam, beta, delta):
        #compute voter dpp given initial value of politician strategy
        nstates = self.nstates
        ygl = self.ygl
        nm = self.nm
        vdpp = self.voter_dpp(etah, etal, tauy, pipar, prg, prb, Pm, init, etam, beta, delta)
        incwin = np.concatenate((1-vdpp.vsol.sigma[range(8*nm*ygl)], vdpp.vsol.sigma[(-8*nm*ygl):nstates]))
        pbr = self.pol_response_fun(etah, etal, tauy, pipar, prg, prb, incwin) #compute new politiciaan best response
        return init - pbr #return difference

    #an equilibrium is a mutual best response, so I find this by finding the root
    #of eq_iter, at such a point all players are best responding
    #solutions to this problem should be unique
    def eq_compute(self, init, etah, etal, tauy, pipar, prg, prb, Pm, etam, beta, delta):
        roottest = opt.least_squares(fun=self.eq_iter,
                                    x0=init,
                                    bounds=(0, 1),
                                    args=(etah,etal,tauy, pipar, prg, prb, Pm, etam, beta, delta)
                                    )
        vdpp = self.voter_dpp(etah, etal, tauy, pipar, prg, prb, Pm, roottest.x,etam, beta, delta)
        sol = Equilibrium(roottest.x, vdpp.vsol, vdpp.R, vdpp.Q)
        return sol

#this is just my attempt to vectorize the logsumexp function for speed
def vlogsumexp(A, B, C):
    mymax = np.maximum(A, np.maximum(B, C))
    Adiff = np.exp(A - mymax)
    Bdiff = np.exp(B - mymax)
    Cdiff = np.exp(C - mymax)
    return mymax + np.log(Adiff + Bdiff + Cdiff)

#find nearest to a given value in an array
def find_nearest(array, value):
    ''' Find nearest value in an array '''
    idx = (np.abs(array-value)).argmin()
    return idx

#the structural model class includes a dataset and some information about the
#model that need to be set by the researcher
class Structmodel(Model):
    def __init__(self, data, ygrid, nm):
        self.data = data
        self.ygrid = ygrid
        ygl = len(ygrid)
        yidata1 = np.empty(len(data))
        yidata2 = np.empty(len(data))
        for i in range(len(data)):
            yidata1[i] = round(find_nearest(ygrid, data.iloc[i, data.columns.get_loc("y_1")]))
            yidata2[i] = round(find_nearest(ygrid, data.iloc[i, data.columns.get_loc("y_2")]))
        yidata1 = yidata1.astype(int)
        yidata2 = yidata2.astype(int)

        ot = np.where(data['twoterm']==0)[0]
        tt = np.where(data['twoterm']==1)[0]
        infoset1 = data['party_1']*(8*nm*ygl) + data['term_1']*(4*nm*ygl) + data['zd_1']*(2*nm*ygl) + data['zr_1']*(nm*ygl) + data['m_1']*(ygl) + yidata1 #label voter infosets for first term election
        infoset1 = infoset1.astype(int)
        infoset2 = data['party_2']*(8*nm*ygl) + data['term_2']*(4*nm*ygl) + data['zd_2']*(2*nm*ygl) + data['zr_2']*(nm*ygl) + data['m_2']*(ygl) + yidata2 #label voter infosets for second term election (lots of NaNs for one term governors but we'll avoid these in the likelihood)
        infoset2 = infoset2[tt].astype(int)
        qd = (np.sum(data['party_1']*data['zd_1']) + np.nansum(data['party_2']*data['zd_2']))/(np.sum(data['party_1']) + np.nansum(data['party_2']))
        qr = (np.sum((1-data['party_1'])*data['zr_1']) + np.nansum((1-data['party_2'])*data['zr_2']))/(np.sum(1-data['party_1']) + np.nansum((1-data['party_2'])))
        qhigh = max(qd, qr)
        qlow = min(qd, qr)
        data['party_1'] = data['party_1'].astype(int)
        data['zd_1'] = data['zd_1'].astype(int)
        data['zr_1'] = data['zr_1'].astype(int)
        data['m_1'] = data['m_1'].astype(int)
        govind = data['party_1']*2*nm + (data['party_1']*data['zr_1'] + (1-data['party_1'])*data['zd_1'])*nm + data['m_1']
        self.infoset1 = infoset1
        self.infoset2 = infoset2
        self.ot = ot
        self.tt = tt
        self.qd = qd
        self.qr = qr
        self.qhigh = qhigh
        self.qlow = qlow
        self.govind = govind
        Model.__init__(self, ygrid, nm)
    #the next function is the likelihood function -- the main target function
    #I want to optimize. note that the function first computes an equilibrium to
    #the model given the parameters and THEN computes the likelihood, which is why
    #it is so slow.
    def my_ll(self, pars, Pm, pol_br_init, delta):
        #assign parameters to natural names for readability
        etah = pars[1] + math.exp(pars[0])
        etal = pars[1]
        tauy = math.exp(pars[2])
        prg = self.qhigh + (1-self.qhigh)*(1 + math.exp(-pars[3]))**(-1)
        prb = self.qlow*(1 + math.exp(-pars[4]))**(-1)
        etam = pars[5:(5+self.nm)]
        beta = pars[5+self.nm]
        ot = self.ot
        tt = self.tt

        #party type probabilities computed from trait frequencies given prg and prb
        pipar = np.array([(self.qd - prb)/(prg - prb), (self.qr - prb)/(prg - prb)])
        pipardata = pipar[1]*self.data['party_1'] + pipar[0]*(1 - self.data['party_1'])
        zdata = self.data['party_1']*self.data['zr_1'] + (1-self.data['party_1'])*self.data['zd_1']
        mu_top = pipardata*(prg**zdata)*(1-prg)**(1 - zdata)
        mu_bottom = mu_top + (1-pipardata)*(prb**zdata)*(1-prb)**(1-zdata)
        mu = mu_top/mu_bottom

        #compute equilibrium
        eq = self.eq_compute(pol_br_init, etah, etal, tauy, pipar, prg, prb, Pm, etam, beta, delta)
        peff = eq.pol_strat[self.govind]
        peff[peff==0] = .000001
        lli, g, by, bn = (np.empty(self.data.shape[0]) for i in range(4))
        # log-likelihood contributions for one term governors
        g[ot] = np.log(mu[ot]) + norm.logpdf(x=self.data.loc[ot, 'y_1'], loc=etah, scale=tauy**(-2)) #given good type
        by[ot] = np.log(1 - mu[ot]) + np.log(peff[ot]) + norm.logpdf(x=self.data.loc[ot, 'y_1'], loc=etah, scale=tauy**(-2)) #given disciplined bad type
        bn[ot] = np.log(1 - mu[ot]) + np.log(1-peff[ot]) + norm.logpdf(x=self.data.loc[ot, 'y_1'], loc=etal, scale=tauy**(-2)) #given undisciplined bad type
        lli[ot] = vlogsumexp(g[ot], by[ot], bn[ot]) #summed together
        #log-likelihood contributions for one term governors
        g[tt] = np.log(mu[tt]) + norm.logpdf(x=self.data.loc[tt, 'y_1'], loc=etah, scale=tauy**(-2)) + \
                norm.logpdf(x=self.data.loc[tt, 'y_2'], loc=etah, scale=tauy**(-2))#given good type
        by[tt] = np.log(1 - mu[tt]) + np.log(peff[tt]) + norm.logpdf(x=self.data.loc[tt, 'y_1'], loc=etah, scale=tauy**(-2)) +  \
                norm.logpdf(x=self.data.loc[tt, 'y_2'], loc=etal, scale=tauy**(-2))#given disciplined bad type
        bn[tt] = np.log(1 - mu[tt]) + np.log(1-peff[tt]) + norm.logpdf(x=self.data.loc[tt, 'y_1'], loc=etal, scale=tauy**(-2)) + \
                norm.logpdf(x=self.data.loc[tt, 'y_2'], loc=etal, scale=tauy**(-2)) #given undisciplined bad type
        lli[tt] = vlogsumexp(g[tt], by[tt], bn[tt]) #summed together
        #choice specific value functions
        v_dem = eq.Q[range(self.nstates), :] @ eq.voter_sol.v + eq.R[range(self.nstates)]
        v_rep = eq.Q[range(self.nstates, 2*self.nstates), :] @ eq.voter_sol.v + eq.R[range(self.nstates, 2*self.nstates)]
        rvoteprob1 = np.exp(v_rep[self.infoset1])/(np.exp(v_dem[self.infoset1]) + np.exp(v_rep[self.infoset1]))
        rvoteprob2 = np.exp(v_rep[self.infoset2])/(np.exp(v_dem[self.infoset2]) + np.exp(v_rep[self.infoset2]))
        #LL contributions for election results
        lli = lli + binom.logpmf(k=self.data['rvotes_1'], n=self.data['rvotes_1'] + \
              self.data['dvotes_1'], p=rvoteprob1)
        lli[tt] = lli[tt] + binom.logpmf(k=self.data.loc[tt, 'rvotes_2'], n=self.data.loc[tt, 'rvotes_2'] + \
                  self.data.loc[tt, 'rvotes_2'], p=rvoteprob2)
        return np.sum(-2*lli)
    #this function would compute the mle 
    def my_mle(self, init, Pm, pol_br_init, delta):
        sol = opt.basinhopping(func=self.my_ll, x0=init, minimizer_kwargs={"args":(Pm, pol_br_init, delta)}, niter=5000)
        return sol





def transform_pars(pars, qd, qr):
    qhigh = max(qd, qr)
    qlow = min(qd, qr)
    etah = pars[1] + math.exp(pars[0])
    etal = pars[1]
    tauy = math.exp(pars[2])
    prg = qhigh + (1-qhigh)*(1 + math.exp(-pars[3]))**(-1)
    prb = qlow*(1 + math.exp(-pars[4]))**(-1)
    etam = pars[5:(5+nm)]
    beta = pars[5+nm]
    return [etah, etal, tauy, np.array([(qd - prb)/(prg - prb), (qr - prb)/(prg - prb)]), prg, prb, etam, beta]
