import EoN
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import random

import matplotlib.pyplot as plt
import scipy.sparse as sps

import time

import sys
sys.path.append('../risknet')

import epimodels
import ensemble
import epitools

import pdb

def load_G():
    net_size = '1e3'
    edge_list = np.loadtxt('../data/networks/edge_list_SBM_%s.txt'%net_size, usecols = [0,1], dtype = int, skiprows = 1)
    G = nx.Graph([tuple(k) for k in edge_list])
    G = nx.relabel_nodes(G, dict(zip(list(G.nodes()), range(len(G)))))
    return G

def model_settings(M, G):
    # Load city-like network
    N = len(G)
    net_size = '1e3'
    
    # Set city label
    city_classes = ['hospital', 'healthcare', 'city']
    city_classes_range = [range(int(float(net_size) * 0.005)),
                     range(int(float(net_size) * 0.005), int(float(net_size) * 0.045)),
                     range(int(float(net_size) * 0.045), N)]
    
    cohort = defaultdict()
    for node in G.nodes:
        if node in city_classes_range[0]:
            cohort[node] = {'cohort': city_classes[0]}
        elif node in city_classes_range[1]:
            cohort[node] = {'cohort': city_classes[1]}
        else:
            cohort[node] = {'cohort': city_classes[2]}
    
    nx.set_node_attributes(G, cohort)
    
    # latent period distribution: E -> I (σ)
    l = lambda k = 1.7, theta = 2: 2+np.random.gamma(k, theta)
    # infectiousness duration outside hospitals: I -> . (γ)
    g = lambda k = 1.5, theta = 2: 1+np.random.gamma(k, theta)
    # infectiousness duration in hospitals: H -> . (γ')
    gp = lambda k = 1.5, theta = 3: 1+np.random.gamma(k, theta)
    
    
    # age structure King County, WA: https://datausa.io/profile/geo/king-county-wa#demographics
    # we use the age classes: 0--19, 20--44, 45--64, 65--74, >= 75
    age_classes = np.asarray([0.2298112587,0.3876994201,0.2504385036,0.079450985,0.0525998326])
    
    # age distribution in working population: 20--44 and 45--64
    age_classes_working = np.asarray([0.3876994201,0.2504385036])/sum([0.3876994201,0.2504385036])
    
    # age-dependent hospitalization and recovery rates
    
    fh = np.asarray([0.02, 0.17, 0.25, 0.35, 0.45])
    fd = np.asarray([1e-15, 0.001, 0.005, 0.02, 0.05])
    fdp = np.asarray([1e-15, 0.001, 0.01, 0.04, 0.1])
    
    # hospitalization fraction (a...age class integers to refer to 0--19, 20--44, 45--64, 65--74, >= 75)
    
    h = lambda a, beta = 4: np.random.beta(beta*fh[a]/(1-fh[a]), b = beta)
    d = lambda a, beta = 4: np.random.beta(beta*fd[a]/(1-fd[a]), b = beta)
    dp = lambda a, beta = 4: np.random.beta(beta*fdp[a]/(1-fdp[a]), b = beta)
    
    # transmission
    
    beta0 = 0.05
    betap0 = 0.75*beta0
    
    attrs_dict = defaultdict()
    np.random.seed(1)
    
    for kk, node in enumerate(list(G.nodes())):
        age_group = np.random.choice(len(age_classes), p = age_classes)
    
        attrs = {
            'age_group': age_group
        }
    
        attrs_dict[node] = attrs
    
    nx.set_node_attributes(G, attrs_dict)
    
    Gstar = G.copy()
    
    Gs = list()
    
    for mm in range(M):
        Gmm = Gstar.copy()
        np.random.seed(mm)
        attrs_dict = defaultdict()
    
        for kk, node in enumerate(list(Gstar.nodes())):
    
            cohort = Gstar.nodes[node]['cohort']
            age_group = Gstar.nodes[node]['age_group']
    
            g_samp    = 1/g()
            gp_samp   = 1/gp()
            h_samp    = h(age_group)
            d_samp    = d(age_group)
            dp_samp   = dp(age_group)
    
            attrs = {
                'node'     : node,
                'cohort'   : cohort,
                'age_group': age_group,
                'sigma'    : 1/l(),
                'gamma'    : g_samp,
                'gammap'   : gp_samp,
                'theta'    : (1 - h_samp - d_samp) * g_samp,
                'delta'    : h_samp * g_samp,
                'mu'       : d_samp * g_samp,
                'thetap'   : (1 - dp_samp) * gp_samp,
                'mup'      : dp_samp * gp_samp
            }
    
            attrs_dict[node] = attrs
    
        nx.set_node_attributes(Gmm, attrs_dict)
        Gs.append(Gmm)
    return Gs

def get_model(Gs, M, N):
    ens = ensemble.epiens(M, Gs, N)
    return ens

def get_IC(ens, M, N):
    infected = np.random.choice(N, replace = False, size = int(N * 0.01))
    
    for member in ens.ensemble:
        member.infected = infected
        member.init_infected(infected)
    
    y0 = np.zeros([M, 5 * N])
    
    for mm, member in enumerate(ens.ensemble):
        E, I, H, R, D = np.zeros([5, N])
        S = np.ones(N,)
        I[member.infected] = 1.
        S[member.infected] = 0.
    
        y0[mm, : ] = np.hstack((S, I, H, R, D))
    return y0
