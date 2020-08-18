"""
Utility functions for simulated data.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def gen_fuzzy_rdd(n_samples, fuzzy_gap, tau, take=0.0, seed=0):
    """
    Builds a fuzzy RDD with imperfect compliance.
    
    p is the "true" probability of getting treatment, and is a function of x and the unknown covariate u.
    
    Args:
        n_samples (int): number of samples to draw
        fuzzy_fuzzy_gap (float): the fuzzy_gap in treatment probability, centered at 0.5. Max value is 1, and induces a sharp RDD.
        tau (float): the treatment effect on the outcome
        take (float): the coefficient for probability of taking treatment, defaults to 0.2
        seed (int): seed for reproducibility
        
    Returns:
        df (pd.DataFrame): pandas Dataframe with x,y,t,z,u,p populated (note, do not use u, p in regression)
    """
    
    np.random.seed(seed)
    
    # observed covariates
    x = np.random.uniform(0, 1, n_samples)

    # unobserved confounder
    #u = np.random.uniform(-.1, 0.1, n_samples)
    u = np.random.normal(0, 0.1, n_samples)

    # boundary
    b = 0.5

    # boundary indicator
    d = (x > b).astype(int)

    gamma_p = np.random.normal(0, 1, 1)
    noise_p = np.random.normal(0, 1, n_samples)

    
    mu = d*(0.5 + fuzzy_gap/2) + (1-d)*(0.5 - fuzzy_gap/2)
    #print(mu)
    p_take = take*x - take/2 + u #(fuzzy_gap/2)

    p_adj = mu + p_take
    p_adj = np.clip(p_adj, 0,1)

    # treatment indicator
    t = np.random.binomial(1, p_adj, n_samples)
    
    # outcome and treatment
    gamma_y = np.random.normal(0, 1, 1)
    noise_y = np.random.normal(0, 1, n_samples)
    y = x*gamma_y + t*tau + noise_y #+ u
    
    df = pd.DataFrame()
    df['x'] = x # running variable
    df['z'] = d # indicator for above/below threshold
    df['t'] = t # indicator for actual treatment assignment
    df['y'] = y # outcome
    df['p'] = p_adj # true probability of treatment
    df['u'] = u # "unobserved" covariate
    
    df['x_lower'] = (1-df['z'])*(df['x'] - 0.5) # adjusted x for 2SLS
    df['x_upper'] = df['z']*(df['x'] - 0.5) # adjusted x for 2SLS
    df['p_compliance'] = (df['z']*df['p']) + (1-df['z'])*(1-df['p']) # probability of compliance
    
    return df


def point_plot(x, target, df, scale, errwidth=0):
    """Visualizes the simulated data with a pointplot."""

    sns.pointplot(np.floor(df[x]*scale) / scale, df[target], join=False, errwidth=errwidth)
    plt.xticks(np.arange(0,scale+1,scale/10), [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])