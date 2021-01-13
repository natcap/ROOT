import sys
import os
import glob
import json
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_sols(opt_folder, cols, sol_file_str='solution_*.csv'):
    """
    Loads solution files from opt_folder, returning a list of np.arrays
    containing the values from the columns listed in cols.
    :param opt_folder:
    :param cols:
    :param sol_file_str:
    :return:
    """
    sol_files = glob.glob(os.path.join(opt_folder, sol_file_str))
    decisions = []
    for f in sol_files:
        df = pd.read_csv(f)
        decisions.append(np.array(df[cols]))
    return decisions


def make_agreement_map_sample(sols, ppm, nsamps):
    """
    Generates a list of nsamps agreement maps. Each map is constructed from
    the values in sols, using ppm (points per map) draws per sample. Draws are
    with replacement.

    :param sols:
    :param ppm:
    :param nsamps:
    :return:
    """
    agreement_map_samples = []
    nsols = len(sols)
    for i in range(nsamps):
        sol_selection = np.random.choice(nsols, ppm)
        amap = sum([sols[s] for s in sol_selection]) / ppm
        agreement_map_samples.append(amap)
    return agreement_map_samples


def rmse(amap1, amap2):
    return np.sqrt(sum( np.square(amap1 - amap2) ))


def bootstrap_pairwise_rmse_from_sample(amap_sample, ndraws, area_weight_vec=None):
    """
    Calculates ndraws pairwise agreement map rmse scores. This version draws
    with replacement from a list of agreement maps (amap_sample).
    :param amap_sample:
    :param ndraws:
    :param area_weight_vec:
    :return:
    """

    nsols = len(amap_sample)
    rmse_vals = np.zeros(ndraws)

    for i in range(ndraws):
        m1, m2 = np.random.choice(nsols, 2)
        if area_weight_vec is not None:
            rmse_vals[i] = rmse(amap_sample[m1] * area_weight_vec,
                                amap_sample[m2] * area_weight_vec)
        else:
            rmse_vals[i] = rmse(amap_sample[m1], amap_sample[m2])

    return rmse_vals


def boostrap_pairwise_rmse_from_sols(sols, ppm, ndraws, area_weight_vec=None):
    """
    Calculates ndraws pairwise agreement map rmse scores. This version creates
    agreement maps from the solutions for each comparison.
    :param sols:
    :param ppm:
    :param ndraws:
    :param area_weight_vec:
    :return:
    """

    nsols = len(sols)
    rmse_vals = np.zeros(ndraws)

    for i in range(ndraws):
        sol_selection1 = np.random.choice(nsols, ppm)
        sol_selection2 = np.random.choice(nsols, ppm)
        amap1 = sum([sols[s] for s in sol_selection1]) / ppm
        amap2 = sum([sols[s] for s in sol_selection2]) / ppm
        if area_weight_vec is not None:
            amap1 *= area_weight_vec
            amap2 *= area_weight_vec
        rmse_vals[i] = rmse(amap1, amap2)

    return rmse_vals


def bootstrap_mean_variance_from_sample(amap_sample):
    # TODO think about how this is going to work with multiple activity amaps
    sdu_var = np.var(np.array(amap_sample), axis=0)
    mean_var = np.mean(sdu_var)
    return mean_var


def bootstrap_mean_variance_from_sols(sols, ppm, nsamps):
    sample = make_agreement_map_sample(sols, ppm, nsamps)
    return bootstrap_mean_variance_from_sample(sample)












