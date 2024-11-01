#######################################################################################################################
# Simplex Solver for Square Root Transformation Problem
#
# Author: Livia Qoshe, algorithm adapted from Robert A. Parker
#
# Back end for CD4 fitter tool. Uses Nelder-Mead optimization to generate a normal distribution which, when transformed
# via the square root transformation, yields a desired distribution with a specified target mean and standard deviation.
#######################################################################################################################


from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
# assert(sp.__version__ == '1.7.1')  # Nelder-Mead does not accept bounds on variables in scipy versions < 1.7.1


def get_sample_dist_values(tmean, tsd, upper, init_step=0.00005, step_sz=0.0001):
    """
    Samples from a truncated, square-root-transformed normal distribution.

    Initial, untruncated normal distribution has parameters, mean = tmean,
    std = tsd. Input distribution is truncated so that values below 0 or above an upper limit, upper, are not allowed.
    Returns the distribution mean and standard deviation after truncation. Truncation is implemented by truncating the
    probability space of the initial distribution and drawing sampled values from the truncated probability space.

    :param tmean: (float) mean of input normal distribution
    :param tsd: (float) standard deviation of input normal distribution
    :param upper: (float) upper limit for truncation
    :param init_step: (float, default = 0.00005) initial probability increment to sample from trunc probability space
    :param step_sz: (float, default 0.0001) step size to increment probability as we sample from trunc probability space
    :return: (tuple)
             0. (float) mean of truncated distribution
             0. (float) standard deviation of truncated distribution
    """
    debug = False

    if debug is True:
        upper = 1200
        print("mean: ", tmean)
        print("sd: ", tsd)

    # get cumulative probability of truncated regions
    p_less_than_zero = np.double(norm.cdf(0, loc=tmean, scale=tsd))
    p_greater_than_limit = np.double(norm.cdf(upper, loc=tmean, scale=tsd))

    if debug is True:
        print("p < 0 = ", p_less_than_zero)
        print("p > 1200 = ", p_greater_than_limit)

    p_truncated_region = p_greater_than_limit - p_less_than_zero

    # sample from truncated, transformed distribution
    # (i.e. sample within the truncated probability region of the transformed dist)

    initial_step = np.double(init_step)
    step_size = np.double(step_sz)

    start_sample = p_less_than_zero + initial_step * p_truncated_region

    p_values = np.arange(start_sample, 1, step_size)

    p = start_sample

    if debug is True:
        print("p start = ", p)

    sampled_dist = norm.ppf(p_values, loc=tmean, scale=tsd)
    sampled_dist = np.square(sampled_dist)

    if debug is True:
        print(len(sampled_dist))

    # transformed distribution parameters after truncation
    tmean = np.mean(sampled_dist)
    tsd = np.std(sampled_dist)

    return tmean, tsd


def obj_fcn(fitted_normal_params, target_mean, target_sd, upper_limit):
    """
    Calculates objective function value for a fitted mean and standard deviation (fitted_normal_params) given a
    target mean and standard deviation (target mean, target_sd). For a square root transformed distribution.
    Objective function represents the absolute value difference between target and fitted parameters.

    :param fitted_normal_params: (tuple of floats) fitted mean, fitted standard deviation.
                                 * Note: this input is specified by the scipy.optimize.minimize() function!
    :param target_mean: (float) target distribution mean
    :param target_sd: (float) target distribution standard deviation
    :param upper_limit: (float) target distribution upper limit
    :return: (float) objective function value

    scipy.optimize.minimize() doc:
    https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    """
    tmean = fitted_normal_params[0]
    tsd = fitted_normal_params[1]

    transformed_mean, transformed_sd = get_sample_dist_values(tmean, tsd, upper_limit)

    return abs(target_mean - transformed_mean) + abs(target_sd - transformed_sd)


def cd4_fitter(target_mean, target_sd, upper_lim, bounds_residue=0.00001, xtol=0.0001, ftol=0.0001, display=False):

    """
    Fits a desired square-root transform distribution using scipy.optimize.minimize() library to minimize the
    objective function value. Uses Nelder-Mead algorithm (simplex method).

    :param target_mean: (float) target/desired distribution mean
    :param target_sd: (float) target/desired distribution standard deviation
    :param upper_lim: (float) upper limit of target/desired distribution
    :param bounds_residue: (float, default = 0.0001) Mean and std must be strictly > 0. Scipy.optimize.minimize() fcn
                           doesn't accept strict inequalities for bounds, so a residue is required to bound.
    :param xtol: (float, default = 0.0001) modifies 'xaopt' parameter in Nelder-Mead algorithm. See scipy doc for more.
    :param ftol: (float, default = 0.0001) modifies 'faopt' parameter in Nelder-Mead algorithm. See scipy doc for more.
    :param display: (boolean, default = False) if True, displays information about optimization (iterations, etc)
    :return: (tuple)
            0. (float) best-fit mean
            1. (float) best-fit standard deviation
            2. (float) minimized objective function at end of optimization
            3. (OptimizeResult object) contains information about optimization process and results. See scipy doc

    scipy.optimize.minimize() doc:
    https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    Nelder-Mead algorithm doc:
    https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/optimize.minimize-neldermead.html#optimize-minimize-neldermead
    OptimizeResult object doc:
    https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
    """

    # Define bounds for fitted variables - must be greater than 0
    # residue required since there is no option for strict inequality
    residue = bounds_residue
    bounds_seq = ((0 + residue, None), (0 + residue, None))

    initial_guess = np.array([np.sqrt(target_mean), np.sqrt(target_sd)])

    output = minimize(obj_fcn, initial_guess, (target_mean, target_sd, upper_lim),
                      method='Nelder-Mead', bounds=bounds_seq, options={'xatol': xtol, 'fatol': ftol, 'disp': display})

    fitted_mean, fitted_sd = output.x
    min_obj_fcn_value = output.fun

    return fitted_mean, fitted_sd, min_obj_fcn_value, output
