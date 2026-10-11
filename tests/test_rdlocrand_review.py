"""Independent review of the rdlocrand family: new R cross-checks, oracles and regressions.

Reference numbers come from tests/reference/rdlocrand_review.R (R 4.6.1, rdlocrand 3.0,
AER, sandwich; run offline, R is not needed here). They cover configurations that
tests/test_rdlocrand_reference.py does not: discrete and heaped running variables in
every window sequence, Hotelling with five and six covariates, asymmetric windows with
polynomial and kernel fits, fuzzy TSLS / Anderson-Rubin fits with kernels, window
selection inside rdrandinf, sensitivity grids with kernels and fuzzy designs, and exact
randomization laws enumerated in R with the package's own statistics (Bernoulli trials
with logistic unit probabilities, Bernoulli Anderson-Rubin, kernel-weighted
Anderson-Rubin, Hotelling, tied outcomes on mass points, every Rosenbaum configuration).
Deterministic output agrees to rounding; exact laws to 1e-10.

Regressions for the review's fixes: capped assignment counts (no huge-integer
arithmetic or raw ValueError for method='exact' on large windows), total work budgets of
rdwinselect/rdsensitivity, undefined approximate balance p-values with one unit on a
side, the kernel HC2 companion with fewer than two positive weights, the Rosenbaum
search (both bounds over the package's configurations) and the zero-kernel-weight
sandwich (positive-weight units; R's p = 0 path is wrong there).
"""

import itertools
import math
import time
import warnings

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.rdlocrand import kernels as k

REF = {
    'check_d': [-14.505, 18.169, 2.62, 2.72699999999999],
    'check_dd': [93.0, 128.02, 299.0, 84.0, 19.516, -11.496, -2.08],
    'check_dh': [-5.8, 10.616, -25.337],
    'win_dd_default_left': [-3.0, -4.0, -5.0, -6.0, -7.0, -9.0],
    'win_dd_default_right': [3.0, 4.0, 5.0, 6.0, 7.0, 9.0],
    'win_dd_default_n': [19.0, 24.0, 34.0, 40.0, 49.0, 63.0, 32.0, 39.0, 45.0, 52.0, 61.0, 75.0],
    'win_dd_default_p': [0.252806825560671, 0.0504631243499663, 0.0833889131169035, 0.144771784666038, 0.0196119681368981, 0.00614868635964149],
    'win_dd_default_var': [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
    'win_dd_default_binom': [0.0919145498158544, 0.0769258336096096, 0.260425525897245, 0.251322103474855, 0.294233708444292, 0.349115166376781],
    'win_dd_default_rec': [-3.0, 3.0],
    'win_dd_default_summary': [65.0, 7.0, 7.0, 7.0, 14.0, 85.0, 8.0, 8.0, 16.0, 20.0],
    'win_dd_wobs_left': [-3.0, -8.0],
    'win_dd_wobs_right': [3.0, 8.0],
    'win_dd_wobs_n': [19.0, 56.0, 32.0, 65.0],
    'win_dd_wobs_p': [0.252806825560671, 0.00797761489716382],
    'win_dd_wobs_var': [1.0, 1.0],
    'win_dd_wobs_binom': [0.0919145498158544, 0.467207102076967],
    'win_dd_wobs_rec': [-3.0, 3.0],
    'win_dd_wobs_summary': [65.0, 7.0, 7.0, 7.0, 14.0, 85.0, 8.0, 8.0, 16.0, 20.0],
    'win_dd_asym_left': [-3.0, -7.0, -10.0],
    'win_dd_asym_right': [2.0, 5.0, 9.0],
    'win_dd_asym_n': [19.0, 49.0, 65.0, 20.0, 45.0, 75.0],
    'win_dd_asym_p': [0.259871460007116, 0.0956806290463738, 0.00580833134824881],
    'win_dd_asym_var': [1.0, 1.0, 1.0],
    'win_dd_asym_binom': [1.0, 0.757189074142406, 0.446989613274168],
    'win_dd_asym_rec': [-3.0, 2.0],
    'win_dd_asym_summary': [65.0, 7.0, 7.0, 7.0, 14.0, 85.0, 8.0, 8.0, 16.0, 20.0],
    'win_dd_wmin_left': [-2.5, -6.0, -10.0],
    'win_dd_wmin_right': [2.5, 6.0, 10.0],
    'win_dd_wmin_n': [14.0, 40.0, 65.0, 20.0, 52.0, 85.0],
    'win_dd_wmin_p': [0.556254259417097, 0.144771784666038, 0.00293899180955706],
    'win_dd_wmin_var': [1.0, 2.0, 1.0],
    'win_dd_wmin_binom': [0.391528305131942, 0.251322103474855, 0.120529169709489],
    'win_dd_wmin_rec': [-2.5, 2.5],
    'win_dd_wmin_summary': [65.0, 7.0, 7.0, 7.0, 14.0, 85.0, 8.0, 8.0, 16.0, 20.0],
    'win_dd_pairasym_left': [-3.0, -6.0, -9.0, -10.0],
    'win_dd_pairasym_right': [1.0, 4.0, 8.0, 10.0],
    'win_dd_pairasym_n': [19.0, 40.0, 63.0, 65.0, 16.0, 39.0, 65.0, 85.0],
    'win_dd_pairasym_p': [0.591271615075635, 0.181957351706834, 0.0154012525631666, 0.00293899180955706],
    'win_dd_pairasym_var': [1.0, 2.0, 1.0, 1.0],
    'win_dd_pairasym_binom': [0.735878800856881, 1.0, 0.929613907829986, 0.120529169709489],
    'win_dd_pairasym_rec': [-6.0, 4.0],
    'win_dd_pairasym_summary': [65.0, 7.0, 7.0, 7.0, 14.0, 85.0, 8.0, 8.0, 16.0, 20.0],
    'win_dd_step_left': [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
    'win_dd_step_right': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    'win_dd_step_n': [7.0, 14.0, 19.0, 24.0, 34.0, 40.0, 49.0, 56.0, 16.0, 20.0, 32.0, 39.0, 45.0, 52.0, 61.0, 65.0],
    'win_dd_step_p': [0.333432582454513, 0.556254259417097, 0.252806825560671, 0.0504631243499663, 0.0833889131169035, 0.144771784666038, 0.0196119681368981, 0.00797761489716382],
    'win_dd_step_var': [2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
    'win_dd_step_binom': [0.0931396484375001, 0.391528305131942, 0.0919145498158544, 0.0769258336096096, 0.260425525897245, 0.251322103474855, 0.294233708444292, 0.467207102076967],
    'win_dd_step_rec': [-3.0, 3.0],
    'win_dd_step_summary': [65.0, 7.0, 7.0, 7.0, 14.0, 85.0, 8.0, 8.0, 16.0, 20.0],
    'win_dd_cut_left': [-2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5, -9.5],
    'win_dd_cut_right': [2.5, 3.5, 4.5, 5.5, 6.5, 8.5, 9.5, 9.5],
    'win_dd_cut_n': [22.0, 27.0, 32.0, 42.0, 48.0, 57.0, 64.0, 71.0, 24.0, 31.0, 37.0, 44.0, 53.0, 67.0, 77.0, 77.0],
    'win_dd_cut_p': [0.090758045577179, 0.00580804782602442, 0.00294624534696835, 0.0419432078631192, 0.00519416660902439, 9.21651351521886e-05, 4.08640578717663e-05, 0.000107366900825058],
    'win_dd_cut_var': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'win_dd_cut_binom': [0.882995912122399, 0.694004094076131, 0.630456491786801, 0.914211637082451, 0.69082642399596, 0.419073742516636, 0.312212931656995, 0.681217086893931],
    'win_dd_cut_rec': [None, None],
    'win_dd_cut_summary': [73.0, 8.0, 8.0, 8.0, 15.0, 77.0, 8.0, 8.0, 8.0, 24.0],
    'win_dd_hot_left': [-3.0, -4.0, -5.0, -6.0, -7.0, -9.0],
    'win_dd_hot_right': [3.0, 4.0, 5.0, 6.0, 7.0, 9.0],
    'win_dd_hot_n': [19.0, 24.0, 34.0, 40.0, 49.0, 63.0, 32.0, 39.0, 45.0, 52.0, 61.0, 75.0],
    'win_dd_hot_p': [0.479664142251255, 0.153071965718361, 0.12541235116199, 0.274585106527699, 0.088184545711072, 0.0630785413993254],
    'win_dd_hot_var': [None, None, None, None, None, None],
    'win_dd_hot_binom': [0.0919145498158544, 0.0769258336096096, 0.260425525897245, 0.251322103474855, 0.294233708444292, 0.349115166376781],
    'win_dd_hot_rec': [-4.0, 4.0],
    'win_dd_hot_summary': [65.0, 7.0, 7.0, 7.0, 14.0, 85.0, 8.0, 8.0, 16.0, 20.0],
    'win_dh_default_left': [-0.1, -0.4, -0.7, -1.0],
    'win_dh_default_right': [0.1, 0.4, 0.7, 1.0],
    'win_dh_default_n': [18.0, 54.0, 103.0, 147.0, 29.0, 68.0, 112.0, 153.0],
    'win_dh_default_p': [0.648531074028737, 0.0142703609164466, 1.82268080013386e-06, 2.20302079422417e-08],
    'win_dh_default_var': [2.0, 1.0, 1.0, 1.0],
    'win_dh_default_binom': [0.143865088440549, 0.239088812476915, 0.585447145772406, 0.772886543090602],
    'win_dh_default_rec': [-0.1, 0.1],
    'win_dh_default_summary': [147.0, 18.0, 18.0, 18.0, 30.0, 153.0, 24.0, 24.0, 24.0, 45.0],
    'win_dh_asym_left': [-0.1, -0.4, -0.6, -0.9, -1.0],
    'win_dh_asym_right': [0.0, 0.3, 0.6, 0.7, 1.0],
    'win_dh_asym_n': [18.0, 54.0, 92.0, 136.0, 147.0, 24.0, 58.0, 91.0, 112.0, 153.0],
    'win_dh_asym_p': [0.290236743991264, 0.0321697393122133, 0.000377340787755081, 1.17698653765364e-06, 2.20302079422417e-08],
    'win_dh_asym_var': [2.0, 1.0, 1.0, 1.0, 1.0],
    'win_dh_asym_binom': [0.440799067342597, 0.776964751369673, 1.0, 0.143999817854825, 0.772886543090602],
    'win_dh_asym_rec': [-0.1, 0.0],
    'win_dh_asym_summary': [147.0, 18.0, 18.0, 18.0, 30.0, 153.0, 24.0, 24.0, 24.0, 45.0],
    'win_d_hot6_left': [-0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4, -0.45],
    'win_d_hot6_right': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
    'win_d_hot6_n': [11.0, 17.0, 22.0, 25.0, 30.0, 36.0, 40.0, 44.0, 8.0, 12.0, 23.0, 27.0, 29.0, 35.0, 42.0, 46.0],
    'win_d_hot6_p': [0.0301983566968723, 0.0506421878848826, 0.0312498945963232, 0.0321793032261946, 0.0165407609429593, 0.0808631211777031, 0.253437619522866, 0.254749662957882],
    'win_d_hot6_var': [None, None, None, None, None, None, None, None],
    'win_d_hot6_binom': [0.647605895996094, 0.458258319646121, 1.0, 0.889883965276538, 1.0, 1.0, 0.912156607552603, 0.91612887701129],
    'win_d_hot6_rec': [None, None],
    'win_d_hot6_summary': [99.0, 1.0, 5.0, 10.0, 20.0, 101.0, 2.0, 6.0, 11.0, 21.0],
    'win_d_rank6_left': [-0.047, -0.121, -0.172, -0.272, -0.318, -0.37, -0.44, -0.513, -0.579, -0.659],
    'win_d_rank6_right': [0.088, 0.169, 0.183, 0.253, 0.336, 0.385, 0.449, 0.482, 0.538, 0.625],
    'win_d_rank6_n': [8.0, 14.0, 20.0, 26.0, 32.0, 38.0, 44.0, 50.0, 56.0, 62.0, 8.0, 15.0, 22.0, 28.0, 34.0, 40.0, 46.0, 53.0, 59.0, 65.0],
    'win_d_rank6_p': [0.115183731515051, 0.0361806167443217, 0.0284492617820354, 0.0120656350186657, 0.020916486750767, 0.0455708969824861, 0.0657163209068848, 0.12909523478991, 0.126615097670678, 0.119831529128838],
    'win_d_rank6_var': [5.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
    'win_d_rank6_binom': [1.0, 1.0, 0.877614328752315, 0.89192315110475, 0.902158500096702, 0.909946451874527, 0.91612887701129, 0.843897655252862, 0.852155526629713, 0.85922781565997],
    'win_d_rank6_rec': [None, None],
    'win_d_rank6_summary': [99.0, 1.0, 5.0, 10.0, 20.0, 101.0, 2.0, 6.0, 11.0, 21.0],
    'win_d_ks6_left': [-0.1, -0.295, -0.353, -0.422, -0.513, -0.614, -0.694, -0.791, -0.854, -0.902],
    'win_d_ks6_right': [0.1, 0.295, 0.353, 0.422, 0.513, 0.614, 0.694, 0.791, 0.854, 0.902],
    'win_d_ks6_n': [11.0, 29.0, 36.0, 43.0, 50.0, 59.0, 66.0, 77.0, 84.0, 91.0, 8.0, 29.0, 36.0, 45.0, 56.0, 63.0, 73.0, 80.0, 88.0, 95.0],
    'win_d_ks6_p': [0.101214574898785, 0.120734774311485, 0.0680654200974495, 0.0864731645405594, 0.04728526574602, 0.0321784430023144, 0.013516612681228, 0.0063567940412632, 0.0014705420098792, 0.00143804390653632],
    'win_d_ks6_var': [5.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
    'win_d_ks6_binom': [0.647605895996094, 1.0, 1.0, 0.915186504842878, 0.627430977078634, 0.78605785689386, 0.610973689956023, 0.873248139297002, 0.819147252434749, 0.825967627487243],
    'win_d_ks6_rec': [None, None],
    'win_d_ks6_summary': [99.0, 1.0, 5.0, 10.0, 20.0, 101.0, 2.0, 6.0, 11.0, 21.0],
    'win_d_p2_6_left': [-0.15, -0.2, -0.25, -0.3, -0.35, -0.4],
    'win_d_p2_6_right': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    'win_d_p2_6_n': [17.0, 22.0, 25.0, 30.0, 36.0, 40.0, 12.0, 23.0, 27.0, 29.0, 35.0, 42.0],
    'win_d_p2_6_p': [0.0866344373549539, 0.0265880124312989, 0.196843620664372, 0.251222793531637, 0.164791013055495, 0.137415604323756],
    'win_d_p2_6_var': [2.0, 6.0, 6.0, 5.0, 5.0, 5.0],
    'win_d_p2_6_binom': [0.458258319646121, 1.0, 0.889883965276538, 1.0, 1.0, 0.912156607552603],
    'win_d_p2_6_rec': [None, None],
    'win_d_p2_6_summary': [99.0, 1.0, 5.0, 10.0, 20.0, 101.0, 2.0, 6.0, 11.0, 21.0],
    'win_d_epan6_left': [-0.1, -0.15, -0.2, -0.25, -0.3, -0.35],
    'win_d_epan6_right': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
    'win_d_epan6_n': [11.0, 17.0, 22.0, 25.0, 30.0, 36.0, 8.0, 12.0, 23.0, 27.0, 29.0, 35.0],
    'win_d_epan6_p': [0.105192183717675, 0.0305617835687224, 0.0413806875011567, 0.0179545934857179, 0.0087265632626936, 0.006424567287037],
    'win_d_epan6_var': [5.0, 5.0, 5.0, 3.0, 3.0, 3.0],
    'win_d_epan6_binom': [0.647605895996094, 0.458258319646121, 1.0, 0.889883965276538, 1.0, 1.0],
    'win_d_epan6_rec': [None, None],
    'win_d_epan6_summary': [99.0, 1.0, 5.0, 10.0, 20.0, 101.0, 2.0, 6.0, 11.0, 21.0],
    'ri_T1_stat': [0.985805058449475],
    'ri_T1_asy': [0.0133080790152533],
    'ri_T1_sum': [99.0, 40.0, 0.748525, 1.17946650233409, -0.4, 101.0, 35.0, 1.69565714285714, 1.13800932857898, 0.35],
    'ri_T1_ci': [0.205272881830836, 1.76633723506811],
    'ri_T2_stat': [0.181526141983922],
    'ri_T2_asy': [0.975191513684548],
    'ri_T2_sum': [99.0, 44.0, 0.690636363636364, 1.24973563885288, -0.45, 101.0, 29.0, 1.76686206896552, 1.19504213088128, 0.3],
    'ri_T2_se': [0.594056020325504],
    'ri_T3_stat': [0.0561567444878545],
    'ri_T3_asy': [0.91916232881653],
    'ri_T3_sum': [99.0, 40.0, 0.748525, 1.17946650233409, -0.4, 101.0, 42.0, 1.73128571428571, 1.15738330361996, 0.4],
    'ri_T3_se': [0.553328657320163],
    'ri_T4_stat': [0.570581009029966],
    'ri_T4_asy': [0.170695412111078],
    'ri_T4_sum': [99.0, 49.0, 0.652795918367347, 1.19849591467097, -0.5, 101.0, 46.0, 1.74297826086957, 1.13940123630558, 0.45],
    'ri_T4_se': [0.416490776418661],
    'ri_T5_stat': [0.33710245486587],
    'ri_T5_asy': [0.69812280658769],
    'ri_T5_sum': [99.0, 44.0, 0.690636363636364, 1.24973563885288, -0.45, 101.0, 29.0, 1.76686206896552, 1.19504213088128, 0.3],
    'ri_T5_se': [0.86914395104455],
    'ri_T5_ci': [-0.992512125352691, 1.86671703508443],
    'ri_T6_stat': [0.325837716963972],
    'ri_T6_asy': [0.542533003976742],
    'ri_T6_sum': [99.0, 44.0, 0.690636363636364, 1.24973563885288, -0.45, 101.0, 56.0, 1.80767857142857, 1.09367643722933, 0.5],
    'ri_T6_se': [0.535048654832966],
    'ri_T6_ci': [-0.5, 1.2],
    'ri_T7_stat': [0.764725714285714, 0.317142857142857, -2.2168567345484],
    'ri_T7_asy': [0.0117904392141889, 0.0628844175106102, 0.0266328831045652],
    'ri_T7_sum': [99.0, 25.0, 0.96656, 1.227967462924, -0.25, 101.0, 42.0, 1.73128571428571, 1.15738330361996, 0.4],
    'ri_T8_stat': [1.19, 0.564285714285714, -3.00937311256219],
    'ri_T8_asy': [0.0018509585071795, 0.00576038959992145, 0.0026178740376017],
    'ri_T8_sum': [65.0, 14.0, 0.195, 1.18145119369228, -2.0, 85.0, 20.0, 1.385, 0.963450487625214, 2.0],
    'ri_T9_stat': [1.97144568186594],
    'ri_T9_asy': [0.0631241802172227],
    'ri_T9_sum': [65.0, 19.0, 0.366842105263158, 1.38747193217488, -3.0, 85.0, 32.0, 1.115, 1.00740804431071, 3.0],
    'ri_T9_se': [1.06087763608694],
    'ri_T10_stat': [-3.76820082364141],
    'ri_T10_asy': [0.000164428398838709],
    'ri_T10_sum': [65.0, 24.0, 1.41666666666667, 1.05980583400786, -4.0, 85.0, 32.0, 2.65625, 1.00352202348175, 3.0],
    'ri_T11_stat': [-0.77368842364532],
    'ri_T11_asy': [0.282093305945284],
    'ri_T11_sum': [73.0, 15.0, 0.875333333333333, 1.17557807461853, -1.5, 77.0, 24.0, 0.985833333333333, 0.976622024997463, 3.5],
    'ri_T11_se': [0.719289433569724],
    'ri_T11_ci': [-2.18346980790219, 0.636092960611554],
    'ri_T12_stat': [1.23958333333333, 0.427083333333333, -3.65902063639218],
    'ri_T12_asy': [9.39026678981486e-06, 0.00243775729695952, 0.000253180954352272],
    'ri_T12_sum': [65.0, 24.0, 1.41666666666667, 1.05980583400786, -4.0, 85.0, 32.0, 2.65625, 1.00352202348175, 3.0],
    'ri_T12_ranksum_stat': [-3.76820082364141, 0.000164428398838709],
    'ri_covsel_window': [-0.427, 0.627],
    'ri_covsel_stat': [1.11162102772143, 1.03175047424596e-06],
    'sens_kernel_results': [0.201216472779946, 0.604360834329678, 0.80892417001671, 0.316455672347647, 0.0781163168011137, 0.237384354708857, 0.680498746625489, 0.720354846238864, 0.259424746578599, 0.057759467197117, 0.154239962461808, 0.572771126329007, 0.766646926415108, 0.247067153343564, 0.043566153815993],
    'sens_kernel_ci': [-0.5, 1.0],
    'sens_fuzzy_results': [0.925066799908209, 0.444516281916093, 0.103949733736525, 0.014046882553472, 0.434810011223107, 0.98067774963724, 0.466310804923015, 0.146429050866554],
    'sens_dd_wlist': [3.0, 4.0, 5.0, 6.0, 7.0, 9.0],
    'sens_dd_tlist': [0.09, 0.22, 0.35, 0.48, 0.61, 0.74, 0.87, 1.0, 1.13, 1.26, 1.39],
    'sens_fuzzy_default_tlist': [0.21, 0.37, 0.53, 0.69, 0.85, 1.01, 1.17, 1.33, 1.49, 1.65],
    'logit_prob': [0.365168666552802, 0.390550216371675, 0.436347249774632, 0.375662489122496, 0.318646266210974, 0.363084824719586, 0.458595035960828, 0.354801395436818, 0.514745722759735, 0.30512721449923, 0.316695466033755, 0.50124999739584],
    'bern_logit_diffmeans': [0.435592941362288],
    'bern_logit_ranksum': [0.754689284512583],
    'bern_logit_ksmirnov': [0.704267736775973],
    'bern_logit_diffmeans_tau1': [0.645933979769759],
    'bern_ar_tau05': [0.855579434791744],
    'exact_ar_triangular_tau04': [0.8997668997669, 6435.0],
    'exact_ar_uniform_tau04': [0.246775446775447, 6435.0],
    'hot_n': [8.0, 7.0],
    'hot5_exact': [0.176068376068376, 6435.0],
    'hot6_exact': [0.177622377622378, 6435.0],
    'hot6_F': [0.162834717051395],
    'disc_n': [7.0, 8.0],
    'disc_ranksum': [0.0132090132090132, 6435.0],
    'disc_ksmirnov': [0.0590520590520591, 6435.0],
    'disc_diffmeans': [0.0203574203574204, 6435.0],
    'disc_kstest': [0.0590520590520591],
    'disc_stats': [-2.43863362608329, 0.607142857142857],
    'rb_diffmeans_base': [0.457549531235773],
    'rb_diffmeans_G1.5_u': [0.431167726522433, 0.441257742110763, 0.448874437856113, 0.458986705456004, 0.473089936242599, 0.486779795030989, 0.488018664778911, 0.489570082136757, 0.480035935827471, 0.453477539143018, 0.439209807481204, 0.431544459366966],
    'rb_diffmeans_G3_u': [0.516223730676733, 0.562628251270266, 0.608648327501718, 0.655870455000527, 0.712108995011918, 0.759978297254884, 0.764885346376401, 0.770345248767275, 0.742060065501394, 0.639838671222033, 0.566108833543192, 0.509030594664188],
    'rb_ksmirnov_base': [0.741557692891155],
    'rb_ksmirnov_G1.5_u': [0.690508630219414, 0.701768906796884, 0.71398182715069, 0.714706223713478, 0.715701542246421, 0.71885969504611, 0.715701542246423, 0.714706223713477, 0.713981827150685, 0.701768906796873, 0.690508630219408, 0.697408881706125],
    'rb_ksmirnov_G3_u': [0.785052661838814, 0.844345292106941, 0.883418690108182, 0.886149871793297, 0.888071299389453, 0.8957645907863, 0.888071299389453, 0.886149871793297, 0.883418690108182, 0.844345292106941, 0.785052661838814, 0.810107662460404],
    'zero_tsls_package': [1.54976123595506, 1.40348640724043e-07],
    'zero_hc2_package': [1.19418831168831, 1.17151682816185e-06],
    'zero_weights': [17.0, 51.0],
    'zero_tsls_positive': [1.54976123595506, 0.446120630807166, 0.000513025975440755],
    'zero_hc2_positive': [1.19418831168831, 0.369125659249781, 0.00121565649915968],
}


def _minstd(seed, n):
    s, values = seed, []
    for _ in range(n):
        s = (48271 * s) % 2147483647
        values.append(s / 2147483647)
    return values


def _normals(seed, n):
    u = _minstd(seed, 12 * n)
    values = []
    for i in range(n):
        acc = 0.0
        for j in range(12):
            acc += u[12 * i + j]
        values.append(acc - 6)
    return values


def _rnd(x, digits):
    return math.floor(x * 10 ** digits + 0.5) / 10 ** digits


def dataset(n=200):
    """The continuous design of rdlocrand_reference.R plus covariates x3..x6."""
    u, v = _minstd(20261010, n), _minstd(1234567891, n)
    z1, z2, z3 = _normals(987654321, n), _normals(1122334455, n), _normals(2024681357, n)
    r = [_rnd(2 * a - 1, 3) for a in u]
    t = [1.0 if (ri >= 0 and b < 0.85) or (ri < 0 and b < 0.15) else 0.0 for ri, b in zip(r, v)]
    x1 = [_rnd(0.5 * ri + a, 3) for ri, a in zip(r, z1)]
    x2 = [_rnd(b + (0.6 if ri > 0.5 else -0.6 if ri < -0.5 else 0.0), 3) for ri, b in zip(r, z2)]
    y = [_rnd(1 + ri - 0.4 * ri * ri + 0.7 * ti + c, 3) for ri, ti, c in zip(r, t, z3)]
    x3 = [_rnd(z + 0.3 * a, 3) for z, a in zip(_normals(31415, n), x1)]
    x4 = [_rnd(z - 0.2 * ri, 3) for z, ri in zip(_normals(27182, n), r)]
    x5 = [_rnd(z + 0.5 * a, 3) for z, a in zip(_normals(16180, n), x2)]
    x6 = [_rnd(z, 3) for z in _normals(14142, n)]
    return pd.DataFrame({"r": r, "y": y, "t": t, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5,
                         "x6": x6})


def discrete(n=150):
    """21 mass points, a fuzzy treatment, a continuous and an integer (tied) outcome."""
    rd = [math.floor(u * 21) - 10 for u in _minstd(424242, n)]
    v = _minstd(515151, n)
    td = [1.0 if (r >= 0 and b < 0.8) or (r < 0 and b < 0.2) else 0.0 for r, b in zip(rd, v)]
    yd = [_rnd(0.5 + 0.08 * r + 0.6 * t + z, 2) for r, t, z in zip(rd, td, _normals(616161, n))]
    yc = [math.floor(u * 4) + (r >= 0) for u, r in zip(_minstd(717171, n), rd)]
    x1 = [_rnd(z + 0.03 * r, 3) for z, r in zip(_normals(818181, n), rd)]
    x2 = [_rnd(z, 3) for z in _normals(919191, n)]
    x3 = [_rnd(z - 0.02 * r, 3) for z, r in zip(_normals(101010, n), rd)]
    return pd.DataFrame({"r": rd, "y": yd, "yc": yc, "t": td, "x1": x1, "x2": x2, "x3": x3})


def heaped(n=300):
    """A running variable rounded to one decimal (21 heaps)."""
    rh = [_rnd(2 * u - 1, 1) for u in _minstd(232323, n)]
    return pd.DataFrame({"r": rh, "x1": [_rnd(z + 0.4 * r, 3) for z, r in zip(_normals(343434, n), rh)],
                         "x2": [_rnd(z, 3) for z in _normals(454545, n)]})


D, DD, DH = dataset(), discrete(), heaped()
X6 = [f"x{i}" for i in range(1, 7)]
C3 = ["x1", "x2", "x3"]


def close(actual, expected, rtol=1e-8, atol=1e-11):
    actual = [None if a is None or (isinstance(a, float) and math.isnan(a)) else float(a) for a in actual]
    assert len(actual) == len(expected), (actual, expected)
    for a, e in zip(actual, expected):
        if e is None:
            assert a is None, (actual, expected)
        else:
            assert a is not None and abs(a - e) <= atol + rtol * abs(e), (actual, expected)


def test_data_are_regenerated_bit_for_bit():
    close([D[c].sum() for c in ("x3", "x4", "x5", "x6")], REF["check_d"], rtol=1e-12)
    close([DD[c].sum() for c in ("r", "y", "yc", "t", "x1", "x2", "x3")], REF["check_dd"], rtol=1e-12)
    close([DH[c].sum() for c in ("r", "x1", "x2")], REF["check_dh"], rtol=1e-12)


# ---- window selection: mass points, heaps, many covariates --------------------------------------

WIN_CASES = {
    "dd_default": (DD, C3, {}),
    "dd_wobs": (DD, C3, dict(obsmin=15, wobs=25)),
    "dd_asym": (DD, C3, dict(obsmin=12, wobs=20, wasymmetric=True)),
    "dd_wmin": (DD, C3, dict(wmin=2.5, wobs=18)),
    "dd_pairasym": (DD, C3, dict(wmin=[-3, 1], wobs=15, wasymmetric=True)),
    "dd_step": (DD, C3, dict(wmin=1, wstep=1, nwindows=8)),
    "dd_cut": (DD, C3, dict(cutoff=0.5, wasymmetric=True)),
    "dd_hot": (DD, C3, dict(statistic="hotelling")),
    "dh_default": (DH, ["x1", "x2"], dict(obsmin=15, wobs=20)),
    "dh_asym": (DH, ["x1", "x2"], dict(obsmin=15, wobs=20, wasymmetric=True)),
    "d_hot6": (D, X6, dict(statistic="hotelling", wmin=0.1, wstep=0.05, nwindows=8)),
    "d_rank6": (D, X6, dict(statistic="ranksum", obsmin=8, wobs=6, wasymmetric=True)),
    "d_ks6": (D, X6, dict(statistic="ksmirnov", wmin=[-0.2, 0.1], wobs=7)),
    "d_p2_6": (D, X6, dict(p=2, vce="HC2", wmin=0.15, wstep=0.05, nwindows=6)),
    "d_epan6": (D, X6, dict(kernel="epan", wmin=0.1, wstep=0.05, nwindows=6)),
}


@pytest.mark.parametrize("case", sorted(WIN_CASES))
def test_rdwinselect_matches_r_on_mass_points_heaps_and_many_covariates(case):
    data, covariates, options = WIN_CASES[case]
    res = oe.rdwinselect(data=data, running="r", covariates=covariates, approx=True, **options)
    cut = options.get("cutoff", 0.0)
    windows = res["windows"]
    close([v - cut for v in windows["w_left"]], REF[f"win_{case}_left"], atol=1e-12)
    close([v - cut for v in windows["w_right"]], REF[f"win_{case}_right"], atol=1e-12)
    close([*windows["n_left"], *windows["n_right"]], REF[f"win_{case}_n"])
    close(windows["p_value"], REF[f"win_{case}_p"], rtol=1e-7)
    variable = [None if v == "hotelling" else covariates.index(v) + 1 for v in windows["variable"]]
    close(variable, REF[f"win_{case}_var"])
    close(windows["binomial_p_value"], REF[f"win_{case}_binom"], rtol=1e-9)
    close([res.attrs["recommended_left"], res.attrs["recommended_right"]], REF[f"win_{case}_rec"],
          atol=1e-12)
    s = res["sample"]
    close([s[col][side] for side in (0, 1) for col in ("n", "n_1pct", "n_5pct", "n_10pct", "n_20pct")],
          REF[f"win_{case}_summary"])


# ---- rdrandinf: kernels with fuzzy designs, asymmetric windows, discrete running variables -------

RI_CASES = {
    "T1": (D, "y", dict(wl=-0.4, wr=0.35, fuzzy="t", fuzzy_statistic="tsls", kernel="triangular",
                        ci=0.05)),
    "T2": (D, "y", dict(wl=-0.45, wr=0.3, fuzzy="t", fuzzy_statistic="tsls", kernel="epan", p=1,
                        vce="HC2", nulltau=0.2)),
    "T3": (D, "y", dict(wl=-0.4, wr=0.4, fuzzy="t", kernel="triangular", p=1, vce="HC1", nulltau=0.3)),
    "T4": (D, "y", dict(wl=-0.5, wr=0.45, fuzzy="t", p=2, evall=-0.05, evalr=0.1, vce="HC2")),
    "T5": (D, "y", dict(wl=-0.45, wr=0.3, p=2, kernel="triangular", nulltau=0.1, ci=0.1)),
    "T6": (D, "y", dict(wl=-0.45, wr=0.5, p=1, kernel="epan", ci=0.1,
                        ci_grid=[round(-1 + 0.1 * i, 10) for i in range(36)])),
    "T7": (D, "y", dict(wl=-0.25, wr=0.4, statistic="all", d=0.4)),
    "T8": (DD, "y", dict(wl=-2, wr=2, statistic="all")),
    "T9": (DD, "y", dict(wl=-3, wr=3, fuzzy="t", fuzzy_statistic="tsls", p=1, vce="HC3")),
    "T10": (DD, "yc", dict(wl=-4, wr=3, statistic="ranksum")),
    "T11": (DD, "y", dict(cutoff=0.5, wl=-1.5, wr=3.5, p=1, vce="HC1", ci=0.05)),
}


@pytest.mark.parametrize("case", sorted(RI_CASES))
def test_rdrandinf_deterministic_output_matches_r_in_new_configurations(case):
    data, y, options = RI_CASES[case]
    res = oe.rdrandinf(data=data, y=y, running="r", reps=100, **options)
    tests, sample = res["tests"], res["sample"]
    close(tests["statistic"], REF[f"ri_{case}_stat"])
    close(tests["asymptotic_p_value"], REF[f"ri_{case}_asy"], rtol=1e-7)
    close([sample[col][side] for side in (0, 1)
           for col in ("n", "n_window", "mean", "std_dev", "window_limit")], REF[f"ri_{case}_sum"])
    if f"ri_{case}_se" in REF:
        close([tests["std_error"][0]], REF[f"ri_{case}_se"])
    if f"ri_{case}_ci" in REF:
        close([res["ci"]["ci_low"][0], res["ci"]["ci_high"][0]], REF[f"ri_{case}_ci"])


def test_all_uses_the_tie_corrected_rank_sum_variance_of_the_package_ranksum():
    # Tied integer outcome: rdlocrand's 'all' standardizes with var(1:n), its 'ranksum' with
    # var(rank(Y)); the port uses the exact permutation variance (the latter) for both.
    res = oe.rdrandinf(data=DD, y="yc", running="r", wl=-4, wr=3, statistic="all", reps=100)
    statistic, asymptotic = list(res["tests"]["statistic"]), list(res["tests"]["asymptotic_p_value"])
    close(statistic[:2], REF["ri_T12_stat"][:2])
    close(asymptotic[:2], REF["ri_T12_asy"][:2], rtol=1e-7)
    close([statistic[2], asymptotic[2]], REF["ri_T12_ranksum_stat"], rtol=1e-9)
    assert abs(statistic[2] - REF["ri_T12_stat"][2]) > 0.1


def test_window_selected_inside_rdrandinf_matches_r():
    res = oe.rdrandinf(data=D, y="y", running="r", cutoff=0.1, covariates=["x1", "x2"], approx=True,
                       reps=100)
    close([res.attrs["window_left"], res.attrs["window_right"]], REF["ri_covsel_window"], atol=1e-12)
    close([res["tests"]["statistic"][0], res["tests"]["asymptotic_p_value"][0]], REF["ri_covsel_stat"])


def test_rdsensitivity_matches_r_with_kernels_fuzzy_designs_and_mass_points():
    res = oe.rdsensitivity(data=D, y="y", running="r", wlist=[0.2, 0.3, 0.4],
                           wlist_left=[-0.3, -0.3, -0.35], tlist=[-0.5, 0, 0.5, 1, 1.5], p=1,
                           kernel="triangular", vce="HC2", ci=[-0.3, 0.3], ci_alpha=0.1)
    close(res["pvalues"]["p_value"], REF["sens_kernel_results"])
    close([res["ci"]["ci_low"][0], res["ci"]["ci_high"][0]], REF["sens_kernel_ci"])
    res = oe.rdsensitivity(data=D, y="y", running="r", wlist=[0.3, 0.4], tlist=[0, 0.5, 1, 1.5], p=1,
                           fuzzy="t")
    close(res["pvalues"]["p_value"], REF["sens_fuzzy_results"])
    res = oe.rdsensitivity(data=DD, y="y", running="r", reps=10)
    close(res["windows"]["w_right"], REF["sens_dd_wlist"])
    close(res.attrs["tlist"], REF["sens_dd_tlist"])
    res = oe.rdsensitivity(data=D, y="y", running="r", wlist=[0.25, 0.35], fuzzy="t", reps=10)
    close(res.attrs["tlist"], REF["sens_fuzzy_default_tlist"])


# ---- exact randomization laws enumerated in R ---------------------------------------------------


def test_bernoulli_with_logistic_unit_probabilities_matches_r_enumeration():
    inside = (D.r.round(8) >= -0.06) & (D.r.round(8) <= 0.06)
    data = D.assign(prob=0.5)
    data.loc[inside, "prob"] = REF["logit_prob"]
    res = oe.rdrandinf(data=data, y="y", running="r", wl=-0.06, wr=0.06, statistic="all",
                       bernoulli="prob", method="exact")
    assert res.attrs["randomization"] == "Bernoulli" and res.attrs["draws"] == 2 ** 12
    close(res["tests"]["p_value"], [REF["bern_logit_diffmeans"][0], REF["bern_logit_ksmirnov"][0],
                                    REF["bern_logit_ranksum"][0]], rtol=1e-10)
    res = oe.rdrandinf(data=data, y="y", running="r", wl=-0.06, wr=0.06, nulltau=1.0, bernoulli="prob",
                       method="exact")
    close(res["tests"]["p_value"], REF["bern_logit_diffmeans_tau1"], rtol=1e-10)
    # Row order and the probability column travel together through the canonical order.
    shuffled = data.sample(frac=1.0, random_state=3).reset_index(drop=True)
    again = oe.rdrandinf(data=shuffled, y="y", running="r", wl=-0.06, wr=0.06, nulltau=1.0,
                         bernoulli="prob", method="exact")
    assert again["tests"]["p_value"][0] == res["tests"]["p_value"][0]


def test_bernoulli_and_kernel_weighted_anderson_rubin_match_r_enumeration():
    # rdrandinf in R fails for fuzzy + bernoulli; the enumerated law uses its diffmeans on Y - tau T.
    res = oe.rdrandinf(data=D, y="y", running="r", wl=-0.06, wr=0.06, fuzzy="t", nulltau=0.5,
                       bernoulli=0.35, method="exact")
    close(res["tests"]["p_value"], REF["bern_ar_tau05"], rtol=1e-10)
    for kernel, key in (("triangular", "exact_ar_triangular_tau04"), ("uniform", "exact_ar_uniform_tau04")):
        res = oe.rdrandinf(data=D, y="y", running="r", wl=-0.08, wr=0.07, fuzzy="t", nulltau=0.4,
                           kernel=kernel, method="exact")
        close([res["tests"]["p_value"][0], res.attrs["draws"]], REF[key], rtol=1e-12)


@pytest.mark.parametrize("columns, key", [(5, "hot5_exact"), (6, "hot6_exact")])
def test_hotelling_with_many_covariates_matches_r_enumeration(columns, key):
    res = oe.rdwinselect(data=D, running="r", covariates=X6[:columns], statistic="hotelling", wmin=0.07,
                         nwindows=1, method="exact")
    windows = res["windows"]
    close([windows["n_left"][0], windows["n_right"][0]], REF["hot_n"])
    close([windows["p_value"][0], windows["draws"][0]], REF[key], rtol=1e-12)
    if columns == 6:
        res = oe.rdwinselect(data=D, running="r", covariates=X6, statistic="hotelling", wmin=0.07,
                             nwindows=1, approx=True)
        close(res["windows"]["p_value"], REF["hot6_F"], rtol=1e-9)


def test_tied_outcome_on_two_mass_points_matches_r_enumeration():
    res = oe.rdrandinf(data=DD, y="yc", running="r", wl=-1, wr=0, statistic="all", method="exact")
    close([res.attrs["n_left"], res.attrs["n_right"]], REF["disc_n"])
    tests = res["tests"]
    close(tests["p_value"], [REF["disc_diffmeans"][0], REF["disc_ksmirnov"][0], REF["disc_ranksum"][0]],
          rtol=1e-12)
    close([tests["asymptotic_p_value"][1]], REF["disc_kstest"], rtol=1e-10)
    close([tests["statistic"][2], tests["statistic"][1]], REF["disc_stats"])


@pytest.mark.parametrize("statistic", ["diffmeans", "ksmirnov"])
def test_rosenbaum_bounds_are_the_extremes_over_the_package_configurations(statistic):
    res = oe.rdrbounds(data=D, y="y", running="r", wlist=[0.06], expgamma=[1.5, 3.0], statistic=statistic,
                       method="exact")
    close(res["pvalues"]["p_value"], REF[f"rb_{statistic}_base"], rtol=1e-10)
    for row, gamma in enumerate(("1.5", "3")):
        values = REF[f"rb_{statistic}_G{gamma}_u"]
        close([res["bounds"]["lower_bound"][row], res["bounds"]["upper_bound"][row]],
              [min(values), max(values)], rtol=1e-10)
    assert "largest outcomes" in res.attrs["bound_search"]


def test_zero_kernel_weights_use_the_positive_weight_sandwich():
    # Units at r = -3 and r = 3 get triangular weight 0 in the window [-3, 3].
    assert REF["zero_weights"] == [17.0, 51.0]
    res = oe.rdrandinf(data=DD, y="y", running="r", wl=-3, wr=3, fuzzy="t", fuzzy_statistic="tsls",
                       kernel="triangular")
    tests = res["tests"]
    close([tests["statistic"][0], tests["std_error"][0], tests["asymptotic_p_value"][0]],
          REF["zero_tsls_positive"], rtol=1e-9)
    res = oe.rdrandinf(data=DD, y="y", running="r", wl=-3, wr=3, kernel="triangular", reps=100)
    tests = res["tests"]
    close([tests["statistic"][0], tests["std_error"][0], tests["asymptotic_p_value"][0]],
          REF["zero_hc2_positive"], rtol=1e-9)
    # rdlocrand's p = 0 kernel path (sandwich::vcovHC with zero weights) is anticonservative here.
    assert REF["zero_tsls_package"][1] < REF["zero_tsls_positive"][2] / 100
    assert REF["zero_hc2_package"][1] < REF["zero_hc2_positive"][2] / 100


def test_kernel_companion_is_undefined_with_one_positive_weight_per_side():
    data = pd.DataFrame({"r": [-0.2, -0.1, 0.1, 0.2, 0.5], "y": [1.0, 2.0, 3.0, 5.0, 0.0]})
    res = oe.rdrandinf(data=data, y="y", running="r", wl=-0.2, wr=0.2, kernel="triangular")
    row = res["tests"].iloc[0]
    # Only -0.1 and 0.1 have positive weight: every admissible split keeps one on each side.
    assert row["statistic"] == 1.0 and row["p_value"] == 1.0 and res.attrs["draws"] == 6
    assert row["asymptotic_p_value"] is None and row["std_error"] is None
    assert any("HC2" in note for note in res.attrs["notes"])


def _single_treated_data():
    rng = np.random.default_rng(1)
    r = np.round(rng.uniform(-1, 1, 200), 3)
    r[np.argmin(np.abs(r - 0.001))] = 0.001
    return pd.DataFrame({"r": r, "x1": rng.normal(size=200), "x2": rng.normal(size=200)})


@pytest.mark.parametrize("options", [dict(), dict(kernel="triangular")])
def test_approximate_balance_with_one_unit_on_a_side_is_undefined(options):
    data = _single_treated_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = oe.rdwinselect(data=data, running="r", covariates=["x1", "x2"], wmin=[-0.2, 0.001], wobs=5,
                             wasymmetric=True, nwindows=3, approx=True, **options)
    windows = res["windows"]
    assert windows["n_right"][0] == 1 and windows["n_right"][1] > 1
    # Was 0.0 (a spurious rejection) or an unit_leverage abort; the package reports NaN.
    assert windows["p_value"][0] is None or math.isnan(windows["p_value"][0])
    assert windows["variable"][0] is None and windows["p_value"][1] > 0
    assert res.attrs["recommended_window"] is None
    randomized = oe.rdwinselect(data=data, running="r", covariates=["x1", "x2"], wmin=[-0.2, 0.001],
                                wobs=5, wasymmetric=True, nwindows=1, reps=200, **options)
    assert 0 < randomized["windows"]["p_value"][0] <= 1


def _large(n, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(-1, 1, n)
    return pd.DataFrame({"r": r, "y": r + rng.normal(size=n), "x1": rng.normal(size=n)})


def test_exact_refusal_on_large_windows_is_fast_and_clean():
    # C(20000, 10000) and 2^20000 have about 6,000 digits: the old message formatting raised
    # a raw ValueError and computing C(10^6, 5 10^5) alone took seconds.
    data = _large(20_000)
    for options in (dict(), dict(bernoulli=0.5)):
        start = time.perf_counter()
        with pytest.raises(AnalysisError) as error:
            oe.rdrandinf(data=data, y="y", running="r", wl=-1, wr=1, method="exact", **options)
        assert error.value.code == "work_budget_exceeded" and "10,000,000" in str(error.value)
        assert time.perf_counter() - start < 10


def test_assignment_counts_are_exact_below_the_cap_and_capped_above():
    for n in range(1, 31):
        for n1 in range(n + 1):
            assert k.fixed_support(n, n1, 10 ** 12) == math.comb(n, n1)
    assert k.fixed_support(10 ** 7, 5 * 10 ** 6, 10 ** 7) == 10 ** 7 + 1
    assert k.fixed_support(10 ** 7, 1, 10 ** 7) == 10 ** 7
    assert k.plan(k.fixed_support(40, 20, 10 ** 7), 1000, "auto") == ("monte_carlo", 1000)
    assert k.plan(k.fixed_support(12, 4, 10 ** 7), 1000, "auto") == ("exact", 495)
    with pytest.raises(AnalysisError):
        k.plan(k.fixed_support(40, 20, 10 ** 7), 1000, "exact")


def test_window_procedures_check_their_total_work_before_drawing():
    data = _large(50_000)
    start = time.perf_counter()
    with pytest.raises(AnalysisError) as error:
        oe.rdwinselect(data=data, running="r", covariates=["x1"], wmin=0.5, wstep=1e-4, nwindows=1000,
                       reps=1000)
    assert error.value.code == "work_budget_exceeded"
    with pytest.raises(AnalysisError) as error:
        oe.rdsensitivity(data=data, y="y", running="r", wlist=[0.5 + 1e-4 * i for i in range(200)],
                         tlist=[0.0, 1.0], reps=1000)
    assert error.value.code == "work_budget_exceeded"
    assert time.perf_counter() - start < 20
    # Large-sample window tests draw nothing and are not limited.
    res = oe.rdwinselect(data=data, running="r", covariates=["x1"], wmin=0.5, wstep=0.05, nwindows=5,
                         approx=True)
    assert len(res["windows"]) == 5


# ---- Rosenbaum search: NumPy enumeration of every high/low configuration --------------------------


def _bernoulli_pvalues(y, d, configurations):
    """Exact two-sided Bernoulli p-values of the difference in means for each probability row."""
    n = len(y)
    a = np.array(list(itertools.product([False, True], repeat=n)))
    a = a[(a.sum(1) > 0) & (a.sum(1) < n)]
    values = (a * y).sum(1) / a.sum(1) - (~a * y).sum(1) / (~a).sum(1)
    hit = np.abs(values) >= abs(y[d].mean() - y[~d].mean()) - 1e-10
    weights = np.prod(np.where(a[None], configurations[:, None, :], 1 - configurations[:, None, :]), axis=2)
    return (weights * hit).sum(1) / weights.sum(1)


def test_rosenbaum_search_sets_coincide_and_lower_bound_is_not_the_infimum():
    rng = np.random.default_rng(12)
    r = np.array([-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04])
    y = np.round(rng.normal(size=9) + 1.2 * (r >= 0), 1)
    data = pd.DataFrame({"r": [*r, -0.5, 0.5], "y": [*y, 0.0, 0.0]})
    gamma = 2.0
    low, high = 1 / (1 + gamma), gamma / (1 + gamma)
    order = np.argsort(-y, kind="stable")
    largest = np.full((9, 9), low)
    smallest = np.full((9, 9), low)
    for u in range(1, 10):
        largest[u - 1, order[:u]] = high
        smallest[u - 1, order[::-1][:u]] = high
    d = r >= 0
    p_largest, p_smallest = _bernoulli_pvalues(y, d, largest), _bernoulli_pvalues(y, d, smallest)
    every = np.array(list(itertools.product([low, high], repeat=9)))
    p_every = _bernoulli_pvalues(y, d, every)
    # Complement symmetry: the smallest-outcome search visits the same p-values.
    assert np.allclose(np.sort(p_largest), np.sort(p_smallest), atol=1e-12)
    res = oe.rdrbounds(data=data, y="y", running="r", wlist=[0.05], expgamma=[gamma],
                       statistic="diffmeans", method="exact")
    lower, upper = res["bounds"]["lower_bound"][0], res["bounds"]["upper_bound"][0]
    assert lower == pytest.approx(p_largest.min(), abs=1e-12)
    assert upper == pytest.approx(p_largest.max(), abs=1e-12)
    # Documented limitation: for these two-sided tests neither extreme over the 2^9 high/low
    # assignments needs to lie in the package's n configurations (here both do not).
    assert p_every.min() < lower - 0.01
    assert upper < p_every.max() - 0.005
