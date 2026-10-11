"""rdlocrand procedures against R rdlocrand 3.0 (R 4.6.1).

Numbers come from tests/reference/rdlocrand_reference.R (run offline; R is not
needed here). The data are regenerated bit for bit from the MINSTD integer
generator with sequential sums and explicit floor rounding. Deterministic
package output (large-sample statistics, HC1/HC2/HC3 polynomial fits, TSLS,
window sequences, approximate balance tests, default grids) must agree to
rounding. Randomization p-values are compared with brute-force enumeration of
every assignment in R using the package's own statistic function (the KS law
also from ks.test's exact tie-aware Smirnov p-value); Monte Carlo p-values only
within simulation error, since R's and torch's random streams differ.
"""

import math

import pandas as pd
import pytest

import openecon as oe

REF = {
    'check_sums': [-0.434000000000001, 230.69, 104.0, 21.936, 14.429],
    'ri_A_stat': [0.944695402298851, 0.383908045977011, -2.72915295688405],
    'ri_A_asy': [0.00279812143274082, 0.0207890114188599, 0.00634972443044461],
    'ri_A_sum': [99.0, 30.0, 0.822166666666667, 1.23256676627813, -0.3, 101.0, 29.0, 1.76686206896552, 1.19504213088128, 0.3],
    'ri_B_stat': [0.641812879517365],
    'ri_B_asy': [0.0788882219070592],
    'ri_B_sum': [99.0, 30.0, 0.822166666666667, 1.23256676627813, -0.3, 101.0, 27.0, 1.77740740740741, 1.22762808384735, 0.25],
    'ri_C_stat': [0.415125888175227],
    'ri_C_asy': [0.385099141581011],
    'ri_C_sum': [99.0, 49.0, 0.652795918367347, 1.19849591467097, -0.5, 101.0, 56.0, 1.80767857142857, 1.09367643722933, 0.5],
    'ri_C_se': [0.477958643227861],
    'ri_C_ci': [-0.52165583865101, 1.35190761500146],
    'ri_D_stat': [-0.156736028716283],
    'ri_D_asy': [0.824205968080193],
    'ri_D_sum': [99.0, 49.0, 0.652795918367347, 1.19849591467097, -0.5, 101.0, 42.0, 1.73128571428571, 1.15738330361996, 0.4],
    'ri_D_se': [0.705577548771543],
    'ri_D_ci': [-1.01730781890868, 1.30383576147612],
    'ri_E_stat': [0.352089929316094],
    'ri_E_asy': [0.48767425706821],
    'ri_E_sum': [99.0, 49.0, 0.652795918367347, 1.19849591467097, -0.5, 101.0, 56.0, 1.80767857142857, 1.09367643722933, 0.5],
    'ri_E_se': [0.507324922886278],
    'ri_F_stat': [1.30002992125984],
    'ri_F_asy': [0.000227236380045818],
    'ri_F_sum': [99.0, 40.0, 0.748525, 1.17946650233409, -0.4, 101.0, 42.0, 1.73128571428571, 1.15738330361996, 0.4],
    'ri_F_ci': [0.60888206982939, 1.99117777269029],
    'ri_G_stat': [0.515953965980245],
    'ri_G_asy': [0.980679653544637],
    'ri_G_sum': [99.0, 40.0, 0.748525, 1.17946650233409, -0.4, 101.0, 42.0, 1.73128571428571, 1.15738330361996, 0.4],
    'ri_G_se': [0.658796625242264],
    'ri_H_stat': [0.604784523809524],
    'ri_H_asy': [0.0187608488750257],
    'ri_H_sum': [99.0, 40.0, 0.748525, 1.17946650233409, -0.4, 101.0, 42.0, 1.73128571428571, 1.15738330361996, 0.4],
    'ri_I_stat': [0.264463421895297],
    'ri_I_asy': [0.631602608300814],
    'ri_I_sum': [99.0, 40.0, 0.748525, 1.17946650233409, -0.4, 101.0, 42.0, 1.73128571428571, 1.15738330361996, 0.4],
    'ri_I_se': [0.551571747480876],
    'ri_J_stat': [-1.53896151491693],
    'ri_J_asy': [0.123813691910394],
    'ri_J_sum': [99.0, 30.0, 0.822166666666667, 1.23256676627813, -0.3, 101.0, 29.0, 1.76686206896552, 1.19504213088128, 0.3],
    'ri_K_stat': [0.657038530350413],
    'ri_K_asy': [0.0946325964733534],
    'ri_K_sum': [99.0, 22.0, 1.05886363636364, 1.27439966428472, -0.2, 101.0, 35.0, 1.69565714285714, 1.13800932857898, 0.35],
    'ri_K_se': [0.393094661169454],
    'ri_L_stat': [0.415125888175227],
    'ri_L_asy': [0.385099141581011],
    'ri_L_sum': [99.0, 49.0, 0.652795918367347, 1.19849591467097, -0.5, 101.0, 56.0, 1.80767857142857, 1.09367643722933, 0.5],
    'ri_L_se': [0.477958643227861],
    'ri_L_ci': [-0.5, 1.35],
    'ri_M_stat': [0.588658865886588],
    'ri_M_asy': [1.68469326215128e-16],
    'ri_M_sum': [99.0, 99.0, 0.247959595959596, 1.10305951066347, -0.998, 101.0, 101.0, 2.0410099009901, 1.11086266023347, 0.964],
    'power_A': [0.496043630537682, None, 0.425209107186195],
    'power_J': [0.559472598242835],
    'exact_w010_n': [11.0, 8.0],
    'exact_w010_diffmeans': [0.417004048582996, 75582.0],
    'exact_w010_diffmeans_tau05': [0.99798893916541, 75582.0],
    'exact_w010_ranksum': [0.442142309015374, 75582.0],
    'exact_w010_ksmirnov': [0.418591728189251, 75582.0],
    'exact_w010_ksmirnov_kstest': [0.418591728189251],
    'exact_w010_ar_tau04': [0.821293429652563, 75582.0],
    'exact_w008_triangular': [0.306759906759907, 6435.0],
    'exact_w006_n': [8.0, 4.0],
    'bern_w006_p04_diffmeans': [0.431544459366972],
    'bern_w006_share_diffmeans': [0.457549531235773],
    'bern_w006_p04_ranksum': [0.755962289260567],
    'bern_w006_share_ranksum': [0.760270335871835],
    'bern_w006_p04_ksmirnov': [0.697408881706127],
    'bern_w006_unit_diffmeans': [0.414784768024502],
    'ci_w006_grid_p': [0.00202020202020202, 0.00404040404040404, 0.00404040404040404, 0.00606060606060606, 0.0101010101010101, 0.0161616161616162, 0.0181818181818182, 0.0303030303030303, 0.0545454545454545, 0.103030303030303, 0.18989898989899, 0.290909090909091, 0.418181818181818, 0.61010101010101, 0.850505050505051, 0.892929292929293, 0.656565656565657, 0.450505050505051, 0.290909090909091, 0.185858585858586, 0.111111111111111, 0.0787878787878788, 0.0424242424242424, 0.0282828282828283, 0.0242424242424242, 0.0121212121212121, 0.0101010101010101, 0.00606060606060606, 0.00404040404040404],
    'ci_w006_set': [-1.0, 2.25],
    'rbounds_w006_G1.5_upper_u': [0.758352092822057, 0.767180227518456, 0.777273354642933, 0.785865022110357, 0.790835236669056, 0.79253150801952, 0.790835236669056, 0.785865022110357, 0.777273354642928, 0.76718022751845, 0.758352092822051, 0.755962289260566],
    'rbounds_w006_G1.5_lower_u': [0.758352092822057, 0.767180227518456, 0.777273354642934, 0.785865022110356, 0.790835236669055, 0.792531508019519, 0.790835236669055, 0.785865022110357, 0.777273354642931, 0.767180227518454, 0.758352092822055, 0.755962289260566],
    'rbounds_w006_G2_upper_u': [0.768088237238636, 0.792199663281251, 0.817509572987314, 0.836616971246432, 0.847176917676324, 0.850754639920338, 0.847176917676323, 0.836616971246431, 0.817509572987313, 0.792199663281251, 0.768088237238636, 0.760270335871838],
    'rbounds_w006_G2_lower_u': [0.768088237238635, 0.79219966328125, 0.817509572987315, 0.836616971246435, 0.847176917676328, 0.850754639920343, 0.847176917676328, 0.836616971246435, 0.817509572987314, 0.79219966328125, 0.768088237238635, 0.760270335871838],
    'mc_w030_diffmeans': [0.00395],
    'mc_w030_ranksum': [0.0051],
    'mc_w030_bernoulli05_diffmeans': [0.0041],
    'sens_default_wlist': [0.132, 0.187, 0.272, 0.321, 0.366, 0.419, 0.486, 0.543, 0.624, 0.686],
    'sens_default_wlist_left': [-0.132, -0.187, -0.272, -0.321, -0.366, -0.419, -0.486, -0.543, -0.624, -0.686],
    'sens_default_tlist': [-0.61, -0.43, -0.25, -0.07, 0.11, 0.29, 0.47, 0.65, 0.83, 1.01, 1.19],
    'sens_p1_results': [0.308137919496853, 0.711429304064263, 0.780043244432615, 0.353167694822771, 0.11464319245782, 0.387130814395937, 0.925066799908209, 0.498580005074978, 0.147755613047361, 0.0265354912001162, 0.0919537923472247, 0.434810011223107, 0.90194409244163, 0.304232658819775, 0.0534098848741872],
    'sens_p1_ci': [-0.5, 1.0],
    'sens_p2_means_results': [0.557766046448843, 0.768926746381585, 0.240506795318245, 0.113952915905553, 0.650085169232428, 0.500702482452011, 0.00245001422337585, 0.0887764392258113, 0.708160568665785],
    'win_default_left': [-0.132, -0.187, -0.272, -0.321, -0.366, -0.419, -0.486, -0.543, -0.624, -0.686],
    'win_default_right': [0.132, 0.187, 0.272, 0.321, 0.366, 0.419, 0.486, 0.543, 0.624, 0.686],
    'win_default_n': [16.0, 21.0, 26.0, 32.0, 37.0, 42.0, 47.0, 52.0, 60.0, 65.0, 10.0, 22.0, 28.0, 33.0, 38.0, 45.0, 53.0, 59.0, 64.0, 71.0],
    'win_default_p': [0.0450014033253065, 0.126372914964155, 0.139196072416442, 0.17864447683463, 0.220230582811035, 0.411021959399533, 0.179971933440853, 0.134823715357588, 0.127961489686356, 0.0354124842835327],
    'win_default_binom': [0.326939582824707, 1.0, 0.89192315110475, 1.0, 1.0, 0.830373009685753, 0.617299413589252, 0.569220227803575, 0.787746582402333, 0.668265852313181],
    'win_default_rec': [None, None],
    'win_default_summary': [99.0, 1.0, 5.0, 10.0, 20.0, 101.0, 2.0, 6.0, 11.0, 21.0],
    'win_asym_left': [-0.089, -0.124, -0.187, -0.272, -0.308, -0.342, -0.401, -0.467, -0.539, -0.579],
    'win_asym_right': [0.132, 0.169, 0.182, 0.22, 0.311, 0.353, 0.387, 0.449, 0.479, 0.53],
    'win_asym_n': [10.0, 16.0, 21.0, 26.0, 31.0, 36.0, 41.0, 46.0, 51.0, 56.0, 10.0, 15.0, 21.0, 26.0, 31.0, 36.0, 41.0, 46.0, 52.0, 57.0],
    'win_asym_p': [0.0812121729371682, 0.0233088410075343, 0.116644450042975, 0.141067908379792, 0.221035590023658, 0.302132858965624, 0.375666483288592, 0.286380512409344, 0.255326041428599, 0.218808619177097],
    'win_asym_binom': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'win_asym_rec': [None, None],
    'win_asym_summary': [99.0, 1.0, 5.0, 10.0, 20.0, 101.0, 2.0, 6.0, 11.0, 21.0],
    'win_step_left': [-0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4, -0.45, -0.5, -0.55],
    'win_step_right': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
    'win_step_n': [11.0, 17.0, 22.0, 25.0, 30.0, 36.0, 40.0, 44.0, 49.0, 53.0, 8.0, 12.0, 23.0, 27.0, 29.0, 35.0, 42.0, 46.0, 56.0, 59.0],
    'win_step_p': [0.146877159344236, 0.0640488430395292, 0.0924549074483777, 0.11905137298432, 0.177412584830854, 0.267086256482095, 0.37893833331009, 0.290592038590204, 0.192866162355791, 0.122631807845993],
    'win_step_binom': [0.647605895996094, 0.458258319646121, 1.0, 0.889883965276538, 1.0, 1.0, 0.912156607552603, 0.91612887701129, 0.55839429116559, 0.636799771614234],
    'win_step_rec': [None, None],
    'win_step_summary': [99.0, 1.0, 5.0, 10.0, 20.0, 101.0, 2.0, 6.0, 11.0, 21.0],
    'win_pair_left': [-0.06, -0.165, -0.24, -0.321, -0.37, -0.44, -0.513, -0.61, -0.686, -0.734],
    'win_pair_right': [0.06, 0.165, 0.24, 0.321, 0.37, 0.44, 0.513, 0.61, 0.686, 0.734],
    'win_pair_n': [8.0, 19.0, 25.0, 32.0, 38.0, 44.0, 50.0, 59.0, 65.0, 72.0, 4.0, 14.0, 27.0, 33.0, 39.0, 45.0, 56.0, 62.0, 71.0, 77.0],
    'win_pair_p': [0.672827846451034, 0.0188495350455228, 0.11905137298432, 0.17864447683463, 0.333366150369238, 0.380972891092362, 0.237355913883662, 0.174818662544681, 0.0354124842835327, 0.00820738861698845],
    'win_pair_binom': [0.3876953125, 0.486850241664797, 0.889883965276538, 1.0, 1.0, 1.0, 0.627430977078634, 0.855821599211081, 0.668265852313181, 0.743269021767057],
    'win_pair_rec': [-0.06, 0.06],
    'win_pair_summary': [99.0, 1.0, 5.0, 10.0, 20.0, 101.0, 2.0, 6.0, 11.0, 21.0],
    'win_small_left': [-0.07, -0.132, -0.172, -0.211, -0.272, -0.311],
    'win_small_right': [0.07, 0.132, 0.172, 0.211, 0.272, 0.311],
    'win_small_n': [8.0, 16.0, 20.0, 23.0, 26.0, 31.0, 7.0, 10.0, 16.0, 25.0, 28.0, 31.0],
    'win_small_p': [0.257255265269764, 0.0450014033253065, 0.00865700944167474, 0.106082722218026, 0.139196072416442, 0.221035590023658],
    'win_small_binom': [1.0, 0.326939582824707, 0.617719317029696, 0.885433497286515, 0.89192315110475, 1.0],
    'win_small_rec': [-0.07, 0.07],
    'win_small_summary': [99.0, 1.0, 5.0, 10.0, 20.0, 101.0, 2.0, 6.0, 11.0, 21.0],
    'win_pairasym_left': [-0.12, -0.165, -0.24, -0.308, -0.354, -0.422, -0.491, -0.569, -0.637, -0.7],
    'win_pairasym_right': [0.06, 0.132, 0.171, 0.183, 0.253, 0.336, 0.385, 0.449, 0.482, 0.538],
    'win_pairasym_n': [13.0, 19.0, 25.0, 31.0, 37.0, 43.0, 49.0, 55.0, 61.0, 67.0, 4.0, 10.0, 16.0, 22.0, 28.0, 34.0, 40.0, 46.0, 53.0, 59.0],
    'win_pairasym_p': [0.601715630237879, 0.0219701114688148, 0.0043636309277894, 0.245345108318882, 0.186721578360907, 0.203171276133234, 0.443414444010066, 0.400690780673971, 0.303034809813951, 0.108891672682367],
    'win_pairasym_binom': [0.049041748046875, 0.136045947670937, 0.211023597608801, 0.271679166065503, 0.321084135684997, 0.362031797739663, 0.396570158145516, 0.426160015466994, 0.512254872229845, 0.533052750267557],
    'win_pairasym_rec': [-0.12, 0.06],
    'win_pairasym_summary': [99.0, 1.0, 5.0, 10.0, 20.0, 101.0, 2.0, 6.0, 11.0, 21.0],
    'win_cut_left': [-0.112, -0.205, -0.249, -0.289, -0.373, -0.418, -0.481, -0.527, -0.59, -0.644],
    'win_cut_right': [0.112, 0.205, 0.249, 0.289, 0.373, 0.418, 0.481, 0.527, 0.59, 0.644],
    'win_cut_n': [10.0, 20.0, 25.0, 30.0, 35.0, 40.0, 46.0, 51.0, 56.0, 61.0, 17.0, 22.0, 27.0, 34.0, 42.0, 48.0, 53.0, 58.0, 64.0, 70.0],
    'win_cut_p': [0.649991259295987, 0.811474082169704, 0.73378773812589, 0.600270798536993, 0.452951426424338, 0.337271322010018, 0.416063394900529, 0.21764569803737, 0.127887041469182, 0.0146199271367024],
    'win_cut_binom': [0.247788563370705, 0.877614328752315, 0.889883965276538, 0.707980754134766, 0.494382273695501, 0.455741279437913, 0.546713483598813, 0.565700113973634, 0.522990983889995, 0.484720218128531],
    'win_cut_rec': [-0.427, 0.627],
    'win_cut_summary': [107.0, 2.0, 5.0, 11.0, 22.0, 93.0, 1.0, 6.0, 10.0, 19.0],
    'winstat_ks_p': [0.178177343812019, 0.0114752725815615, 0.0180438842002992, 0.0229165925128879, 0.155102587065664, 0.0809354653122441],
    'winstat_rank_p': [0.186449181219981, 0.0513727695950745, 0.0594865861559745, 0.0741498977382223, 0.165338205612794, 0.189815683461764],
    'winstat_hot_p': [0.44416521752728, 0.127583222059412, 0.188051497423456, 0.215691951244974, 0.20331511038492, 0.501544861478626],
    'winstat_tri_p': [0.44028518656029, 0.230373177978178, 0.11644598970815, 0.101575316617418, 0.0992350698635552, 0.109896939219927],
    'winstat_p1_p': [0.328959271533181, 0.586276095150051, 0.360881254520626, 0.21689319849399, 0.113965505291484, 0.0970895939997085],
    'winstat_p1means_p': [0.127591125291179, 0.0669011044214916, 0.0993942334769213, 0.12421654769421, 0.17327725854226, 0.266420120900251],
    'win_exact_right': [0.051, 0.089, 0.132],
    'win_exact_pmin': [0.717171717171717, 0.189725307372366, 0.0791884384292515],
    'mass_left': [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
    'mass_right': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    'mass_n': [13.0, 18.0, 28.0, 39.0, 49.0, 58.0, 70.0, 83.0, 94.0, 107.0, 12.0, 24.0, 30.0, 37.0, 44.0, 54.0, 70.0, 84.0, 94.0, 107.0],
    'mass_p': [0.0837166610456918, 0.227282699033875, 0.449566404558705, 0.393539029912034, 0.145041917959208, 0.0967051391102996, 0.0524868339801698, 0.0189086472946965, 0.018228606559134, 0.000452455660228882],
    'mass_rec': [None, None],
    'big_ks': [0.54695117803125, 1.0425765490925e-27, 209.0, 211.0],
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


def _rnd(x, k):
    return math.floor(x * 10 ** k + 0.5) / 10 ** k


def dataset(n=200):
    u, v = _minstd(20261010, n), _minstd(1234567891, n)
    z1, z2, z3 = _normals(987654321, n), _normals(1122334455, n), _normals(2024681357, n)
    r = [_rnd(2 * a - 1, 3) for a in u]
    t = [1.0 if (ri >= 0 and b < 0.85) or (ri < 0 and b < 0.15) else 0.0 for ri, b in zip(r, v)]
    x1 = [_rnd(0.5 * ri + a, 3) for ri, a in zip(r, z1)]
    x2 = [_rnd(b + (0.6 if ri > 0.5 else -0.6 if ri < -0.5 else 0.0), 3) for ri, b in zip(r, z2)]
    y = [_rnd(1 + ri - 0.4 * ri * ri + 0.7 * ti + c, 3) for ri, ti, c in zip(r, t, z3)]
    return pd.DataFrame({"r": r, "y": y, "t": t, "x1": x1, "x2": x2})


DATA = dataset()
COVARIATES = ["x1", "x2"]


def close(actual, expected, rtol=1e-8, atol=1e-11):
    actual = [None if a is None or (isinstance(a, float) and math.isnan(a)) else float(a) for a in actual]
    assert len(actual) == len(expected), (actual, expected)
    for a, e in zip(actual, expected):
        if e is None:
            assert a is None, (actual, expected)
        else:
            assert a is not None and abs(a - e) <= atol + rtol * abs(e), (actual, expected)


def test_data_are_regenerated_bit_for_bit():
    close([DATA.r.sum(), DATA.y.sum(), DATA.t.sum(), DATA.x1.sum(), DATA.x2.sum()], REF["check_sums"], rtol=1e-12)


RI_CASES = {
    "A": dict(wl=-0.3, wr=0.3, statistic="all"),
    "B": dict(wl=-0.3, wr=0.25, kernel="triangular"),
    "C": dict(wl=-0.5, wr=0.5, p=1, ci=0.05),
    "D": dict(wl=-0.5, wr=0.4, p=2, vce="HC1", nulltau=0.3, ci=0.1),
    "E": dict(wl=-0.5, wr=0.5, p=1, vce="HC2", kernel="epan"),
    "F": dict(wl=-0.4, wr=0.4, fuzzy="t", fuzzy_statistic="tsls", ci=0.05),
    "G": dict(wl=-0.4, wr=0.4, fuzzy="t", fuzzy_statistic="tsls", p=1, nulltau=0.5),
    "H": dict(wl=-0.4, wr=0.4, fuzzy="t", nulltau=0.5),
    "I": dict(wl=-0.4, wr=0.4, fuzzy="t", p=1, nulltau=0.2),
    "J": dict(wl=-0.3, wr=0.3, statistic="ranksum", nulltau=0.4, d=0.7),
    "K": dict(wl=-0.2, wr=0.35, p=1, evall=-0.1, evalr=0.1),
    "L": dict(wl=-0.5, wr=0.5, p=1, ci=0.05, ci_grid=[round(-1 + 0.05 * i, 10) for i in range(61)]),
    "M": dict(statistic="ksmirnov"),
}


@pytest.mark.parametrize("case", sorted(RI_CASES))
def test_rdrandinf_deterministic_output_matches_r(case):
    res = oe.rdrandinf(data=DATA, y="y", running="r", reps=200, **RI_CASES[case])
    tests, sample = res["tests"], res["sample"]
    close(tests["statistic"], REF[f"ri_{case}_stat"])
    close(tests["asymptotic_p_value"], REF[f"ri_{case}_asy"], rtol=1e-7)
    close([sample["n"][0], sample["n_window"][0], sample["mean"][0], sample["std_dev"][0],
           sample["window_limit"][0], sample["n"][1], sample["n_window"][1], sample["mean"][1],
           sample["std_dev"][1], sample["window_limit"][1]], REF[f"ri_{case}_sum"])
    if f"ri_{case}_se" in REF:
        close([tests["std_error"][0]], REF[f"ri_{case}_se"])
    if f"ri_{case}_ci" in REF:
        close([res["ci"]["ci_low"][0], res["ci"]["ci_high"][0]], REF[f"ri_{case}_ci"])


def test_asymptotic_power_matches_package_formula():
    res = oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.3, wr=0.3, statistic="all", reps=100)
    close(res["tests"]["power"], REF["power_A"])
    res = oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.3, wr=0.3, statistic="ranksum",
                       nulltau=0.4, d=0.7, reps=100)
    close(res["tests"]["power"], REF["power_J"])


def test_exact_enumeration_matches_r_brute_force():
    res = oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.1, wr=0.1, statistic="all", method="exact")
    assert res.attrs["method"] == "exact" and res.attrs["draws"] == REF["exact_w010_diffmeans"][1]
    assert [res.attrs["n_left"], res.attrs["n_right"]] == REF["exact_w010_n"]
    p = list(res["tests"]["p_value"])
    close(p, [REF["exact_w010_diffmeans"][0], REF["exact_w010_ksmirnov"][0], REF["exact_w010_ranksum"][0]], rtol=1e-12)
    # R's ks.test exact (tie-aware Smirnov) law equals the enumerated KS randomization law.
    close([res["tests"]["asymptotic_p_value"][1], p[1]], REF["exact_w010_ksmirnov_kstest"] * 2, rtol=1e-10)
    res = oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.1, wr=0.1, nulltau=0.5, method="exact")
    close([res["tests"]["p_value"][0]], REF["exact_w010_diffmeans_tau05"][:1], rtol=1e-12)
    res = oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.1, wr=0.1, fuzzy="t", nulltau=0.4,
                       method="exact")
    close([res["tests"]["p_value"][0]], REF["exact_w010_ar_tau04"][:1], rtol=1e-12)
    res = oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.08, wr=0.08, kernel="triangular",
                       method="exact")
    close([res["tests"]["p_value"][0]], REF["exact_w008_triangular"][:1], rtol=1e-12)


def test_exact_bernoulli_matches_r_brute_force():
    def p_values(bernoulli, statistic, data=DATA):
        res = oe.rdrandinf(data=data, y="y", running="r", wl=-0.06, wr=0.06, statistic=statistic,
                           bernoulli=bernoulli, method="exact")
        assert res.attrs["randomization"] == "Bernoulli" and res.attrs["draws"] == 2 ** 12
        return list(res["tests"]["p_value"])
    close(p_values(0.4, "all"), [REF["bern_w006_p04_diffmeans"][0], REF["bern_w006_p04_ksmirnov"][0],
                                 REF["bern_w006_p04_ranksum"][0]], rtol=1e-10)
    close(p_values(True, "diffmeans"), REF["bern_w006_share_diffmeans"], rtol=1e-10)
    close(p_values(True, "ranksum"), REF["bern_w006_share_ranksum"], rtol=1e-10)
    inside = (DATA.r.round(8) >= -0.06) & (DATA.r.round(8) <= 0.06)
    data = DATA.assign(prob=0.5)
    data.loc[inside, "prob"] = [0.2 + 0.5 * i / 11 for i in range(12)]
    close(p_values("prob", "diffmeans", data), REF["bern_w006_unit_diffmeans"], rtol=1e-10)


def test_confidence_set_by_exact_inversion_matches_r():
    grid = [-3 + 0.25 * i for i in range(29)]
    res = oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.06, wr=0.06, ci=0.05, ci_grid=grid,
                       method="exact")
    close(res["ci_grid"]["p_value"], REF["ci_w006_grid_p"], rtol=1e-12)
    close([res["ci"]["ci_low"][0], res["ci"]["ci_high"][0]], REF["ci_w006_set"])


def test_rosenbaum_bounds_match_exact_configurations():
    res = oe.rdrbounds(data=DATA, y="y", running="r", wlist=[0.06], expgamma=[1.5, 2.0], method="exact")
    close([res["pvalues"]["p_value"][0]], REF["bern_w006_share_ranksum"], rtol=1e-10)
    bounds = res["bounds"]
    for row, gamma in enumerate(("1.5", "2")):
        close([bounds["upper_bound"][row]], [max(REF[f"rbounds_w006_G{gamma}_upper_u"])], rtol=1e-10)
        close([bounds["lower_bound"][row]], [min(REF[f"rbounds_w006_G{gamma}_lower_u"])], rtol=1e-10)


@pytest.mark.parametrize("statistic, bernoulli", [("diffmeans", None), ("ranksum", None), ("diffmeans", 0.5)])
def test_monte_carlo_agrees_with_r_within_simulation_error(statistic, bernoulli):
    res = oe.rdrandinf(data=DATA, y="y", running="r", wl=-0.3, wr=0.3, statistic=statistic,
                       bernoulli=bernoulli, reps=20000, seed=7)
    ours = res["tests"]["p_value"][0]
    key = f"mc_w030_{'bernoulli05_' if bernoulli else ''}{statistic}"
    theirs = REF[key][0]
    pooled = (ours + theirs) / 2
    assert abs(ours - theirs) <= 4.5 * math.sqrt(max(pooled * (1 - pooled), 1e-4) * 2 / 20000)


def test_rdsensitivity_matches_r():
    res = oe.rdsensitivity(data=DATA, y="y", running="r", reps=50)
    close(res["windows"]["w_right"], REF["sens_default_wlist"])
    close(res["windows"]["w_left"], REF["sens_default_wlist_left"])
    close(res.attrs["tlist"], REF["sens_default_tlist"])
    res = oe.rdsensitivity(data=DATA, y="y", running="r", wlist=[0.2, 0.3, 0.4],
                           tlist=[-0.5, 0.0, 0.5, 1.0, 1.5], p=1, ci=[-0.3, 0.3])
    close(res["pvalues"]["p_value"], REF["sens_p1_results"])
    close([res["ci"]["ci_low"][0], res["ci"]["ci_high"][0]], REF["sens_p1_ci"])
    res = oe.rdsensitivity(data=DATA, y="y", running="r", wlist=[0.2, 0.3, 0.4],
                           wlist_left=[-0.25, -0.3, -0.45], tlist=[0, 0.5, 1], p=2, evalat="means",
                           vce="HC2")
    close(res["pvalues"]["p_value"], REF["sens_p2_means_results"])


WIN_CASES = {
    "default": {}, "asym": dict(wasymmetric=True), "step": dict(wmin=0.1, wstep=0.05),
    "pair": dict(wmin=[-0.12, 0.06], wobs=6), "small": dict(obsmin=6, wobs=3, nwindows=6),
    "pairasym": dict(wmin=[-0.12, 0.06], wobs=6, wasymmetric=True), "cut": dict(cutoff=0.1),
}


@pytest.mark.parametrize("case", sorted(WIN_CASES))
def test_rdwinselect_windows_and_tests_match_r(case):
    options = WIN_CASES[case]
    res = oe.rdwinselect(data=DATA, running="r", covariates=COVARIATES, approx=True, **options)
    cut = options.get("cutoff", 0.0)
    windows = res["windows"]
    close([v - cut for v in windows["w_left"]], REF[f"win_{case}_left"], atol=1e-12)
    close([v - cut for v in windows["w_right"]], REF[f"win_{case}_right"], atol=1e-12)
    close([*windows["n_left"], *windows["n_right"]], REF[f"win_{case}_n"])
    close(windows["p_value"], REF[f"win_{case}_p"], rtol=1e-7)
    close(windows["binomial_p_value"], REF[f"win_{case}_binom"], rtol=1e-9)
    rec = [res.attrs["recommended_left"], res.attrs["recommended_right"]]
    expected = REF[f"win_{case}_rec"]
    close(rec, expected, atol=1e-12)
    s = res["sample"]
    close([*[s[col][0] for col in ("n", "n_1pct", "n_5pct", "n_10pct", "n_20pct")],
           *[s[col][1] for col in ("n", "n_1pct", "n_5pct", "n_10pct", "n_20pct")]], REF[f"win_{case}_summary"])


@pytest.mark.parametrize("case, options", [
    ("ks", dict(statistic="ksmirnov")), ("rank", dict(statistic="ranksum")),
    ("hot", dict(statistic="hotelling")), ("tri", dict(kernel="triangular")), ("p1", dict(p=1)),
    ("p1means", dict(p=1, evalat="means", vce="HC1"))])
def test_rdwinselect_balance_statistics_match_r(case, options):
    res = oe.rdwinselect(data=DATA, running="r", covariates=COVARIATES, approx=True, wmin=0.1,
                         wstep=0.05, nwindows=6, **options)
    close(res["windows"]["p_value"], REF[f"winstat_{case}_p"], rtol=1e-7)


def test_rdwinselect_exact_randomization_matches_r():
    res = oe.rdwinselect(data=DATA, running="r", covariates=COVARIATES, obsmin=4, wobs=2, nwindows=3,
                         method="exact")
    close(res["windows"]["w_right"], REF["win_exact_right"])
    close(res["windows"]["p_value"], REF["win_exact_pmin"], rtol=1e-12)
    assert list(res["windows"]["method"]) == ["exact"] * 3


def test_rdwinselect_mass_points_match_r():
    rd = [math.floor(u * 31) - 15 for u in _minstd(55555, 300)]
    xd1 = [_rnd(z, 3) for z in _normals(66666, 300)]
    xd2 = [_rnd(z + 0.05 * r, 3) for z, r in zip(_normals(77777, 300), rd)]
    data = pd.DataFrame({"r": rd, "x1": xd1, "x2": xd2})
    res = oe.rdwinselect(data=data, running="r", covariates=["x1", "x2"], wmasspoints=True, approx=True)
    windows = res["windows"]
    close(windows["w_left"], REF["mass_left"])
    close(windows["w_right"], REF["mass_right"])
    close([*windows["n_left"], *windows["n_right"]], REF["mass_n"])
    close(windows["p_value"], REF["mass_p"], rtol=1e-7)
    assert res.attrs["recommended_left"] is None and REF["mass_rec"] == [None, None]


def test_kolmogorov_limit_branch_matches_r():
    big = dataset(420)
    res = oe.rdrandinf(data=big, y="y", running="r", statistic="ksmirnov", reps=100)
    statistic, asymptotic, n_left, n_right = REF["big_ks"]
    assert [res.attrs["n_left"], res.attrs["n_right"]] == [n_left, n_right]
    close([res["tests"]["statistic"][0]], [statistic], rtol=1e-12)
    # R's Kolmogorov series stops at tol = 1e-6; at this tail both sums agree to about 1e-9 relative.
    close([res["tests"]["asymptotic_p_value"][0]], [asymptotic], rtol=1e-6)
