# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow


def tar_imp_hists(all_scores, all_labels):
    # Function to compute target and impostor histogram
    
    tar_scores = []
    imp_scores = []

    ###########################################################
    # Here is your code
    all_scores = np.asarray(all_scores)
    all_labels = np.asarray(all_labels)

    tar_scores = all_scores[all_labels == 1]  # target trials: label 1
    imp_scores = all_scores[all_labels == 0]  # impostor trials: label 0
    ###########################################################
    
    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)
    
    return tar_scores, imp_scores

def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf):
    # Function to compute log-likelihood ratio
    
    tar_scores_mean = np.mean(tar_scores)
    tar_scores_std  = np.std(tar_scores)
    imp_scores_mean = np.mean(imp_scores)
    imp_scores_std  = np.std(imp_scores)
    
    all_scores_sort   = np.zeros(len(all_scores))
    ground_truth_sort = np.zeros(len(all_scores), dtype='bool')
    
    ###########################################################
    # Here is your code
    sort_idx          = np.argsort(all_scores)
    all_scores_sort   = np.asarray(all_scores)[sort_idx]
    ground_truth_sort = np.asarray(all_labels)[sort_idx] == 1
    ###########################################################
    
    tar_gauss_pdf = np.zeros(len(all_scores))
    imp_gauss_pdf = np.zeros(len(all_scores))
    LLR           = np.zeros(len(all_scores))
    
    ###########################################################
    # Here is your code
    tar_gauss_pdf = gauss_pdf(all_scores_sort, tar_scores_mean, tar_scores_std)  # W(s|Htar)
    imp_gauss_pdf = gauss_pdf(all_scores_sort, imp_scores_mean, imp_scores_std)  # W(s|Himp)
    LLR           = np.log(tar_gauss_pdf + 1e-300) - np.log(imp_gauss_pdf + 1e-300)
    ###########################################################
    
    return ground_truth_sort, all_scores_sort, tar_gauss_pdf, imp_gauss_pdf, LLR


def error_rates(ground_truth_sort, LLR, tar_scores, imp_scores):
    # Function to compute FNR and FPR for every threshold LLR[idx] (decision: LLR > LLR[idx]), vectorized version of the loop in map_test
    
    tar_llr = np.sort(LLR[ ground_truth_sort])
    imp_llr = np.sort(LLR[~ground_truth_sort])
    
    fnr_thr = np.searchsorted(tar_llr, LLR, side='right')/len(tar_scores)                    # P(Dimp|Htar): targets with LLR <= thr
    fpr_thr = (len(imp_llr) - np.searchsorted(imp_llr, LLR, side='right'))/len(imp_scores)  # P(Dtar|Himp): impostors with LLR > thr
    
    return fnr_thr, fpr_thr

def average_cost(fnr, fpr, P_Htar, C00, C10, C01, C11):
    # Function to compute average cost (Bayes' risk) for given error rates
    
    return C00*(1 - fnr)*P_Htar + C10*fnr*P_Htar + C01*fpr*(1 - P_Htar) + C11*(1 - fpr)*(1 - P_Htar)

def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    # Function to perform maximum a posteriori test
    
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    P_err   = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        err = (solution != ground_truth_sort)                          # error vector
        
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        
        P_err[idx]   = fnr_thr[idx]*P_Htar + fpr_thr[idx]*(1 - P_Htar) # prob. of error
    
    # Plot error's prob.
    plot(LLR, P_err, color='blue')
    xlabel('$LLR$'); ylabel('$P_e$'); title('Probability of error'); grid(); show()
        
    P_err_idx = np.argmin(P_err) # argmin of error's prob.
    P_err_min = fnr_thr[P_err_idx]*P_Htar + fpr_thr[P_err_idx]*(1 - P_Htar)
    
    return LLR[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min

def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    # Function to perform Neyman-Pearson test
    
    thr   = 0.0
    fpr   = 0.0
    
    ###########################################################
    # Here is your code
    fnr_thr, fpr_thr = error_rates(ground_truth_sort, LLR, tar_scores, imp_scores)

    # FNR grows and FPR falls with the threshold, so take the largest threshold that keeps FNR <= given value
    valid   = np.where(fnr_thr <= fnr)[0]
    thr_idx = valid[np.argmax(LLR[valid])]

    thr = LLR[thr_idx]
    fpr = fpr_thr[thr_idx]
    ###########################################################
    
    return thr, fpr

def bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11):
    # Function to perform Bayes' test
    
    thr   = 0.0
    fnr   = 0.0
    fpr   = 0.0
    AC    = 0.0
    
    ###########################################################
    # Here is your code
    fnr_thr, fpr_thr = error_rates(ground_truth_sort, LLR, tar_scores, imp_scores)

    AC_thr  = average_cost(fnr_thr, fpr_thr, P_Htar, C00, C10, C01, C11)
    thr_idx = np.argmin(AC_thr)

    thr = LLR[thr_idx]
    fnr = fnr_thr[thr_idx]
    fpr = fpr_thr[thr_idx]
    AC  = AC_thr[thr_idx]
    ###########################################################
    
    return thr, fnr, fpr, AC

def minmax_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar_thr, C00, C10, C01, C11):
    # Function to perform minimax test
    
    thr    = 0.0
    fnr    = 0.0
    fpr    = 0.0
    AC     = 0.0
    P_Htar = 0.0
    
    ###########################################################
    # Here is your code
    fnr_thr, fpr_thr = error_rates(ground_truth_sort, LLR, tar_scores, imp_scores)

    # Bayes risk (minimum of average cost over thresholds) for every prior P(Htar)
    thr_idx_P = np.zeros(len(P_Htar_thr), dtype='int')
    AC_P      = np.zeros(len(P_Htar_thr))

    for idx, P in enumerate(P_Htar_thr):
        AC_thr         = average_cost(fnr_thr, fpr_thr, P, C00, C10, C01, C11)
        thr_idx_P[idx] = np.argmin(AC_thr)
        AC_P[idx]      = AC_thr[thr_idx_P[idx]]

    # Least favourable prior maximizes the Bayes risk
    P_idx   = np.argmax(AC_P)
    thr_idx = thr_idx_P[P_idx]

    thr    = LLR[thr_idx]
    fnr    = fnr_thr[thr_idx]
    fpr    = fpr_thr[thr_idx]
    AC     = AC_P[P_idx]
    P_Htar = P_Htar_thr[P_idx]
    ###########################################################
    
    return thr, fnr, fpr, AC, P_Htar