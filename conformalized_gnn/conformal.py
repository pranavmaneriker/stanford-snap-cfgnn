import numpy as np
import torch

def tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_scores = 1-cal_smx[np.arange(n),cal_labels]
    q_level = np.ceil((n+1)*(1-alpha))/n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    prediction_sets = val_smx >= (1-qhat)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff

def aps_helper(probs):
    probs_pi = np.argsort(-probs, axis=1)
    sorted_probs =  np.take_along_axis(probs, probs_pi, axis=1)
    PI = np.zeros((sorted_probs.shape[0], sorted_probs.shape[1] + 1))
    PI[:, 1:] = np.cumsum(sorted_probs, axis=1)
    # fix ranks
    ranks = np.argsort(probs_pi, axis=1)
    #u_vec = np.random.uniform(low=0.0, high=1.0, size=probs.shape)
    u_vec = np.random.rand(*probs.shape)
    #cls_scores = np.take(PI, ranks + 1)# + (1 - u_vec) * probs
    cls_scores = np.take_along_axis(PI, ranks, axis=1) + u_vec * probs
    #cls_scores = np.take_along_axis(PI, ranks+1, 1)
    #cls_scores = PI.gather(1, ranks) + (1 - u_vec) * probs
    cls_scores = np.clip(cls_scores, a_min=0, a_max=1) #np.min(cls_scores, np.ones_like(cls_scores))
    # TODO: clip vs min
    return cls_scores

def new_aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_scores = aps_helper(cal_smx)
    cal_label_scores = np.take_along_axis(cal_scores, cal_labels.reshape(-1, 1), axis=1)
    qhat = np.quantile(cal_label_scores.flatten(), np.ceil((n + 1) * (1 - alpha)) / n, method='higher')
    val_scores = aps_helper(val_smx)
    prediction_sets = (val_scores <= qhat)
    cov = np.take_along_axis(prediction_sets, val_labels.reshape(-1, 1), axis=1).sum()/len(prediction_sets)
    eff = prediction_sets.sum(axis=1).sum()/(len(prediction_sets))
    return prediction_sets, cov, eff

def old_aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"
    )
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff
        
def raps_fixed(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    import pdb; pdb.set_trace()
    lam_reg = 0.01
    k_reg = min(5, cal_smx.shape[1])
    disallow_zero_sets = False
    rand = True
    # calibration, only needs the label scores
    cal_pi = np.argsort(-cal_smx, axis=1) # argsort probs in descending order
    cal_sorted_pi = np.take_along_axis(cal_smx, cal_pi, axis=1) # sort probs in descending order of probs 
    cal_ranks = np.argsort(cal_pi, axis=1) # ranks for descending order
    label_ranks = np.take_along_axis(cal_ranks, cal_labels.reshape(-1, 1), axis=1) # ranks for each label
    cal_csum = np.cumsum(cal_sorted_pi, axis=1) # cumulative sum of probs acc to descending rank
    label_csum = np.take_along_axis(cal_csum, label_ranks.reshape(-1, 1), axis=1) # cumulative sum of probs acc to descending rank for each label
    reg_vec = np.maximum(0, lam_reg * (label_ranks - k_reg)) # regularization vector for label scores
    cal_scores = label_csum  + reg_vec # label scores for calibration set
    if rand:
        cal_scores -= np.random.rand(n, 1) * np.take_along_axis(cal_smx, cal_labels.reshape(-1, 1), axis=1)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher') # quantile for calibration set
    # validation
    n_val = val_smx.shape[0]
    k = val_smx.shape[1]
    val_pi = np.argsort(-val_smx, axis=1)
    val_sorted_pi = np.take_along_axis(val_smx, val_pi, axis=1)
    val_ranks = np.argsort(val_pi, axis=1)
    val_csum = np.cumsum(val_sorted_pi, axis=1)
    range_array = np.tile(np.arange(k).repeat(1, -1), (n_val, 1))
    val_reg = np.maximum(0, lam_reg * (range_array - k_reg))
    val_sets = (val_csum + val_reg) <= qhat
    L_vec = np.sum(val_sets, axis=1) # + 1 # L as defined in the paper, +1 not added since 0 indexed
    if rand:
        u = np.random.rand(n_val, 1)
        num = np.take_along_axis(val_csum, L_vec.reshape(-1, 1), axis=1) + lam_reg * np.maximum(0, L_vec.reshape(-1, 1) - k_reg) - qhat
        deno = np.take_along_axis(val_sorted_pi, L_vec.reshape(-1, 1), axis=1) + lam_reg * (L_vec.reshape(-1, 1) > k_reg)
        indicator_vector = (num/deno <= u)
        L_vec = L_vec - indicator_vector.reshape(-1)
    prediction_sets = np.zeros(val_smx.shape, dtype=np.bool_)
    if disallow_zero_sets: prediction_sets[:, 0] = True
    for row, ind in enumerate(L_vec):
        indices_to_set = val_pi[row, :(ind + 1)]
        prediction_sets[row, indices_to_set] = True
    cov = np.take_along_axis(prediction_sets, val_labels.reshape(-1, 1), axis=1).sum()/len(prediction_sets)
    eff = prediction_sets.sum(axis=1).sum()/(len(prediction_sets))
    return prediction_sets, cov, eff


def raps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    lam_reg = 0.01
    k_reg = min(5, cal_smx.shape[1])
    disallow_zero_sets = False 
    rand = True
    reg_vec = np.array(k_reg*[0,] + (cal_smx.shape[1]-k_reg)*[lam_reg,])[None,:]

    cal_pi = cal_smx.argsort(1)[:,::-1]; 
    cal_srt = np.take_along_axis(cal_smx,cal_pi,axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:,None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]
    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    # Deploy
    n_val = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff
    
def cqr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha):
    cal_scores = np.maximum(cal_labels-cal_upper, cal_lower-cal_labels)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    prediction_sets = [val_lower - qhat, val_upper + qhat]
    cov = ((val_labels >= prediction_sets[0]) & (val_labels <= prediction_sets[1])).mean()
    eff = np.mean(val_upper + qhat - (val_lower - qhat))
    return prediction_sets, cov, eff

def qr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha):
    prediction_sets = [val_lower, val_upper]
    cov = ((val_labels >= prediction_sets[0]) & (val_labels <= prediction_sets[1])).mean()
    eff = np.mean(val_upper - val_lower)
    return prediction_sets, cov, eff

def threshold(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    
    prediction_sets = np.take_along_axis(val_srt <= 1-alpha, val_pi.argsort(axis=1), axis=1)
    prediction_sets[np.arange(prediction_sets.shape[0]), val_pi[:, 0]] = True

    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff


def run_conformal_classification(pred, data, n, alpha, score = 'aps', 
                                 calib_eval = False, validation_set = False, 
                                 use_additional_calib = False, return_prediction_sets = False, calib_fraction = 0.5): 
    if calib_eval:
        n_base = int(n * (1-calib_fraction))
    else:
        n_base = n
        
    logits = torch.nn.Softmax(dim = 1)(pred).detach().cpu().numpy()
    
    if validation_set:
        smx = logits[data.valid_mask]
        labels = data.y[data.valid_mask].detach().cpu().numpy()
        n_base = int(len(np.where(data.valid_mask)[0])/2)
    else:
        if calib_eval:
            smx = logits[data.calib_test_real_mask]
            labels = data.y[data.calib_test_real_mask].detach().cpu().numpy()
        else:
            smx = logits[data.calib_test_mask]
            labels = data.y[data.calib_test_mask].detach().cpu().numpy()

    cov_all = []
    eff_all = []
    if return_prediction_sets:
        pred_set_all = []
        val_labels_all = []
        idx_all = []
        
    for k in range(100):
        idx = np.array([1] * n_base + [0] * (smx.shape[0]-n_base)) > 0
        np.random.seed(k)
        np.random.shuffle(idx)
        if return_prediction_sets:
            idx_all.append(idx)
        cal_smx, val_smx = smx[idx,:], smx[~idx,:]
        cal_labels, val_labels = labels[idx], labels[~idx]
        
        if use_additional_calib and calib_eval:
            smx_add = logits[data.calib_eval_mask]
            labels_add = data.y[data.calib_eval_mask].detach().cpu().numpy()
            cal_smx = np.concatenate((cal_smx, smx_add))
            cal_labels = np.concatenate((cal_labels, labels_add))
            
        n = cal_smx.shape[0]
        
        if score == 'tps':
            prediction_sets, cov, eff = tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)  
        elif score == 'old_aps':
            prediction_sets, cov, eff = old_aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)  
        elif score == 'new_aps':
            prediction_sets, cov, eff = new_aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)  
        elif score == 'raps':
            prediction_sets, cov, eff = raps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)  
        elif score == "new_raps":
            prediction_sets, cov, eff = raps_fixed(cal_smx, val_smx, cal_labels, val_labels, n, alpha)
        elif score == 'threshold':
            prediction_sets, cov, eff = threshold(cal_smx, val_smx, cal_labels, val_labels, n, alpha)  
            
        cov_all.append(cov)
        eff_all.append(eff)
        if return_prediction_sets:
            pred_set_all.append(prediction_sets)
            val_labels_all.append(val_labels)
    
    if return_prediction_sets:
        return cov_all, eff_all, pred_set_all, val_labels_all, idx_all
    else:
        return np.mean(cov_all), np.mean(eff_all)

def run_conformal_regression(pred, data, n, alpha, calib_eval = False, validation_set = False, use_additional_calib = False, return_prediction_sets = False, calib_fraction = 0.5, score = 'cqr'): 
    if calib_eval:
        n_base = int(n * (1-calib_fraction))
    else:
        n_base = n
    
    try:
        pred = pred.detach().cpu().numpy()
    except:
        pass
                     
    if validation_set:
        smx = pred[data.valid_mask]
        labels = data.y[data.valid_mask].detach().cpu().numpy().reshape(-1)
        n_base = int(len(np.where(data.valid_mask)[0])/2)
    else:
        if calib_eval:
            smx = pred[data.calib_test_real_mask]
            labels = data.y[data.calib_test_real_mask].detach().cpu().numpy().reshape(-1)
        else:
            smx = pred[data.calib_test_mask]
            labels = data.y[data.calib_test_mask].detach().cpu().numpy().reshape(-1)
    
    cov_all = []
    eff_all = []
    if return_prediction_sets:
        pred_set_all = []
        val_labels_all = []
        idx_all = []
    for k in range(100):
        upper, lower = smx[:, 2], smx[:, 1]

        idx = np.array([1] * n_base + [0] * (labels.shape[0]-n_base)) > 0
        np.random.seed(k)
        np.random.shuffle(idx)
        if return_prediction_sets:
            idx_all.append(idx)
        cal_labels, val_labels = labels[idx], labels[~idx]
        cal_upper, val_upper = upper[idx], upper[~idx]
        cal_lower, val_lower = lower[idx], lower[~idx]
        if score == 'cqr':
            prediction_sets, cov, eff = cqr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha)
        elif score == 'qr':
            prediction_sets, cov, eff = qr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha)
            
        
        cov_all.append(cov)
        eff_all.append(eff)
        if return_prediction_sets:
            pred_set_all.append(prediction_sets)
            val_labels_all.append(val_labels)
    
    if return_prediction_sets:
        return cov_all, eff_all, pred_set_all, val_labels_all, idx_all
    else:
        return np.mean(cov_all), np.mean(eff_all)