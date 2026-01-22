import numpy as np

def per_sample_margin(svm, K_train, y_train):
    """
    Compute the geometric margin for each sample in the dataset
    by first obtaining the functional margin and 
    dividing that by the norm of the weight vector
    """
    alpha_y = svm.dual_coef_[0] #shape (1, n_support_vectors)

    support_indices = svm.support_ #indices of support vectors
    y_support_vec = y_train[support_indices] #+1/-1 (labels) of the support vectors
    alpha = alpha_y/y_support_vec 

    K_sv = K_train[np.ix_(support_indices, support_indices)]
    Y = np.diag(y_support_vec)
    sq_weighted_norm = alpha.T @ Y @ K_sv @ Y @ alpha

    f_train = svm.decision_function(K_train) #using the decision function
    func_margin = y_train * f_train #y_i * f_i #functional margins
    geom_margin = func_margin/np.sqrt(sq_weighted_norm) #array of per-sample geometric margins

    return geom_margin

def corrupt_labels(y, corrupt_lvl):
    """
    Randomly corrupt a fraction of labels 
    by randomly flipping their training labels
    i.e. -1 -> +1 and +1 -> -1
    """
    y_corrupted = y.copy()
    n_corrupt = int(corrupt_lvl*len(y)) #fraction*number of labels = integer

    if n_corrupt > 0:
        corrupt_indices = np.random.choice(len(y), n_corrupt, replace=False) #choose n random indices 
        y_corrupted[corrupt_indices] = -y_corrupted[corrupt_indices] #flip the labels
    return y_corrupted


