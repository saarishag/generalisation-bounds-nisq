import numpy as np

def calc_margin(svm, K_train, y_train):
    """
    Compute the margin using the provided kernel and the
    dual solution from the svm
    """
    alpha_y = svm.dual_coef_[0] #shape (1, n_support_vectors)

    support_indices = svm.support_ #indices of support vectors
    y_support_vec = y_train[support_indices] #+1/-1 (labels) of the support vectors

    alpha = alpha_y/y_support_vec #to get alpha only not alpha_i*y_i
    K_sv = K_train[np.ix_(support_indices, support_indices)]
    Y = np.diag(y_support_vec)
    sq_weighted_norm = alpha.T @ Y @ K_sv @ Y @ alpha
    margin = 1/sq_weighted_norm
    return np.sqrt(margin)

def calc_C_bounds(p, clean_margin, noisy_margin_est, n, n_layers):
    """Compute the range of acceptable values for 
    C to be used in the upper bound """
    prob_term = (1-p)**(2*n*n_layers)
    Cmax = (2-prob_term)/(2*(clean_margin**2))
    Cmin_est = Cmax - (prob_term/(2*(noisy_margin_est**2)))
    return Cmin_est, Cmax

def get_upper_params(): 
    """
    Define parameters for the upper bound calc
    Values below obtained using C_region_test.py and 
    chosen to ensure bound validity for the Heart Dataset
    """
    C = 10 
    C_bound = 10 #Or C_bound = C/m 
    clean_margin = 0.1674201006091044
    return C, C_bound, clean_margin

def calc_upper_bound(p_local, n, n_layers, clean_margin, C_bound):
    """
    Function to compute the upper bound for a given p_local value
    """
    prob_term = (1-p_local)**(2*n*n_layers)
    denominator = (2*(1-(C_bound*(clean_margin**2))))-prob_term
    sq_bound = (prob_term*(clean_margin**2))/denominator
    bound = np.sqrt(np.abs(sq_bound))
    return bound

def get_p_local_vals(n, clean_margin, C_bound):
    """
    Function to get the acceptable p_local values
    on which to test the lower bound based on the value
    of a constraint
    """
    constraint = C_bound*(clean_margin**2)
    max_min_p = 1-(2**(-1/(2*n))) #max or min p depending on constraint
    if constraint < 0.5:
        #p_local_list = [0, 0.01, 0.05, 0.07, 0.1, 0.135, 0.15]  #used for n=2, constraint<0.5
        p_local_list = np.linspace(0, max_min_p, num = 7) #Arbitrary num chosen (can be changed)
        print(f"Constraint = {constraint} < 0.5") 
    elif constraint > 0.5:
        p_local_list = np.linspace(max_min_p, 1, num = 7)
        print(f"Constraint = {constraint} > 0.5") 
    else: 
        raise ValueError("No acceptable p values for valid bounds") #Constraint must NOT be 1/2 for valid results
    return p_local_list
         
def calc_lower_bound(p_local, n, clean_margin, C_bound):
    """
    Function to compute the lower bound for a given p_local value
    """
    lower_bound = (clean_margin**2)*(1-(2*(1-p_local)**(2*n)))
    lower_bound /= (2*C_bound*(clean_margin**2)) - 1
    return lower_bound


