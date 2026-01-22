import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from src.kernel_definitions import clean_rho_fn, get_clean_matrix, local_rho_fn, get_local_matrix
from src.dataset_config import define_wine_dataset, define_heart_dataset, define_BC_dataset, define_gaussian_dataset
from src.bounds_definitions import calc_margin, get_upper_params, calc_upper_bound, get_p_local_vals, calc_lower_bound 
from results.results import margin_results, upper_bound_results, lower_bound_results
from src.plotting_fns import plot_upper_bound, plot_lower_bound
np.random.seed(42)

X_subset, y_subset, n, n_layers, embedding, p_local_list = define_heart_dataset() #Example usage - Heart Disease Dataset

#Split first then do preprocessing on train sets
X_train, X_test, y_train, y_test = train_test_split(
        X_subset, 
        y_subset, 
        test_size = 0.25, 
        random_state=42) 

#Initialise pipeline
preprocessor = make_pipeline(
        StandardScaler(),
        PCA(n_components=n, random_state = 42),
        MinMaxScaler(feature_range=(0,np.pi)) 
    )

#Fit on training data
preprocessor.fit(X_train)

#Transform both x sets
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)

#Scale train and test y_labels
y_train_scaled = 2*(y_train-0.5)
y_test_scaled = 2*(y_test-0.5) #convert from [0,1] to [-1,1]

num_samples = len(X_train)

y_train = np.array(y_train_scaled).ravel()
y_test = np.array(y_test_scaled).ravel()

#Get parameters obtained from running C_region_test.py script
C0, C_bound, clean_margin = get_upper_params() 
print(f"C_bound = {C_bound}")
print(f"C0 = {C0}")
print(f"Clean Margin = {clean_margin}") #Retrieve margin computed in C_region_test.py script #For consistency/valid bounds

#Redefine p_local_list if necessary
p_local_list = [0, 0.05, 0.1, 0.25, 0.375, 0.5, 0.75] 

#Get the ideal kernel matrix
clean_rho = clean_rho_fn(n=n, n_layers=n_layers, embedding=embedding)
clean_K = get_clean_matrix(A = X_train, B = X_train, fn_clean_rho = clean_rho)

margin_arr = []
bound_arr = []

#Iterate through all p_local values 
for p_local in p_local_list:

    local_rho = local_rho_fn(p_local, n, n_layers, embedding)
    noisy_K_train = get_local_matrix(A=X_train, B=X_train, fn_local_rho = local_rho)
    svm_noisy = SVC(kernel = "precomputed", C=C0).fit(noisy_K_train, y_train)

    #Calc and store noisy margin and respective upper bound value for p_local
    noisy_margin = calc_margin(svm_noisy, clean_K, y_train) #compute noisy dual solution in noiseless RKHS space
    print(f"{p_local}: {noisy_margin}")

    upper_bound = calc_upper_bound(p_local, n, n_layers, clean_margin, C_bound)    
    print(f"{p_local}: {upper_bound}")

    margin_arr.append(noisy_margin)
    bound_arr.append(upper_bound)
 
    if noisy_margin <= upper_bound:
        print("Upper Bound")
    else:
        print("Bound Violated")

#Save results to text file
filename = "Heart_MarginBounds.txt"

with open(filename, 'a') as file:
    file.write(f"Upper Bounds\n")
    file.write(f"C_bound = {C_bound}\n")
    file.write(f"C0 = {C0}\n")
    file.write(f"m = {num_samples}\n")
    for p_local, noisy_margin, upper_bound in zip(p_local_list, margin_arr, bound_arr):
        file.write("---------------------------------------------------------------------------\n")
        file.write(f"p_local = {p_local}\n")
        file.write(f"Noisy Margin = {noisy_margin}\n")
        file.write(f"Bound = {upper_bound}\n")
        if (noisy_margin <= upper_bound):
            file.write("Upper Bound \n")
        else:
            file.write("Bound Violated \n")
  
with open(filename, 'r') as file: #read
    content = file.read()
    print(content)

"""
#Uncomment to duplicate plots from the paper 
 
p_local_list, heart_margin, gaus_margin, bc_margin, wineL1_margin, wineL2_margin = margin_results()
p_local_list, heart_upper, gaus_upper, bc_upper, wineL1_upper, wineL2_upper = upper_bound_results()

plot_upper_bound(p_local_list,heart_margin, heart_upper, gaus_margin, gaus_upper, bc_margin, bc_upper, wineL1_margin, wineL1_upper, wineL2_margin, wineL2_upper)
"""
#####################################################################################################

"""
Lower Bound Computation
"""

#Define acceptable p_local values according to conditions for non-trivial bounds
p_local_list = get_p_local_vals(n, clean_margin, C_bound)

noisy_margin_lower = []
lower_bound_arr = []
for p_local in p_local_list:

    local_rho = local_rho_fn(p_local, n, n_layers, embedding)
    noisy_K_train = get_local_matrix(A=X_train, B=X_train, fn_local_rho = local_rho)
    svm_noisy = SVC(kernel = "precomputed", C=C0).fit(noisy_K_train, y_train)

    #Calc and store noisy margin and respective lower bound value for p_local
    noisy_margin = calc_margin(svm_noisy, clean_K, y_train) #compute noisy dual solution in noiseless RKHS space
    print(f"{p_local}: {noisy_margin}")

    lower_bound = calc_lower_bound(p_local, n, clean_margin, C_bound)
    print(f"{p_local}: {lower_bound}")

    noisy_margin_lower.append(noisy_margin)
    lower_bound_arr.append(lower_bound)
    
    
    if (noisy_margin >= lower_bound):
        print("Lower Bound")
    else:
        print("Bound Violated")


#Save results to a text file

with open(filename, 'a') as file:
    file.write(f"Lower Bounds\n")
    for p_local, noisy_marg, lower_bound in zip(p_local_list, noisy_margin_lower, lower_bound_arr):
        file.write("---------------------------------------------------------------------------\n")
        file.write(f"p_local = {p_local}\n")
        file.write(f"Noisy Margin = {noisy_marg}\n")
        file.write(f"Bound = {lower_bound}\n")
        if (noisy_marg >= lower_bound):
            file.write("Lower Bound \n")
        else:
            file.write("Bound Violated \n")
  
with open(filename, 'r') as file: #read
    content = file.read()
    print(content)

"""
#Uncomment to duplicate plots from the paper 
 
p_local_list, heart_margin, gaus_margin, bc_margin, wineL1_margin, wineL2_margin = margin_results()
p_local_list, heart_lower, gaus_lower, bc_lower, wineL1_lower, wineL2_lower = lower_bound_results()

plot_lower_bound(p_local_list,heart_margin, heart_lower, gaus_margin, gaus_lower, bc_margin, bc_lower, wineL1_margin, wineL1_lower, wineL2_margin, wineL2_lower)
"""
