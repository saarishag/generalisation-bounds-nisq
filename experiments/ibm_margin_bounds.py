import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler

from ibm_bounds_definitions import create_iqp_feature_map, create_training_overlap_circuit_list, create_testing_overlap_circuit_list, compute_overlap_matrix
from src.dataset_config import define_BC_dataset
from src.bounds_definitions import calc_margin
from results.results import ideal_kernel_BC, ibm_results, upper_lower_bounds_BC
from src.plotting_fns import ibm_plots

np.random.seed(42)

#get a subset of 20 samples from BC dataset
X_subset, y_subset, n, n_layers, _, _ = define_BC_dataset(start = 200, stop=220) 

#Split first then do preprocessing on train sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_subset, 
        y_subset, 
        test_size = 0.2, 
        random_state=42)  

X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, 
        y_train_val, 
        test_size = 0.2,   
        random_state=42) #altogether a 12/4/4 test/train/val sample split

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

train_size = len(X_train)
test_size = len(X_test)

y_train = np.array(y_train_scaled).ravel()
y_test = np.array(y_test_scaled).ravel()
num_samples = len(X_train)

#Define initial properties
C0 = 1000 
num_features=2
num_shots = 10000

#Create IQP-style feature map
fm = create_iqp_feature_map(n, num_features, reps=n_layers)

# Get a specific backend or the least busy
service = QiskitRuntimeService()
backend = service.backend("ibm_strasbourg") 

# Running on a real hardware
sampler = Sampler(mode=backend)

# Create the circuits for training and testing overlaps
training_overlap_circ_list = create_training_overlap_circuit_list(train_size, X_train, fm)

#Generate pass managers with optimisation level = 3 for this backend
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

#ISA = Instruction Set Architecture
isa_circuit_list = [pm.run(circuit) for circuit in training_overlap_circ_list]
job_training = sampler.run(isa_circuit_list, shots=num_shots) 

testing_overlap_circ_list = create_testing_overlap_circuit_list(test_size, train_size, X_test, X_train, fm)

isa_circuit_list = [pm.run(circuit) for circuit in testing_overlap_circ_list]
job_testing = sampler.run(isa_circuit_list, shots=num_shots)

# Compute training and testing matrix
results_training = job_training.result() #sampler returns meas outcome distributions
kernel_matrix = compute_overlap_matrix(num_shots, results_training, train_size, train_size, is_symmetric=True)
print("Training matrix done")

results_testing = job_testing.result()
test_matrix = compute_overlap_matrix(num_shots, results_testing, test_size, train_size, is_symmetric=False)
print("Test matrix done")

#Use a pre-computed kernel matrix
qml_svc = SVC(kernel="precomputed", C=C0)

# Fit the model using the quantum kernel matrix
qml_svc.fit(kernel_matrix, y_train)

print("QSVM predictions", qml_svc.predict(test_matrix))
print("Test labels", y_test)

# Use the .score to test the model
qml_score_precomputed_kernel = qml_svc.score(test_matrix, y_test)
print(f"Precomputed kernel classification test score: {qml_score_precomputed_kernel}")

#separately compute the ideal kernel for margin computation on the hardware using get_clean_matrix fn from margin_bounds.py
clean_kernel = ideal_kernel_BC() 

#Compute the noisy margin using the svm from the hardware and the ideal clean kernel
noisy_margin = calc_margin(qml_svc, clean_kernel, y_train)

"""
#Uncomment the following code to get the pre-computed clean kernel matrix and resulting noisy margin
ibm_margin = ibm_results()

##################################################################
#Upper Bound for BC dataset (Simulated - for comparison)

#Results from C_region_test.py:

#feasible C_bound region = [0,16.82633547333059)
#C0 = 1000 for consistency and C_bound=C0/(m**3) for valid bounds
#and clean_margin = 0.1723813313166903 

#Use the above in margin_bounds.py

#Uncomment the following code to duplicate plots from the paper

p_local_list_upper, noisy_margin_upper, upper_bound_arr, p_local_list_lower, noisy_margin_lower, lower_bound_arr = upper_lower_bounds_BC()
ibm_plots(p_local_list_upper, noisy_margin_upper, upper_bound_arr, p_local_list_lower, noisy_margin_lower, lower_bound_arr, ibm_margin)
"""
