import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

from src.kernel_definitions import clean_rho_fn, get_clean_matrix, local_rho_fn, get_local_matrix, get_global_matrix
from src.dataset_config import define_wine_dataset, define_heart_dataset
from src.plotting_fns import plot_local_global
from results.results import local_global_results

np.random.seed(42)

X_subset, y_subset, n, n_layers, embedding, p_local_list = define_heart_dataset() #example usage - Heart Disease Dataset

#Initialise pipeline
preprocessor = make_pipeline(
        StandardScaler(),
        PCA(n_components=n, random_state = 42),
        MinMaxScaler(feature_range=(0,np.pi)) 
    )
 
mean_local_acc = []
std_local_acc = []

mean_global_acc = []
std_global_acc = []

for p_local in p_local_list:
    accuracy_loc = []
    accuracy_glob = []

    #Perform 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_subset)):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y_subset[train_idx], y_subset[test_idx]
            
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

        #Get ideal kernel 
        clean_rho = clean_rho_fn(n=n, n_layers=n_layers, embedding=embedding)
        clean_K_train = get_clean_matrix(A = X_train, B=X_train, fn_clean_rho = clean_rho)
        svm_clean = SVC(kernel = 'precomputed').fit(clean_K_train, y_train) #Use default C=1 value

        #Get noisy kernel (local noise model)
        local_rho = local_rho_fn(p_local, n, n_layers, embedding)
        noisy_K_test_loc = get_local_matrix(A=X_test, B=X_train, fn_local_rho = local_rho)
        
        #Get test accuracy when using ideal QSVM model to predict on noisy data 
        predict_loc = svm_clean.predict(noisy_K_test_loc)
        acc_loc = metrics.accuracy_score(y_test, predict_loc)
        print(f"{p_local}: {acc_loc}")
        accuracy_loc.append(acc_loc)

        #Get noisy kernel (global noise model)
        p_global = 1 - (1-p_local)**(n*n_layers) #to match survival probabilities of local and global models
        noisy_K_test_global = get_global_matrix(A=X_test, B=X_train, p=p_global, n=n, clean_rho = clean_rho)
        
        #Get test accuracy when using ideal QSVM model to predict on noisy data 
        predict_glob = svm_clean.predict(noisy_K_test_global)
        acc_glob = metrics.accuracy_score(y_test,predict_glob)

        print(f"{p_local}({p_global}):{acc_glob}")
        accuracy_glob.append(acc_glob)

    mean_local_acc.append(np.mean(accuracy_loc))
    std_local_acc.append(np.std(accuracy_loc))
    print(f"Mean local accuracy: {np.mean(accuracy_loc)} +/- {np.std(accuracy_loc)}")
        
    mean_global_acc.append(np.mean(accuracy_glob))
    std_global_acc.append(np.std(accuracy_glob))
    print(f"Mean global accuracy: {np.mean(accuracy_glob)} +/- {np.std(accuracy_glob)}")
        

#Save results to a file
filename = f"Wine_LocalVGlobalAcc.txt"

with open(filename, 'a') as file:
    file.write("---------------------------------------------------------------------------\n")
    file.write(f"N=2, L=1\n")
    for p_local, mean_loc_acc, std_loc_acc, mean_glob_acc, std_glob_acc in zip(p_local_list, mean_local_acc, std_local_acc, mean_global_acc, std_global_acc):
        file.write(f"p_local = {p_local}\n")
        file.write(f"Local Accuracy = {mean_loc_acc} +/- {std_loc_acc} \n")
        file.write(f"Global Accuracy = {mean_glob_acc} +/- {std_glob_acc} \n")
  
with open(filename, 'r') as file: #read
    content = file.read()
    print(content)

"""
#Uncomment to duplicate plots from the paper 
 
heart_data, wine_2N1L, wine_3N1L = local_global_results()
plot_local_global(heart_data, wine_2N1L, wine_3N1L,n, n_layers)
"""