import numpy as np
import pandas as pd
from pennylane.templates import IQPEmbedding 
from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_blobs

def define_heart_dataset():
    """
    Fetches and prepares (preprocesses) Heart Disease Dataset 
    and defines various parameters for binary classification problem 
    """
    n=2
    n_layers = 1
    embedding = IQPEmbedding #define embedding used for the (Pennylane) quantum circuit
    p_local_list = [0, 0.05, 0.1, 0.2, 0.3, 0.375, 0.5, 0.6, 0.7, 0.75 ]

    #fetch heart disease dataset
    heart_disease = fetch_ucirepo(id=45)

    #data as pd dataframes 
    X = heart_disease.data.features
    y = heart_disease.data.targets

    #variable information
    y_binary = np.where(y['num']>0, 1, 0) #x,y -> x returned if True, y returned if False
    
    #Convert to DataFrame
    y_binary = pd.DataFrame(y_binary, columns = ['heart_disease'])

    print(y_binary['heart_disease'].value_counts())

    #Impute missing values (Use median)
    imputer = SimpleImputer(strategy="median") 
    X_imputed = imputer.fit_transform(X)

    #convert to df
    X_imputed = pd.DataFrame(X_imputed, columns = X.columns)

    print(X_imputed)
    print(X_imputed.isna().sum())

    X_subset = np.array(X_imputed)
    y_subset = np.array(y_binary)

    return X_subset, y_subset, n, n_layers, embedding, p_local_list


def define_wine_dataset():
    """
    Fetches and prepares (preprocesses) (White) Wine Quality Dataset 
    and defines various parameters for binary classification problem 
    """
    n=2 #can be changed 
    n_layers = 1
    embedding = IQPEmbedding
    p_local_list = [0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.375, 0.4, 0.45, 0.5, 0.6, 0.75]

    #fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    #data as pandas dataframes
    X = wine_quality.data.features
    y = wine_quality.data.targets

    #Making y binary by setting all wine quality above and = 6 to 1 and below to 0
    #so classification is now between high quality wine vs low quality 
    #Above average vs below average/Premium wine vs standard wine/Acceptable vs Unacceptable

    y_binary = np.where(y['quality']>=6, 1, 0)
    y_binary = pd.DataFrame(y_binary, columns = ['quality'])

    X_subset = np.array(X[:1000])
    y_subset = np.array(y_binary[:1000]) #first 1000 -> 495 in class 1, 505 in class 0 ~imbalanced

    return X_subset, y_subset, n, n_layers, embedding, p_local_list

def define_gaussian_dataset():
    """
    Fetches and prepares (preprocesses) synthetic Gaussian Dataset 
    and defines various parameters for binary classification problem 
    """
    n=2 #can be changed 
    n_layers = 1
    embedding = IQPEmbedding
    p_local_list = [0, 0.05, 0.1, 0.25, 0.375, 0.5, 0.75] #p_values used for C range test
    
    def create_gaussians_with_overlap(cluster_stdev = 3.0, n_samples = 500, random_state = 42):
        """Create blobs (gaussians) with controlled overlap
        cluster_stdev = Cluster standard deviation
        Larger cluster_stdev = More overlap = Less separable
        """

        centers = [[-2,-2], [2,2]] #define centers of the gaussian distributions (blobs)
        X,y = make_blobs(n_samples=n_samples, centers = centers, 
                        cluster_std=cluster_stdev, random_state=random_state)
        return X,y

    X_subset,y_subset = create_gaussians_with_overlap()
    return X_subset, y_subset, n, n_layers, embedding, p_local_list

def define_BC_dataset(start, stop):
    """
    Fetches and prepares (preprocesses) Breast Cancer Dataset 
    and defines various parameters for binary classification problem 
    """
    n=2 #can be changed 
    n_layers = 1
    embedding = IQPEmbedding

    p_local_list = [0, 0.05, 0.1, 0.25, 0.375, 0.5, 0.75] #p_values used for C range test
        
    #fetch dataset
    breast_cancer = fetch_ucirepo(id=17)

    #Data (as pandas dataframe)
    X = breast_cancer.data.features
    y = breast_cancer.data.targets

    #Variable info 
    print(breast_cancer.variables)

    y_binary = np.where(y=='M', 1, 0)
    y_binary = pd.DataFrame(y_binary, columns = ['Diagnosis'])

    X_subset = np.array(X[start:stop])
    y_subset = np.array(y_binary[start:stop])
    
    return X_subset, y_subset, n, n_layers, embedding, p_local_list


