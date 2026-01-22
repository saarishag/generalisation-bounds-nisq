import pennylane as qml
import numpy as np

def clean_rho_fn(n, n_layers, embedding):
    """
    Function that returns the density matrix resulting from
    an ideal (noiseless) circuit with [n_layers] unitary layers 
    of the chosen [embedding]  
    """
    dev = qml.device("default.mixed", wires = n) 

    @qml.qnode(dev)
    def get_clean_rho(x):
        for _ in range(n_layers): 
            embedding(x, wires = range(n)) 
        return qml.density_matrix(wires = range(n))
    return get_clean_rho


def get_clean_matrix(A,B, fn_clean_rho): 
    """
    Function that computes the (ideal) kernel values between 
    all pairs of elements from the provided datasets, A and B
    and returns a matrix of these elements (the kernel matrix)
    """
    def noiseless_kernel_element(x_i, x_j):
        rho_i = np.array(fn_clean_rho(x_i)) 
        rho_j = np.array(fn_clean_rho(x_j))
        return np.real(np.trace(rho_i @ rho_j))

    #pass a function that computes a matrix of kernel evaluations for datasets A and B
    def noiseless_kernel_matrix(A, B):
        """Compute the matrix whose entries are the kernel
        evaluated on pairwise data from sets A and B"""
        ideal_kernel_matrix = np.array([[noiseless_kernel_element(a,b) for b in B] for a in A])
        return ideal_kernel_matrix
    clean_K = noiseless_kernel_matrix(A,B)
    return clean_K
        
def local_rho_fn(p, n, n_layers, embedding):
    """
    Function that returns the density matrix resulting from
    a circuit with [n_layers] unitary layers 
    of the chosen [embedding] and [n_layers] depolarising 
    noise channels applied on each qubit
    """
    dev = qml.device("default.mixed", wires = n)
     
    @qml.qnode(dev)
    def get_local_rho(x):
        for _ in range(n_layers): 
            embedding(x, wires = range(n)) 
            for i in range(n):
                qml.DepolarizingChannel(p, wires = i)
        return qml.density_matrix(wires = range(n))
    return get_local_rho

def get_local_matrix(A, B, fn_local_rho):
    """
    Function that computes the kernel values obtained 
    using circuits with local depolarising noise channels
    and returns a matrix of these elements (the kernel matrix) 
    """
    def local_noisy_kernel_element(x_i, x_j):
        rho_i_loc = np.array(fn_local_rho(x_i))
        rho_j_loc = np.array(fn_local_rho(x_j))
        return np.real(np.trace(rho_i_loc @ rho_j_loc))

    def local_noisy_kernel(A,B):
        local_noisy_kernel_matrix = np.array([[local_noisy_kernel_element(a,b) for b in B] for a in A])
        return local_noisy_kernel_matrix
    
    noisy_K_test_loc = local_noisy_kernel(A, B)
    return noisy_K_test_loc


def get_global_matrix(A, B, p, n, clean_rho):
    """
    Function that computes the kernel values obtained 
    using circuits with global depolarising noise channels
    and returns a matrix of these elements (the kernel matrix) 
    """
    def get_global_rho(x):
            """
            Density matrix affected by global depolarising noise
            computed analytically
            """
            rho = np.array(clean_rho(x))
            noisy_rho = (1-p)*rho + (p/(2**n))*np.eye(2**n)
            return noisy_rho

    def global_noisy_kernel_element(x_i, x_j):
        rho_i_global = np.array(get_global_rho(x_i))
        rho_j_global = np.array(get_global_rho(x_j))
        return np.real(np.trace(rho_i_global @rho_j_global))

    def global_noisy_kernel(A,B, p=p, n=n):
        global_noisy_kernel_matrix = np.array([[global_noisy_kernel_element(a,b) for b in B] for a in A])
        return global_noisy_kernel_matrix
    
    noisy_K_test_global = global_noisy_kernel(A, B, p, n)
    return noisy_K_test_global

        
