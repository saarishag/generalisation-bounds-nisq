import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import unitary_overlap

def create_iqp_feature_map(n, num_features, reps=1):
    """
    Define an IQP-style encoding using Qiskit
    """
    x = ParameterVector("x", num_features)
    qc = QuantumCircuit(n)

    for _ in range(reps):
        # Hadamards
        for i in range(n):
            qc.h(i)

        # Single-qubit Z encodings
        for i in range(n):
            qc.rz(2 * x[i % num_features], i)

        # ZZ entanglement (IQP-style)
        for i in range(n - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * x[(i + 1) % num_features], i + 1)
            qc.cx(i, i + 1)

    return qc

# Create training circuit lists
def create_training_overlap_circuit_list(train_size, X_train, feature_map):
    """
    Create the circuits for training overlap and measure all 
    """
    training_overlap_circ_list = [
        unitary_overlap(
            feature_map.assign_parameters(list(X_train[x1])),
            feature_map.assign_parameters(list(X_train[x2]))
        ) #U(x1) U(x2)
        for x1 in range(train_size) for x2 in range(x1 + 1, train_size)
    ]

    for circuit in training_overlap_circ_list:
        circuit.measure_all() #measure qubits in the computational basis

    return training_overlap_circ_list

# Create testing circuit lists computing overlaps between test and train data
def create_testing_overlap_circuit_list(test_size, train_size, X_test, X_train, feature_map):
    """
    Create the circuits for test and train overlap and measure all 
    """
    testing_overlap_circ_list = [
        unitary_overlap(
            feature_map.assign_parameters(list(X_test[x1])),
            feature_map.assign_parameters(list(X_train[x2]))
        )
        for x1 in range(test_size) for x2 in range(train_size)
    ]

    for circuit in testing_overlap_circ_list:
        circuit.measure_all()

    return testing_overlap_circ_list

def compute_overlap_matrix(num_shots, results, size1, size2, is_symmetric=True):
    """
    Computes the kernel matrix from the sampler results
    """
    overlap_matrix = np.zeros((size1, size2)) #size1 = number of rows (train/test size) #size2 = number of columns (train size)
    idx = 0

    for x1 in range(size1): 
        for x2 in range(x1 + 1, size2) if is_symmetric else range(size2): 
            counts = results[idx].data.meas.get_int_counts() 
            prob_0 = counts.get(0, 0.0) / num_shots #probability of an all 0 state
            overlap_matrix[x1, x2] = prob_0

            if is_symmetric: 
                overlap_matrix[x2, x1] = prob_0  # Mirror value for symmetry

            idx += 1

        if is_symmetric:
            overlap_matrix[x1, x1] = 1  # Diagonal elements should be 1 for training data

    return overlap_matrix

