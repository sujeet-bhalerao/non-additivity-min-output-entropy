import numpy as np
from scipy.optimize import minimize
from scipy.linalg import logm

def von_neumann_entropy(rho):
    return -np.trace(rho @ logm(rho))

def kraus_channel(rho, kraus_ops):
    output = np.zeros_like(rho)
    for k in kraus_ops:
        output += k @ rho @ k.conj().T
    return output

def tensor_product_kraus_operators(kraus_ops1, kraus_ops2):
    return [np.kron(k1, k2) for k1 in kraus_ops1 for k2 in kraus_ops2]

def objective_function(params, kraus_ops, n=2):
    ket = params[:n] + 1j * params[n:2 * n]
    ket /= np.linalg.norm(ket)
    rho = np.outer(ket, ket.conj())
    output = kraus_channel(rho, kraus_ops)
    return von_neumann_entropy(output)

def random_nonunital_kraus_ops():
    while True:
        kraus_ops = [np.random.randn(2, 2) + 1j * np.random.randn(2, 2) for _ in range(2)]
        if not np.allclose(sum(k @ k.conj().T for k in kraus_ops), np.eye(2)):
            return kraus_ops

def check_additivity(counterexample_threshold=1e-5, num_trials=1000):
    for _ in range(num_trials):
        print("Trial", _ + 1, "of", num_trials)
        kraus_ops1 = random_nonunital_kraus_ops()
        kraus_ops2 = random_nonunital_kraus_ops()
        kraus_ops_prod = tensor_product_kraus_operators(kraus_ops1, kraus_ops2)

        # Optimize over pure input states for E1, E2, and E1 ⊗ E2
        params = np.random.randn(4)
        result1 = minimize(objective_function, params, args=(kraus_ops1,))
        result2 = minimize(objective_function, params, args=(kraus_ops2,))
        params_4x4 = np.random.randn(8)
        result_prod = minimize(objective_function, params_4x4, args=(kraus_ops_prod, 4))

        # Check for counterexample
        if result_prod.fun > result1.fun + result2.fun + counterexample_threshold:
            print("Counterexample found:")
            print("H_min(E1):", result1.fun)
            print("H_min(E2):", result2.fun)
            print("H_min(E1 tensor E2):", result_prod.fun)
            
            print("\nKraus operators for E1:")
            for i, k in enumerate(kraus_ops1):
                print(f"K1_{i}:\n", k)

            print("\nKraus operators for E2:")
            for i, k in enumerate(kraus_ops2):
                print(f"K2_{i}:\n", k)

            return

    print("No counterexample found")

check_additivity()