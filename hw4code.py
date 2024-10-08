import numpy as np

def lu(A):
    '''Computes the (strict) LU factorization of the matrix A'''
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    
    for j in range(n-1):
        if U[j, j] == 0:
            return None, None
        for i in range(j+1, n):
            multiplier = U[i, j] / U[j, j]
            L[i, j] = multiplier
            for k in range(j+1, n):
                U[i, k] -= multiplier * U[j, k]
            U[i, j] = 0
    return L, U

A = np.array([[1, 2],
              [3, 4]])
L, U = lu(A)[0], lu(A)[1]
print("Test 1")
print("A = \n", A)
print("L = \n", L)
print("U = \n", U)
print("L*U = \n", np.dot(L, U))
assert np.dot(L, U).all() == A.all()

A = np.array([[2, 1, 1],
              [4, -6, 0],
              [-2, 7, 2]])
L, U = lu(A)[0], lu(A)[1]
print("\nTest 2")
print("A = \n", A)
print("L = \n", L)
print("U = \n", U)
print("L*U = \n", np.dot(L, U))
assert np.dot(L, U).all() == A.all()

if __name__ == "__main__":
    pass
    import timeit
    import matplotlib.pyplot as plt

    matrix_sizes = []
    times = []

    for n in range(1, 21):
        size = 10 * n
        matrix_sizes.append(size)
        
        start_all = timeit.default_timer()
        A = np.random.randn(size, size)
        lu(A)
        stop_all = timeit.default_timer()
        
        elapsed_time = stop_all - start_all
        times.append(elapsed_time)
        
        print("\nTrial: ", n)
        print("n = ", size)
        print('Time: ', elapsed_time)

    plt.figure()
    plt.plot(matrix_sizes, times)
    plt.title('LU Decomposition Algorithm Performance')
    plt.xlabel('Matrix Size (n x n)')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.show()


