import numpy as np
import scipy.linalg

# Check if matrix is symmetric
def check_symmetric(matrix):
    if np.array_equal(matrix, matrix.T):
        return True
    else:
        return False

# Function to check if matrix is positive definite
def check_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)  # Trying Cholesky decomposition to check
        return True
    except:
        return False

# Cholesky method for solving the system
def cholesky_method(A, b):
    L = np.linalg.cholesky(A)  # Decompose A into lower triangular L
    y = np.linalg.solve(L, b)  # Solve Ly = b
    x = np.linalg.solve(L.T, y)  # Solve L^T x = y
    return x

# Doolittle method for LU decomposition to solve Ax = b
def doolittle_method(A, b):
    P, L, U = scipy.linalg.lu(A)  # LU Decomposition of A
    y = np.linalg.solve(L, b)  # Solve Ly = b
    x = np.linalg.solve(U, y)  # Solve Ux = y
    return x

# The main function where I will solve the problems
def main():
    # First system matrix A1 and vector b1
    A1 = np.array([[1, -1, 3, 2],
                   [-1, 5, -5, -2],
                   [3, -5, 19, 3],
                   [2, -2, 3, 21]], dtype=float)
    b1 = np.array([15, -35, 94, 1], dtype=float)

    # Second system matrix A2 and vector b2
    A2 = np.array([[4, 2, 4, 0],
                   [2, 2, 3, 2],
                   [4, 3, 6, 3],
                   [0, 2, 3, 9]], dtype=float)
    b2 = np.array([20, 36, 60, 122], dtype=float)

    # Solving the first system
    print("I am solving the first system A1x = b1 now...")
    if check_symmetric(A1) and check_positive_definite(A1):
        print("A1 is symmetric and positive definite, so I will use the Cholesky method.")
        solution1 = cholesky_method(A1, b1)
    else:
        print("A1 is NOT symmetric or positive definite, so I am using Doolittle method.")
        solution1 = doolittle_method(A1, b1)

    print("The solution for the first system is:")
    print(solution1)

    # Solving the second system
    print("\nNow I am solving the second system A2x = b2...")
    if check_symmetric(A2) and check_positive_definite(A2):
        print("A2 is symmetric and positive definite, so I will use the Cholesky method.")
        solution2 = cholesky_method(A2, b2)
    else:
        print("A2 is NOT symmetric or positive definite, so I am using Doolittle method.")
        solution2 = doolittle_method(A2, b2)

    print("The solution for the second system is:")
    print(solution2)

# run the program
if __name__ == "__main__":
    main()