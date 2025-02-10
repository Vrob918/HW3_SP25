import numpy as np
import scipy.linalg

# matrix is symmetric
def check_symmetric(matrix):
    if np.array_equal(matrix, matrix.T):
        return True
    else:
        return False

# matrix is positive definite
def check_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)  # Trying Cholesky decomposition to check
        return True
    except:
        return False

# Cholesky method
def cholesky_method(A, b):
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x

# Doolittle method
def doolittle_method(A, b):
    P, L, U = scipy.linalg.lu(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x

def main():
    # First system matrix
    A1 = np.array([[1, -1, 3, 2],
                   [-1, 5, -5, -2],
                   [3, -5, 19, 3],
                   [2, -2, 3, 21]], dtype=float)
    b1 = np.array([15, -35, 94, 1], dtype=float)

    # Second system matrix
    A2 = np.array([[4, 2, 4, 0],
                   [2, 2, 3, 2],
                   [4, 3, 6, 3],
                   [0, 2, 3, 9]], dtype=float)
    b2 = np.array([20, 36, 60, 122], dtype=float)

    # Solving the first system
    print("I am solving the first system A1x = b1 now...")
    if check_symmetric(A1) and check_positive_definite(A1):
        print("A1 is symmetric and positive definite")
        solution1 = cholesky_method(A1, b1)
    else:
        print("A1 is NOT symmetric or positive definite")
        solution1 = doolittle_method(A1, b1)

    print("The solution for the first system is:")
    print(solution1)

    # Solving the second system
    print("\nNow I am solving the second system A2x = b2...")
    if check_symmetric(A2) and check_positive_definite(A2):
        print("A2 is symmetric and positive definite")
        solution2 = cholesky_method(A2, b2)
    else:
        print("A2 is NOT symmetric or positive definite")
        solution2 = doolittle_method(A2, b2)

    print("The solution for the second system is:")
    print(solution2)

# run the program
if __name__ == "__main__":
    main()