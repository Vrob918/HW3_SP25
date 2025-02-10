import numpy as np
from numericalMethods import GPDF, Probability


def Secant(fcn, x0, x1, maxiter=10, xtol=1e-5):
    x_diff = abs(xtol) + 1
    iter = 0
    f0 = fcn(x0)
    f1 = fcn(x1)

    while (iter < maxiter and abs(x_diff) > abs(xtol)):
        x_new = x1 - f1 * ((x1 - x0) / (f1 - f0))
        f0 = f1
        f1 = fcn(x_new)
        x_diff = x_new - x1
        x0 = x1
        x1 = x_new
        iter += 1

    return (x1, iter)


def main():
    again = True

    while again:
        # Get user input
        mean_input = input("Enter the population mean: ")
        mean = float(mean_input)

        stDev_input = input("Enter the standard deviation: ")
        stDev = float(stDev_input)

        response = input("Are you specifying 'c' and seeking 'P' or specifying 'P' and seeking 'c'? (Enter 'c' or 'P'): ").strip().lower()

        if response == 'c':
            c = float(input("Enter the value of c: "))
            GT = input("Probability greater than c? (y/n): ").strip().lower() in ["y", "yes"]
            OneSided = input("One sided? (y/n): ").strip().lower() in ["y", "yes"]

            if OneSided:
                prob = Probability(GPDF, (mean, stDev), c, GT=GT)
                print(f"P(x{'>' if GT else '<'}{c:0.2f}|{mean:0.2f}, {stDev:0.2f}) = {prob:0.2f}")
            else:
                prob = Probability(GPDF, (mean, stDev), c, GT=True)
                prob = 1 - 2 * prob
                if GT:
                    print(f"P({mean - (c - mean)}>x>{mean + (c - mean)}|{mean:0.2f},{stDev:0.2f}) = {1 - prob:0.3f}")
                else:
                    print(f"P({mean - (c - mean)}<x<{mean + (c - mean)}|{mean:0.2f},{stDev:0.2f}) = {prob:0.3f}")

        elif response == 'p':
            P = float(input("Enter the desired probability (P): "))
            func = lambda c: Probability(GPDF, (mean, stDev), c, GT=True) - P
            c_estimate, iterations = Secant(func, mean - 5 * stDev, mean + 5 * stDev)  # Initial guesses for c
            print(f"The value of c that matches the probability {P} is approximately: {c_estimate:.4f}")

        again = input("Do you want to run the program again? (y/n): ").strip().lower() in ["y", "yes"]


if __name__ == "__main__":
    main()