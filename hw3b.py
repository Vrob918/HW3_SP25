import scipy.stats as st
import pandas as pd
import numpy as np


def generate_t_table(degrees_of_freedom_list, f_z_values):
    table_data = {}

    for df in degrees_of_freedom_list:
        row_data = {}
        for f_z in f_z_values:
            if df == "inf":
                critical_value = st.t.ppf(f_z, np.inf)
            else:
                critical_value = st.t.ppf(f_z, df)

            row_data[f_z] = round(critical_value, 3)

        table_data[df] = row_data

    df_column_name = "Degrees of Freedom"
    f_z_column_names = [f"F(z) = {f_z}" for f_z in f_z_values]

    df = pd.DataFrame.from_dict(table_data, orient='index')
    df.index.name = df_column_name
    df = df.reset_index()
    df = df.rename_axis(None, axis=1)
    df = df.reindex(columns=[df_column_name] + f_z_column_names)
    return df


def calculate_probability(degrees_of_freedom, z_value):
    try:
        if degrees_of_freedom.lower() == "inf":
            df = np.inf
        else:
            df = int(degrees_of_freedom)

        z = float(z_value)

        if df < 1 and df is not np.inf:
            return "Error: Degrees of freedom must be a positive integer or 'inf'."

        probability = st.t.cdf(z, df)
        return probability

    except ValueError:
        return "Error: Invalid input. Degrees of freedom must be an integer or 'inf', and z must be a number."


if __name__ == "__main__":
    degrees_of_freedom = list(range(1, 31)) + [40, 60, 120, "inf"]
    f_z_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999]

    while True:
        try:
            df_input = input("Enter the degrees of freedom: ")
            z_input = input("Enter the z-value: ")

            result = calculate_probability(df_input, z_input)

            if isinstance(result, str):
                print(result)
            else:
                print(f"The probability for df={df_input} and z={z_input} is: {result:.4f}")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")