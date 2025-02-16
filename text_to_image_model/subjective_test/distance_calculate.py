import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd


# Define the function f(a, k)
def f(a, k=2):
    # Ensure a is within (0, 1) to avoid log of zero or negative numbers
    if a <= 0 or a >= 1:
        return np.inf
    return - (1 / np.log(k)) * (
        a * np.log(a) + (1 - a) * np.log((1 - a) / (k - 1))
    )


# Define squared distance function
def distance_squared(x, x0, y0, k=2):
    return (x - x0) ** 2 + (f(x, k) - y0) ** 2


# Find the minimum distance from a point to the curve
def find_min_distance(x0, y0, k=2):
    result = minimize_scalar(
        distance_squared,
        bounds=(1e-8, 1 - 1e-8),
        args=(x0, y0, k),
        method='bounded'
    )
    x_min = result.x
    y_min = f(x_min, k)
    distance = np.sqrt(distance_squared(x_min, x0, y0, k))
    return x_min, y_min, distance


# Read the CSV file
input_file = './Test_Results/t2i_subjective_test_result.csv'
data = pd.read_csv(input_file)

# Define a list to store results
results = []

# Iterate through each row to compute distance
for _, row in data.iterrows():
    x0 = row['Accuracy']
    y0 = row['Entropy Ratio']
    attribute = row['Attribute']
    
    # Determine k value based on Attribute
    k = 2 if attribute == 'gender' else 4
    
    # Compute the minimum distance
    x_min, y_min, distance = find_min_distance(x0, y0, k)
    
    # Multiply distance by 100 and round to two decimal places
    results.append({
        'Model': row['Model'],
        'Attribute': attribute,
        'Accuracy': round(x0, 2),
        'Entropy Ratio': round(y0, 2),
        'Min X': round(x_min, 2),
        'Min Y': round(y_min, 2),
        'Distance': round(distance * 100, 2)  # Modified part: Multiply distance by 100 and round to two decimal places
    })

# Save results to a new CSV file
output_file = './Test_Results/dist.csv'
output_df = pd.DataFrame(results)
output_df.to_csv(output_file, index=False)

print(f"Results have been saved to {output_file}")
