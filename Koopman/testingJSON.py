import numpy as np
import json


mat_name = "koopman_linear_ss"
dt = 0.1

# Example 1
A = np.array([[1.0, 0.1], [0.0, 1.0]])  # 2 x 2 -> 2 rows = 2 states
B = np.array([[0.0], [0.1]])            # 2 x 1 -> 2 rows = 2 states, 1 column = 1 control input

# # Example 2
# A = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 1.0]])  # 3 x 3 -> 3 rows = 3 states
# B = np.array([[0.0, 0.5], [0.1, 0.7], [0.0, 0.2]])            # 3 x 2 -> 3 rows = 3 states, 2 columns = 2 control inputs

data = {
    "model_name": mat_name,
    "dt": dt,
    "A": A.tolist(),
    "B": B.tolist()
}

with open("Koopman/model_results.json", "w") as f:
    json.dump(data, f, indent=2)
print("Saved model results to model_results.json")


with open("Koopman/model_results.json", "r") as f:
    data_again = json.load(f)

model_name = data_again["model_name"]
dt = float(data_again["dt"])
A = np.array(data_again["A"], dtype=float)
B = np.array(data_again["B"], dtype=float)

print(f"Model Name: {model_name}")
print(f"dt: {dt}")
print(f"A:\n{A}")
print(f"B:\n{B}")