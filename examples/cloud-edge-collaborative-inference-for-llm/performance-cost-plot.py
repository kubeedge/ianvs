import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

colors = plt.cm.Paired.colors  # Set1 调色板
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors)

# a sigmoid function to fit non-oracle models' performance vs cost
def sigmoid_fit(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def plot_accuracy_cost(models, costs, accuracy, non_oracle_costs, non_oracle_accuracy):
    # Fit the sigmoid model
    params_sigmoid, _ = curve_fit(sigmoid_fit, non_oracle_costs, non_oracle_accuracy, p0=[100, 1, 0.2])

    # Generate points for the sigmoid fitted curve
    curve_x_sigmoid = np.linspace(min(non_oracle_costs), max(non_oracle_costs), 100)
    curve_y_sigmoid = sigmoid_fit(curve_x_sigmoid, *params_sigmoid)

    plt.figure(figsize=(10, 6))

    # Plot all models
    for i in range(len(models)):
        if "Oracle" in models[i]:
            marker = '^'  # Triangle marker for Oracle models
        else:
            marker = 'o'  # Circle marker for non-Oracle models
        plt.scatter(costs[i], accuracy[i], label=models[i], marker=marker)

    # Plot the sigmoid fitted curve
    plt.plot(curve_x_sigmoid, curve_y_sigmoid, 'gray', linestyle='dashed')  # Gray dashed line for the curve

    plt.title('Model Performance vs Cost')
    plt.xlabel('Cost($/M token)')
    plt.ylabel('Accuracy (%)')
    plt.legend(title='Model Name')
    plt.grid(True)
    plt.savefig('model_performance_sigmoid_fitted_curve.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    models = [
        "Oracle-Qwen2.5-7b-instruct + gpt-4o-mini",
        "Oracle-Qwen2.5-1.5b-instruct + gpt-4o-mini",
        "Oracle-Qwen2.5-3b-instruct + gpt-4o-mini",
        "gpt-4o-mini",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-1.5B-Instruct"
    ]
    # The Oracle Routed Model's cost is an average weighted by the Edge Ratio between edge model costs and cloud model costs.
    # The edge model’s cost is estimated based on its parameter size.
    costs = [0.16, 0.18, 0.17, 0.60, 0.10, 0.08, 0.05]
    accuracy = [84.22, 82.75, 82.22, 75.99, 71.84, 60.3, 58.35]

    # Non Oracle Models: gpt-4o-mini, Qwen2.5-7B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct
    non_oracle_costs = costs[-4:] # Costs in $/M token
    non_oracle_accuracy = accuracy[-4:] # Accuracies in %

    plot_accuracy_cost(models, costs, accuracy, non_oracle_costs, non_oracle_accuracy)