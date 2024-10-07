import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic linear data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
# y = 4 + 3X + Gaussian noise
y = 4 + np.random.randn(100, 1)  

# Plot the data: scatter plot with title and labels

"""
# Scatter plot
plt.scatter(x=X,y=y)
# Title
plt.title("Test")
# xlabel
plt.xlabel("X")
# ylabel
plt.ylabel("Y")

plt.show()
"""

# Gradient Descent for Linear Regression
def compute_cost(X, y, theta):
    # define m as length of y
    m = len(y)
    # define predictions as 0 +1X
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        gradients = X.T.dot(X.dot(theta) - y) / m
        theta = theta - learning_rate * gradients
        cost_history[i] = compute_cost(X, y, theta)
        
    return theta, cost_history

# Add a bias column to X
X_b = np.c_[np.ones((100, 1)), X]  # Add a bias (intercept) term

# Initialize parameters and run gradient descent
theta = np.random.randn(2, 1)  # Random initialization
learning_rate = 0.01
iterations = 1000

theta_optimal, cost_history = gradient_descent(X_b,y,theta,learning_rate,iterations)

print("Optimal parameters:", theta_optimal)

# plot cost_history
plt.plot(cost_history)
# add title
plt.title("Cost History Function Plot")
# add x-label
plt.xlabel("X")
# add y-label
plt.ylabel("Y")
# show your figure
plt.show()


# Plot the original data and the fitted line
# plot the original data
plt.scatter(X, y, label="Data")
# plot the fitted line
plt.plot(X, X_b.dot(theta_optimal), color='red', label="Fitted Line")

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Fit using Gradient Descent')
plt.show()

