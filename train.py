import numpy as np
from Neural-network import Neural-network
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4, 5], dtype = float)
y = np.array([2, 4, 6, 8, 10], dtype = float)
model = Neural_network(lr=0.01, epochs=1000)
model.fit(x, y)
predictions = model.predict(x)
print("Predictions:", predictions)
plt.scatter(X, y, label="Real Data")
plt.plot(X, predictions, label="Model", linestyle="--")
plt.legend()
