import numpy as np


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


y = np.array([0, 1, 1, 1])


weights = np.random.rand(2)
bias = np.random.rand(1)

lr = 0.1

def step_function(x):
    return 1 if x >= 0 else 0


for epoch in range(10):  
    print(f"Epoch {epoch + 1}")
    for i in range(len(X)):
      
        z = np.dot(weights, X[i]) + bias

        
        output = step_function(z)

        # حساب الخطأ
        error = y[i] - output

       
        weights += lr * error * X[i]
        bias += lr * error

        print(f"Input: {X[i]}, Predicted: {output}, Target: {y[i]}, Weights: {weights}, Bias: {bias}")

print("\n     ///      ")
print("w", weights)
print(" (bias):", bias)





