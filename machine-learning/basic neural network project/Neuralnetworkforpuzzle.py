
import numpy as np 

# data: 0 if same, 1 if different
X = np.array([[0,0],[1,1],[0,1],[1,0]])
y = np.array([[0],[0],[1],[1]])

# 2 inputs, 2 hidden, 1 output
rng = np.random.default_rng(1)

W1 = rng.normal(size=(2,2))
W2 = rng.normal(size=(2,1))

b1 = np.zeros(2)
b2 = np.zeros(1)

# activation functions 

def relu(z):
    return np.maximum(0,z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# learning rate and training
lr = 1.0

print("step loss")

for step in range(1,1501):

    # the forward pass
    z1 = X @ W1 + b1
    h = relu(z1)

    z2 = h @ W2 + b2
    y_hat = sigmoid(z2)

    # loss: how wrong we are
    eps = 1e-8
    loss = -np.mean(
        y * np.log(y_hat + eps) +
        (1-y) * np.log(1-y_hat + eps)
    )

    # backward pass: gradients
    d2 = (y_hat - y) / 4
    d1 = (d2 @ W2.T) * (z1 > 0)

    # update a small step downhill
    W2 -= lr * (h.T @ d2)
    b2 -= lr * d2.sum(axis=0)

    W1 -= lr * (X.T @ d1)
    b1 -= lr * d1.sum(axis=0)

    if step == 1 or step % 150 == 0:
        print(f"{step:4d}    loss: {loss:.3f}")

print("\nPredictions:")
print(y_hat.round(2))

print("\nFinal classes:")
print((y_hat >= 0.5).astype(int))

print("\nActual outputs:")
print(y)
