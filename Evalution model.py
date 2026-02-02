import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure save directory exists
save_dir = '/mnt/data'
os.makedirs(save_dir, exist_ok=True)

rng = np.random.default_rng(0)

# XOR data
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

# network sizes
in_dim = 2
hid_dim = 4
out_dim = 1

# initialize weights
W1 = rng.normal(0, 1, (in_dim, hid_dim)) * 0.5
b1 = np.zeros(hid_dim)
W2 = rng.normal(0, 1, (hid_dim, out_dim)) * 0.5
b2 = np.zeros(out_dim)

# activations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# training
lr = 0.8
epochs = 5000
loss_history = []

for epoch in range(epochs):
    # forward
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # loss (MSE)
    loss = np.mean((a2 - y) ** 2)
    loss_history.append(loss)

    # backprop (batch)
    grad_a2 = 2 * (a2 - y) / y.size
    grad_z2 = grad_a2 * (a2 * (1 - a2))
    grad_W2 = np.dot(a1.T, grad_z2)
    grad_b2 = np.sum(grad_z2, axis=0)

    grad_a1 = np.dot(grad_z2, W2.T)
    grad_z1 = grad_a1 * (a1 * (1 - a1))
    grad_W1 = np.dot(X.T, grad_z1)
    grad_b1 = np.sum(grad_z1, axis=0)

    # gradient step
    W2 -= lr * grad_W2
    b2 -= lr * grad_b2
    W1 -= lr * grad_W1
    b1 -= lr * grad_b1

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, loss = {loss:.6f}')

# final eval
z1 = np.dot(X, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)
preds = (a2 > 0.5).astype(int)
acc = np.mean(preds.flatten() == y.flatten())
print('\nFinal loss:', loss_history[-1])
print('Predictions:\n', a2)
print('Binary preds:\n', preds)
print('Accuracy:', acc)

# save a quick loss plot
plt.plot(loss_history)
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'numpy_xor_loss.png'))
plt.show()
