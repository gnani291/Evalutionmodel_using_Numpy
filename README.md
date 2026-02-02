Key Description

Dataset Creation (XOR Problem)
The XOR input dataset is defined with four input combinations and their corresponding output labels. XOR is a non-linear problem, so it cannot be solved using a single-layer perceptron, making it a good example for demonstrating multi-layer neural networks.

Neural Network Architecture
The network contains 2 input neurons, 4 hidden neurons, and 1 output neuron. Weights are initialized randomly and biases are initialized to zero. The sigmoid activation function is used in both hidden and output layers.

Forward Propagation
During each training epoch, input data passes through the hidden layer and then to the output layer. The sigmoid function converts linear outputs into probabilities between 0 and 1.

Loss Calculation and Backpropagation
The model uses Mean Squared Error (MSE) as the loss function. Gradients are computed manually using the chain rule, and weights and biases are updated using batch gradient descent with a learning rate of 0.8.

Training and Evaluation
The network trains for 5000 epochs. After training, predictions are generated and converted into binary outputs. The code calculates final loss, prints predicted values, and measures classification accuracy.

Visualization
The training loss over epochs is plotted using Matplotlib and saved as an image file (numpy_xor_loss.png). This helps visualize how the network learns and how the error decreases during training.


Summary (Short)
This program demonstrates how a basic neural network can be implemented using NumPy to solve the XOR problem by performing forward propagation, backpropagation, weight updates, and training visualization without relying on deep learning frameworks.
