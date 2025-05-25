import numpy as np

def sigmoid(z):
    """ Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return sigmoid(z) * (1 - sigmoid(z))

def mse(y_pred, y):
    """Mean Squared Error (MSE) cost function."""
    return np.mean((y_pred - y) ** 2)

def mse_prime(y_pred, y):
    """Derivative of the Mean Squared Error (MSE) cost function."""
    return 2 * (y_pred - y) / y.shape[1]
    
def softmax(z):
    """Softmax function: Converts input values into probabilities""" 
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def softmax_prime(z):
    """Derivative of softmax"""
    s = softmax(z)
    return s * (1 - s)

def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)

def relu_prime(z):
    """Derivative of the ReLU function."""
    return (z > 0).astype(float)

def leaky_relu(z, a=0.01):
    """Leaky ReLU activation function."""
    return np.maximum(a * z, z)

def leaky_relu_prime(z, a=0.01):
    """Derivative of the Leaky ReLU function."""
    dz = np.ones_like(z)
    dz[z <= 0] = a
    return dz

def cross_entropy(y_pred, y):
    """Cross-entropy loss function."""
    m = y.shape[1]
    return -np.sum(y * np.log(y_pred + 1e-9)) / m

def cross_entropy_prime(y_pred, y):
    """Derivative of the cross-entropy loss with respect to the softmax input."""
    return y_pred - y


class NeuralNetwork:
    def __init__(self, layer_sizes, activations, cost):
        """
        Initialize the neural network with random weights and biases.
        
        Args:
            layer_sizes (list): List of integers specifying the number of neurons in each layer.
                               Example: [input_size, hidden_size, ..., output_size]
            activations (list): List of tuples (activation_func, activation_prime) for each layer after the input.
            cost (tuple): Tuple (cost_func, cost_prime) for the cost function and its derivative.
        """
        self.layer_sizes = layer_sizes
        self.activations = activations  # Each element is a tuple (activation_func, activation_prime)
        self.cost = cost  # Tuple (cost_func, cost_prime)
        
        # Validate input
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Number of activations must match the number of layers (excluding input).")
        
        # Initialize weights and biases
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        
        
    def forward(self, X):
        # Reshape input to a column vector
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.a = [X]  # List to store activations for each layer
        self.z = []   # List to store weighted inputs for each layer

        for l, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = w @ self.a[-1] + b  # Weighted input
            self.z.append(z)
            
            a_func = self.activations[l][0]
            a = a_func(z)
            self.a.append(a)
        
        return self.a[-1]
        
    
    def backprop(self, y):
        """
        Compute gradients using backpropagation.
        
        Args:
            y: Target outputs of shape (output_size, batch_size).
        
        Returns:
            Tuple (grad_w, grad_b) containing gradients for each layer.
        """
        # Initialize gradient lists
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]
        
        cost_prime = self.cost[1] # Dericative of the cost function
        a_prime = self.activations[-1][1] # Activation function of the last layer
        
        # Compute output layer error (delta)
        delta = cost_prime(self.a[-1], y) * a_prime(self.z[-1]) 
        # Store output layer gradients
        grad_w[-1] = delta @ self.a[-2].T
        grad_b[-1] = delta
        
        # Backpropagate through hidden layers
        for l in reversed(range(len(self.weights) - 1)):
            a_prime = self.activations[l][1]
            # Calculate delta for current layer
            delta = (self.weights[l+1].T @ delta) * a_prime(self.z[l])
            # Compute gradients
            grad_w[l] = delta @ self.a[l].T
            grad_b[l] = delta
        
        return grad_w, grad_b
        
    
    def update_parameters(self, grad_w, grad_b, eta, batch_size):
        """Update weights and biases using averaged gradients."""
        for l in range(len(self.weights)):
            self.weights[l] -= eta * (grad_w[l] / batch_size)
            self.biases[l] -= eta * (grad_b[l] / batch_size)
            
    
    def train(self, x_train, y_train, epochs, batch_size, eta, results=False):
        """
        Train the network using mini-batch SGD.
        
        Args:
            x_train: Input data of shape (num_samples, input_size).
            y_train: Target outputs of shape (num_samples, output_size).
            epochs: Number of training epochs.
            batch_size: Size of mini-batches.
            eta: Learning rate.
        """
        num_samples = len(x_train)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = [x_train[i] for i in indices]
            y_shuffled = [y_train[i] for i in indices]

            epoch_outputs = []
            epoch_labels = []
            
            # Process mini-batches
            for i in range(0, num_samples, batch_size):
                # Get mini-batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Initialize gradient arrays
                grad_w = [np.zeros_like(w) for w in self.weights]
                grad_b = [np.zeros_like(b) for b in self.biases]
                
                # Accumulate gradients over batch
                for x, y_true in zip(X_batch, y_batch):
                    # Convert to column vectors
                    x = x.reshape(-1, 1)
                    y_true = y_true.reshape(-1, 1)

                    # Forward + Backprop
                    output = self.forward(x)
                    batch_grad_w, batch_grad_b = self.backprop(y_true)

                    if results:
                        epoch_outputs.append(output.flatten())  # Save output
                        epoch_labels.append(y_true.flatten())  # Save true label
                    
                    # Accumulate gradients
                    for l in range(len(self.weights)):
                        grad_w[l] += batch_grad_w[l]
                        grad_b[l] += batch_grad_b[l]
                
                # Update parameters
                self.update_parameters(grad_w, grad_b, eta, batch_size)
                
            
            # Print epoch results
            if results and epoch_outputs:
                epoch_outputs = np.array(epoch_outputs)
                epoch_labels = np.array(epoch_labels)
                
                loss = self.cost[0](epoch_outputs, epoch_labels)
                # Calculate accuracy
                predictions = np.argmax(epoch_outputs, axis=1)
                true_labels = np.argmax(epoch_labels, axis=1)
                accuracy = np.mean(predictions == true_labels)

                # Print results
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
                