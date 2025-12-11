"""
Multi-Layer Perceptron for MNIST Digit Recognition
Complete implementation with GUI in a single file
"""

import numpy as np
import pickle
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading


# ============================================================================
# PART 1: NEURAL NETWORK CLASSES
# ============================================================================

class Perceptron:
    """
    A single perceptron (neuron) in the network
    """
    def __init__(self, num_inputs):
        # Initialize random weights and bias
        self.weights = np.random.randn(num_inputs) * 0.01
        self.bias = np.random.randn() * 0.01
    
    def forward(self, inputs):
        """Calculate z = w1*x1 + w2*x2 + ... + b"""
        self.inputs = inputs
        self.z = np.dot(self.weights, inputs) + self.bias
        return self.z


class Layer:
    """
    A layer of perceptrons (neurons)
    """
    def __init__(self, num_inputs, num_neurons):
        # Create multiple perceptrons in this layer
        self.perceptrons = [Perceptron(num_inputs) for _ in range(num_neurons)]
        self.num_neurons = num_neurons
    
    def forward(self, inputs):
        """Pass inputs through all perceptrons in this layer"""
        self.inputs = inputs
        # Get output from each perceptron
        self.z_values = np.array([p.forward(inputs) for p in self.perceptrons])
        # Apply activation function
        self.activations = self.sigmoid(self.z_values)
        return self.activations
    
    def sigmoid(self, z):
        """Activation function for the layer"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def get_weights(self):
        """Get all weights and biases from this layer"""
        weights = np.array([p.weights for p in self.perceptrons])
        biases = np.array([p.bias for p in self.perceptrons])
        return weights, biases
    
    def set_weights(self, weights, biases):
        """Set weights and biases for this layer"""
        for i, perceptron in enumerate(self.perceptrons):
            perceptron.weights = weights[i]
            perceptron.bias = biases[i]


class NeuralNetwork:
    """
    Multi-Layer Perceptron Neural Network
    Architecture: Input -> Hidden Layer -> Output Layer
    """
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        print(f"Creating Neural Network:")
        print(f"  Input layer: {input_size} neurons (28x28 pixels)")
        print(f"  Hidden layer: {hidden_size} neurons")
        print(f"  Output layer: {output_size} neurons (digits 0-9)")
        
        # Create layers
        self.hidden_layer = Layer(input_size, hidden_size)
        self.output_layer = Layer(hidden_size, output_size)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    def forward(self, inputs):
        """
        Forward pass: Input -> Hidden -> Output
        """
        # Flatten image if needed
        if inputs.shape[0] != self.input_size:
            inputs = inputs.flatten()
        
        # Normalize inputs to 0-1 range
        inputs = inputs / 255.0 if inputs.max() > 1 else inputs
        
        # Pass through hidden layer
        self.hidden_output = self.hidden_layer.forward(inputs)
        
        # Pass through output layer
        self.output = self.output_layer.forward(self.hidden_output)
        
        return self.output
    
    def predict(self, inputs):
        """Make a prediction and return the digit"""
        output = self.forward(inputs)
        predicted_digit = np.argmax(output)
        confidence = output[predicted_digit]
        return predicted_digit, confidence, output
    
    def backward(self, inputs, target, learning_rate=0.01):
        """
        Backpropagation: Calculate gradients and update weights
        """
        # Forward pass
        output = self.forward(inputs)
        
        # Create target vector (one-hot encoding)
        target_vector = np.zeros(self.output_size)
        target_vector[target] = 1
        
        # Calculate error
        error = target_vector - output
        
        # Output layer gradients
        output_delta = error * output * (1 - output)
        
        # Hidden layer gradients
        hidden_error = np.dot(
            np.array([p.weights for p in self.output_layer.perceptrons]).T,
            output_delta
        )
        hidden_delta = hidden_error * self.hidden_output * (1 - self.hidden_output)
        
        # Update output layer weights
        for i, perceptron in enumerate(self.output_layer.perceptrons):
            perceptron.weights += learning_rate * output_delta[i] * self.hidden_output
            perceptron.bias += learning_rate * output_delta[i]
        
        # Update hidden layer weights
        for i, perceptron in enumerate(self.hidden_layer.perceptrons):
            perceptron.weights += learning_rate * hidden_delta[i] * self.hidden_layer.inputs
            perceptron.bias += learning_rate * hidden_delta[i]
        
        # Calculate loss
        loss = np.mean(error ** 2)
        
        # Check if prediction is correct
        predicted_digit = np.argmax(output)
        
        return loss, predicted_digit == target
    
    def train(self, images, labels, epochs=1, learning_rate=0.01, callback=None):
        """
        Train the network
        callback: function to call after each sample for GUI updates
        """
        total_samples = len(images)
        
        for epoch in range(epochs):
            correct = 0
            total_loss = 0
            
            for i, (image, label) in enumerate(zip(images, labels)):
                # Train on this sample
                loss, is_correct = self.backward(image, label, learning_rate)
                
                total_loss += loss
                if is_correct:
                    correct += 1
                
                # Call callback for GUI update
                if callback:
                    accuracy = (correct / (i + 1)) * 100
                    avg_loss = total_loss / (i + 1)
                    callback(i + 1, total_samples, accuracy, avg_loss, image, label)
            
            accuracy = (correct / total_samples) * 100
            avg_loss = total_loss / total_samples
            
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Correct: {correct}/{total_samples}")
    
    def save_weights(self, filename='weights.pkl'):
        """Save network weights to file"""
        weights_data = {
            'hidden_weights': self.hidden_layer.get_weights(),
            'output_weights': self.output_layer.get_weights(),
            'architecture': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(weights_data, f)
        
        print(f"Weights saved to {filename}")
    
    def load_weights(self, filename='weights.pkl'):
        """Load network weights from file"""
        if not os.path.exists(filename):
            print(f"No weights file found at {filename}")
            return False
        
        with open(filename, 'rb') as f:
            weights_data = pickle.load(f)
        
        # Verify architecture matches
        arch = weights_data['architecture']
        if (arch['input_size'] != self.input_size or 
            arch['hidden_size'] != self.hidden_size or 
            arch['output_size'] != self.output_size):
            print("Warning: Saved weights don't match network architecture!")
            return False
        
        # Load weights
        hidden_w, hidden_b = weights_data['hidden_weights']
        output_w, output_b = weights_data['output_weights']
        
        self.hidden_layer.set_weights(hidden_w, hidden_b)
        self.output_layer.set_weights(output_w, output_b)
        
        print(f"Weights loaded from {filename}")
        return True


def load_mnist_sample():
    """
    Load a small sample of MNIST data for demonstration
    """
    try:
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # Use a subset for faster training
        X_train = X_train[:1000]  # First 1000 images
        y_train = y_train[:1000]
        
        X_test = X_test[:200]  # First 200 test images
        y_test = y_test[:200]
        
        print(f"Loaded MNIST dataset:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Image shape: {X_train[0].shape}")
        
        return (X_train, y_train), (X_test, y_test)
    
    except ImportError:
        print("TensorFlow not installed. Install with: pip install tensorflow")
        return None, None


# ============================================================================
# PART 2: GUI APPLICATION
# ============================================================================

class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        self.root.geometry("700x350")
        self.root.configure(bg='#f5f5dc')
        
        # Neural network
        self.nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
        
        # Try to load existing weights
        if self.nn.load_weights('weights.pkl'):
            self.status_text = "Loaded existing weights"
        else:
            self.status_text = "Initialized with random weights"
        
        # Training data
        self.training_data = None
        self.is_training = False
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Create the GUI layout - very simple"""
        
        # Main container with simple beige background
        main_frame = tk.Frame(self.root, bg='#f5f5dc')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left column - Image display
        left_frame = tk.Frame(main_frame, bg='#f5f5dc')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Image canvas
        self.canvas = tk.Canvas(
            left_frame,
            width=280,
            height=280,
            bg='white',
            highlightthickness=1,
            highlightbackground='black'
        )
        self.canvas.pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(left_frame, bg='#f5f5dc')
        button_frame.pack(pady=5)
        
        self.train_button = tk.Button(
            button_frame,
            text="Train",
            command=self.start_training,
            font=("Arial", 11),
            width=12
        )
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.upload_button = tk.Button(
            button_frame,
            text="Upload",
            command=self.upload_image,
            font=("Arial", 11),
            width=12
        )
        self.upload_button.pack(side=tk.LEFT, padx=5)
        
        # Right column - Information display
        right_frame = tk.Frame(main_frame, bg='#f5f5dc')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Status display area
        self.status_display = tk.Text(
            right_frame,
            font=("Courier", 9),
            bg='white',
            fg='black',
            height=20,
            width=35,
            relief=tk.SOLID,
            borderwidth=1
        )
        self.status_display.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Initial status
        self.update_status(f"{self.status_text}\n\n")
        self.update_status("Architecture:\n")
        self.update_status("Input:  784 neurons\n")
        self.update_status("Hidden: 128 neurons\n")
        self.update_status("Output: 10 neurons\n\n")
        self.update_status("Ready!\n")
    
    def update_status(self, text, clear=False):
        """Update the status display"""
        if clear:
            self.status_display.delete(1.0, tk.END)
        self.status_display.insert(tk.END, text)
        self.status_display.see(tk.END)
        self.root.update()
    
    def display_image(self, image_array):
        """Display an image on the canvas"""
        # Ensure it's 28x28
        if image_array.shape != (28, 28):
            image_array = image_array.reshape(28, 28)
        
        # Convert to PIL Image and resize for display
        img = Image.fromarray(image_array.astype('uint8'))
        img = img.resize((280, 280), Image.NEAREST)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(img)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(140, 140, image=self.photo)
    
    def start_training(self):
        """Start training in a separate thread"""
        if self.is_training:
            messagebox.showinfo("Training", "Training already in progress!")
            return
        
        # Load data if not already loaded
        if self.training_data is None:
            self.update_status("\nLoading MNIST dataset...\n", clear=True)
            train_data, test_data = load_mnist_sample()
            
            if train_data is None or train_data[0] is None:
                messagebox.showerror(
                    "Error",
                    "Could not load MNIST data.\nInstall TensorFlow: pip install tensorflow"
                )
                return
            
            self.training_data = train_data
            self.test_data = test_data
            self.update_status("Dataset loaded!\n\n")
        
        # Start training in separate thread
        self.is_training = True
        self.train_button.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self.train_network)
        thread.daemon = True
        thread.start()
    
    def train_network(self):
        """Train the neural network"""
        X_train, y_train = self.training_data
        
        self.update_status("TRAINING\n")
        self.update_status("="*30 + "\n\n")
        
        def training_callback(sample, total, accuracy, loss, image, label):
            """Called after each training sample"""
            if sample % 50 == 0 or sample == total:  # Update every 50 samples
                self.display_image(image)
                
                status = f"{sample}/{total}\n"
                status += f"Label: {label}\n"
                status += f"Accuracy: {accuracy:.1f}%\n"
                status += f"Loss: {loss:.4f}\n"
                status += "-" * 30 + "\n"
                
                self.update_status(status)
        
        # Train
        self.nn.train(
            X_train,
            y_train,
            epochs=1,
            learning_rate=0.1,
            callback=training_callback
        )
        
        # Save weights
        self.nn.save_weights('weights.pkl')
        
        self.update_status("\n" + "="*30 + "\n")
        self.update_status("DONE\n")
        self.update_status("Weights saved\n")
        self.update_status("="*30 + "\n")
        
        self.is_training = False
        self.train_button.config(state=tk.NORMAL)
    
    def upload_image(self):
        """Upload and test an image"""
        filename = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not filename:
            return
        
        try:
            # Load image
            img = Image.open(filename).convert('L')  # Convert to grayscale
            
            # Resize to 28x28
            img = img.resize((28, 28), Image.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Invert if needed (MNIST has white digits on black background)
            if img_array.mean() > 127:
                img_array = 255 - img_array
            
            # Display image
            self.display_image(img_array)
            
            # Make prediction
            predicted, confidence, output = self.nn.predict(img_array)
            
            # Display results
            self.update_status("\n" + "="*30 + "\n", clear=True)
            self.update_status("RESULT\n")
            self.update_status("="*30 + "\n\n")
            self.update_status(f"Digit: {predicted}\n")
            self.update_status(f"Confidence: {confidence*100:.1f}%\n\n")
            self.update_status("Probabilities:\n")
            
            for digit in range(10):
                prob = output[digit] * 100
                bar = "#" * int(prob / 5)
                self.update_status(f"{digit}: {prob:4.1f}% {bar}\n")
            
            self.update_status("\n" + "="*30 + "\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not process image:\n{str(e)}")


# ============================================================================
# PART 3: MAIN PROGRAM
# ============================================================================

def main():
    root = tk.Tk()
    app = DigitRecognizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
