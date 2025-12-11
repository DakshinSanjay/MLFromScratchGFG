import numpy as np
import pickle
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog


class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(784, 128) * 0.01
        self.b1 = np.zeros((1, 128))
        self.W2 = np.random.randn(128, 10) * 0.01
        self.b2 = np.zeros((1, 10))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, X):
        X = X.reshape(1, -1) / 255.0
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        return a2, a1, X
    
    def train(self, X, y, epochs=3, lr=0.5):
        for epoch in range(epochs):
            correct = 0
            for i in range(len(X)):
                xi = X[i].reshape(1, -1) / 255.0
                yi = y[i]
                
                z1 = np.dot(xi, self.W1) + self.b1
                a1 = self.sigmoid(z1)
                z2 = np.dot(a1, self.W2) + self.b2
                a2 = self.sigmoid(z2)
                
                target = np.zeros((1, 10))
                target[0, yi] = 1
                
                if np.argmax(a2) == yi:
                    correct += 1
                
                error2 = target - a2
                delta2 = error2 * a2 * (1 - a2)
                error1 = np.dot(delta2, self.W2.T)
                delta1 = error1 * a1 * (1 - a1)
                
                self.W2 += lr * np.dot(a1.T, delta2)
                self.b2 += lr * delta2
                self.W1 += lr * np.dot(xi.T, delta1)
                self.b1 += lr * delta1
                
                if i % 100 == 0:
                    print(f"Epoch {epoch+1}, Sample {i}/{len(X)}, Accuracy: {100*correct/(i+1):.1f}%")
            
            print(f"Epoch {epoch+1} done - Accuracy: {100*correct/len(X):.1f}%")
    
    def predict(self, X):
        output, _, _ = self.forward(X)
        return np.argmax(output), output.flatten()
    
    def save(self):
        with open('weights.pkl', 'wb') as f:
            pickle.dump({'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}, f)
    
    def load(self):
        with open('weights.pkl', 'rb') as f:
            data = pickle.load(f)
            self.W1, self.b1, self.W2, self.b2 = data['W1'], data['b1'], data['W2'], data['b2']


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MNIST")
        self.root.geometry("600x400")
        
        self.nn = NeuralNetwork()
        
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack(pady=20)
        
        tk.Button(self.root, text="Train", command=self.train, width=15).pack(side=tk.LEFT, padx=20)
        tk.Button(self.root, text="Upload", command=self.upload, width=15).pack(side=tk.RIGHT, padx=20)
        
        self.label = tk.Label(self.root, text="Ready", font=("Arial", 14))
        self.label.pack(pady=20)
        
        self.root.mainloop()
    
    def train(self):
        from tensorflow.keras.datasets import mnist
        (X, y), _ = mnist.load_data()
        X, y = X[:3000], y[:3000]
        
        self.label.config(text="Training...")
        self.root.update()
        
        self.nn.train(X, y)
        self.nn.save()
        
        self.label.config(text="Training done!")
    
    def upload(self):
        file = filedialog.askopenfilename()
        
        img = Image.open(file).convert('L').resize((28, 28))
        arr = np.array(img)
        
        if arr.mean() > 127:
            arr = 255 - arr
        
        display = Image.fromarray(arr).resize((280, 280), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.create_image(140, 140, image=self.photo)
        
        pred, probs = self.nn.predict(arr)
        
        self.label.config(text=f"Prediction: {pred} ({probs[pred]*100:.1f}%)")


GUI()
