"""
Simple Perceptron - Learning y = w*x + b
Uses external CSV files for training and testing
"""

import csv
import random

# ============================================
# STEP 1: Initialize the perceptron
# ============================================
print("STEP 1: Initialize Perceptron")
print("="*50)

# Start with random weight and bias
w = random.uniform(-1, 1)  # weight (slope m)
b = random.uniform(-1, 1)  # bias (intercept c)

print(f"Starting weight (w): {w:.4f}")
print(f"Starting bias (b): {b:.4f}")
print(f"Starting equation: y = {w:.4f}*x + {b:.4f}")
print()

# ============================================
# STEP 2: Read training data from CSV
# ============================================
print("STEP 2: Read Training Data")
print("="*50)

# Read the training data
x_values = []
y_values = []

with open('training_data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        x_values.append(float(row[0]))
        y_values.append(float(row[1]))

print(f"Loaded {len(x_values)} training examples")
print(f"First few examples:")
for i in range(5):
    print(f"  x={x_values[i]}, y={y_values[i]}")
print(f"  ...")
print()

# ============================================
# STEP 3: Train the perceptron
# ============================================
print("STEP 3: Training")
print("="*50)

learning_rate = 0.0002
epochs = 500

for epoch in range(epochs):
    
    # FORWARD PASS - Make predictions
    predictions = []
    for x in x_values:
        y_pred = w * x + b  # This is our perceptron equation!
        predictions.append(y_pred)
    
    # Calculate error
    total_error = 0
    for i in range(len(y_values)):
        error = y_values[i] - predictions[i]
        total_error += error * error
    
    loss = total_error / len(y_values)  # Mean Squared Error
    
    # BACKWARD PASS - Update weights
    # Calculate how much to change w and b
    dw = 0
    db = 0
    for i in range(len(x_values)):
        error = y_values[i] - predictions[i]
        dw += -2 * error * x_values[i]
        db += -2 * error
    
    dw = dw / len(x_values)
    db = db / len(x_values)
    
    # Update weight and bias
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Loss: {loss:.4f}")
        print(f"  Weight (w): {w:.4f}")
        print(f"  Bias (b): {b:.4f}")
        print(f"  Equation: y = {w:.4f}*x + {b:.4f}")

print()
print("Training Complete!")
print(f"Final equation: y = {w:.4f}*x + {b:.4f}")
print(f"(True equation: y = 2.0*x + 3.0)")
print()

# ============================================
# STEP 4: Test predictions
# ============================================
print("STEP 4: Testing Predictions on New Data")
print("="*50)

# Read the testing data
test_x = []
test_y = []

with open('testing_data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        test_x.append(float(row[0]))
        test_y.append(float(row[1]))

print(f"Testing on {len(test_x)} new examples:")
print()

total_test_error = 0
for i in range(len(test_x)):
    x = test_x[i]
    y_true = test_y[i]
    y_pred = w * x + b  # Use our learned equation
    error = abs(y_true - y_pred)
    total_test_error += error
    
    print(f"  x = {x}")
    print(f"    Expected y: {y_true}")
    print(f"    Predicted y: {y_pred:.4f}")
    print(f"    Error: {error:.4f}")
    print()

average_error = total_test_error / len(test_x)
print("="*50)
print(f"Average error on test data: {average_error:.4f}")
print("Done! The perceptron learned the pattern!")
