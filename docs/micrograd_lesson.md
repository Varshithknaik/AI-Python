# 🛠️ Micrograd: Building an Autograd Engine from Scratch

Welcome to Part 1 of the Zero-to-Hero curriculum. In this module, you are going to build `micrograd`—a tiny scalar-valued autograd engine.

## 🎯 The Goal
Understand the mathematical heart of deep learning: **Backpropagation** and **Computational Graphs**.

You will build the core mechanism that powers PyTorch, but instead of complex tensors, you will work with simple scalar values (single numbers). This strips away the complexity of multi-dimensional arrays and lets you see exactly how the chain rule from calculus flows backward through a network to update weights.

---

## 🏗️ Project Architecture

You will build this project in Python across three distinct phases. 

### Phase 1: The Core Engine (`engine.py`)
This is the heart of the system. You will create a `Value` class that wraps standard Python floats and tracks their history.

**Key Requirements:**
1. **State**: The `Value` object must store its `data` (the float) and its `grad` (the derivative/gradient, initially 0.0).
2. **Graph Tracking**: It needs to track its `_prev` (the children `Value` objects that produced it) and `_op` (the mathematical operation, e.g., `+`, `*`).
3. **Operations**: Implement Python dunder methods (like `__add__`, `__mul__`, `__pow__`) so you can add and multiply `Value` objects just like normal numbers. Also, add activation functions like `relu()` or `tanh()`.
4. **The Magic (`backward()`)**: Implement a topological sort to arrange the graph, and write the logic to apply the chain rule locally at each node to compute the `grad`.

### Phase 2: The Neural Network (`nn.py`)
Once your engine can handle derivatives for any mathematical expression, you will build the neural network abstractions on top of it.

**Key Requirements:**
1. **`Module`**: A base class that has a `zero_grad()` function.
2. **`Neuron`**: A single neuron. It holds weights (`w`) and a bias (`b`), all as `Value` objects. Its output is: `activation(sum(w * x) + b)`.
3. **`Layer`**: A list of `Neuron` objects.
4. **`MLP` (Multi-Layer Perceptron)**: A list of `Layer` objects.

### Phase 3: The Training Loop (`train.py`)
Now you put it all together to solve a real problem (like binary classification).

**The ReAct / Event Loop of ML:**
1. **Forward Pass**: Pass inputs through your MLP to get predictions.
2. **Loss**: Calculate how wrong the predictions are (e.g., Mean Squared Error).
3. **Zero Grad**: Clear out the old gradients (`model.zero_grad()`).
4. **Backward Pass**: Run `loss.backward()` to calculate new gradients.
5. **Update (Gradient Descent)**: Loop through all parameters in the model and adjust them: `p.data += -learning_rate * p.grad`.

---

## 📚 Study Resources
- **Reference Video**: [The spell book: Andrej Karpathy's Micrograd Video](https://www.youtube.com/watch?v=VMj-3S1tku0)
- **Reference Code**: [karpathy/micrograd](https://github.com/karpathy/micrograd)

## 🚀 Getting Started

Since this workspace is set to **Tutor Mode**, I will not write the code for you unless you ask. 

**Your first step:**
Create a folder named `part-1-micrograd` and a file inside it called `engine.py`. Start by defining the `Value` class and its `__init__` constructor. 

Whenever you get stuck, encounter a bug, or want to understand *why* the math works the way it does, just ask! I'm here to review your code and explain the concepts.
