# Implementation of an Automatic differentiation engine: from scratch


![Shinchan](shinchan.png)

## Introduction

This repos is a hands-on implementation of an automatic differentiation (autodiff) engine, inspired by karpathy PyTorch. The goal i had in mind was to learn how actually backprop is being done. For keeping thing simple, ive used scalar functions in the repo. 

## What’s Inside

- **engine.ipynb**: The main notebook where the autodiff engine is implemented and tested. It contains the core `value` class, forward and backward passes, and comparisons with PyTorch.
- **main.py**: (If present) Entry point for running the autodiff engine outside the notebook.
- **exp.ipynb**: (If present) Additional random explorations around some functions.
- **pyproject.toml**: Project dependencies.
- **README.md**: This file.

## How It Works

### The `value` Class

At the heart of the engine is the `value` class. Each instance represents a scalar value in the computation, along with its gradient and the operation that produced it.

- **Data**: The actual scalar value.
- **Grad**: The gradient of the output with respect to this value.
- **Prev**: The set of previous `value` objects (parents) that were used to compute this value.
- **Op**: The operation (e.g., '+', '*', 'exp', 'tanh').

### Operator Overloading

The class overloads basic arithmetic operators (`+`, `-`, `*`, `/`, `**`) so you can build up complex expressions naturally. Each operation creates a new `value` object, records its parents, and sets up a local gradient function.

### Backward Pass

To compute gradients, we traverse the computation graph in reverse (topological order), calling each node’s `_backward` function. This accumulates gradients for each variable, just like backpropagation in neural networks.

### Example Usage

You can create variables, combine them with arithmetic, and call `.backward()` to compute gradients:

```python
a = value(0.4)
b = value(0.8)
e = a * b
c = value(1.0)
d = c + e
f = value(-1.5)
l = d * f
o = l.tanh()
o.backward()
print(a.grad, b.grad, ...)
```

### Comparison with PyTorch

The notebook also includes a PyTorch implementation of the same computation, showing that our engine produces the same gradients as a mature library.

---

## Why: The Problem and the Solution

### The Problem

Suppose you want to train a neural network, or optimize any function with respect to its inputs. You need to compute derivatives—gradients—of outputs with respect to inputs, no matter how complex the function. Doing this by hand is tedious and error-prone, especially as the function grows in complexity.

### The Solution: Build an Engine from First Principles

#### 1. Representing Values and Their History

The first challenge is to keep track of not just the value of each computation, but also how it was computed. This is why every `value` object stores its parents (`prev`) and the operation (`op`) that produced it. This forms a computation graph—a map of how each value depends on others.

#### 2. Operator Overloading

To make the engine intuitive, I overloaded Python’s arithmetic operators. This lets you write mathematical expressions as you normally would, while the engine automatically builds the computation graph behind the scenes.

#### 3. Local Gradients

For each operation (addition, multiplication, etc.), I defined a local gradient function. This function knows how to propagate gradients to its parents, using the chain rule. For example, for multiplication, the gradient with respect to each input is the other input.

#### 4. Backward Pass: Topological Traversal

To compute all gradients, I traverse the graph in reverse topological order. This ensures that each node’s gradient is computed only after all nodes it depends on have been processed. The backward pass starts from the output (setting its gradient to 1) and accumulates gradients for all inputs.

#### 5. Testing and Comparison

To verify correctness, I implemented the same computation in PyTorch and compared the gradients. This step is crucial: it ensures that the engine works as expected and matches industry standards.

### Why Each Step Matters

- **Computation Graph**: Without tracking dependencies, you can’t apply the chain rule automatically.
- **Operator Overloading**: Makes the engine usable and readable, so you can focus on the math, not the plumbing.
- **Local Gradients**: Encapsulates the math for each operation, making the engine extensible.
- **Backward Pass**: Ensures gradients are propagated correctly, even for complex graphs.
- **Comparison with PyTorch**: Validates the implementation and builds confidence.

---

## Conclusion

This project is a minimal, readable autodiff engine. By working through the code, you’ll gain a deep understanding of how frameworks like PyTorch and TensorFlow compute gradients, and you’ll be equipped to experiment with your own custom operations and models.

Feel free to explore the notebook, tweak the code, and run your own experiments. The best way to learn is by building!
