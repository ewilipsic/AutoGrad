# Grail - Automatic Differentiation Library

A modern C++ automatic differentiation library built from scratch, designed for efficient gradient computation and neural network training.

## Overview

Grail is a lightweight automatic differentiation (autodiff) library implemented in C++ that enables efficient computation of derivatives for mathematical functions[2][3]. The library implements reverse-mode automatic differentiation, making it particularly well-suited for machine learning applications where you need gradients with respect to many parameters[12].

## Key Features

- **Pure C++ Implementation**: Built entirely in C++ with minimal dependencies
- **Reverse-Mode Autodiff**: Implements efficient reverse-mode automatic differentiation with computational graph[2][3]
- **Tensor Operations**: Multi-dimensional array support with automatic gradient tracking[3]
- **Neural Network Layers**: Built-in support for common NN layers (Linear, ReLU, Sigmoid)[3]
- **Loss Functions**: Includes MSE and Binary Cross-Entropy loss functions[3]
- **Operator Overloading**: Intuitive mathematical expressions using C++ operator overloading

## Architecture

### Core Components

The library is organized into several key modules[2]:

- **`tensor.h/cpp`**: Core tensor class with automatic gradient tracking[3]
- **`node.h/cpp`**: Computational graph nodes for backward propagation[2]
- **`engine.h/cpp`**: Automatic differentiation engine
- **`nn.h/cpp`**: Neural network layers and loss functions[3]
- **`operations.cpp`**: Mathematical operations implementation
- **`utils.cpp`**: Utility functions

### Computational Graph

Grail builds a computational graph during the forward pass and uses reverse-mode differentiation to compute gradients efficiently[12]. Each operation creates nodes in the graph with backward functions that implement the chain rule[2].

## Installation

### Requirements
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- No external dependencies required

### Build Instructions

Since Grail appears to be a header-based library, you can include it directly in your project:

```bash
git clone https://github.com/ewilipsic/Grail-A_autograd_library_from_scatch.git
cd Grail-A_autograd_library_from_scatch
```

Include the main header in your C++ files:
```cpp
#include "src/grail.h"
```

If you want to build a proper CMake project, you can create a `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.12)
project(Grail)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create interface library for header-only usage
add_library(grail INTERFACE)

target_include_directories(grail INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

# Optional: Build examples
option(BUILD_EXAMPLES "Build example programs" ON)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
```

## Quick Start

### Basic Automatic Differentiation

```cpp
#include "src/grail.h"
using namespace grail;

int main() {
    // Create tensors with gradient tracking enabled
    tensor a(std::make_shared<Tensor>(
        std::vector<int>({3, 3}), 
        std::vector<std::vector<float>>({{2,1,2}, {2,5,6}, {1,2,2}}), 
        1  // requires_grad = true
    ));
    
    tensor b(std::make_shared<Tensor>(
        std::vector<int>({3, 3}), 
        std::vector<std::vector<float>>({{1,1,1}, {1,2,3}, {1,6,1}}), 
        1
    ));

    // Forward pass
    tensor c = matmul(a, b);
    tensor d = sigmoid(a) + c;
    tensor e = c + d;

    // Backward pass - compute gradients
    backward(e);

    // Access gradients
    std::cout << "grad a: "; printvec(a.grad()->arr);
    std::cout << "grad b: "; printvec(b.grad()->arr);
    
    return 0;
}
```

### Neural Network Training

```cpp
#include "src/grail.h"
using namespace grail;

class SimpleNetwork : public model {
public:
    SimpleNetwork() {
        params = {
            std::make_shared<linear>(4, 8),   // Input layer
            std::make_shared<relu>(),         // Activation
            std::make_shared<linear>(8, 3),   // Output layer
            std::make_shared<sigmoid>(),      // Final activation
        };
    }

    tensor forward(tensor input) override {
        tensor x = input;
        for (auto& layer : params) {
            x = layer->forward(x);
        }
        return x;
    }
};

int main() {
    SimpleNetwork model;
    
    // Training loop
    for (int epoch = 0; epoch < 100; epoch++) {
        // Forward pass
        tensor output = model.forward(input_data);
        
        // Compute loss
        tensor loss = BCEloss(output, target_data);
        
        // Backward pass
        backward(loss);
        
        // Update parameters
        model.update(0.01);  // learning rate = 0.01
        model.zero_grad();   // Clear gradients
    }
    
    return 0;
}
```

## Supported Operations

### Tensor Operations
- **Arithmetic**: Addition (`+`), Subtraction (`-`), Element-wise multiplication, Division
- **Matrix Operations**: Matrix multiplication (`matmul`)
- **Activation Functions**: ReLU, Sigmoid, Tanh
- **Mathematical Functions**: Square, Square root, Logarithm

### Neural Network Layers
- **Linear Layer**: Fully connected layer with weights and bias[3]
- **Activation Layers**: ReLU, Sigmoid[3]
- **Sequential Model**: Container for chaining multiple layers[3]

### Loss Functions
- **Mean Squared Error (MSE)**: For regression tasks[3]
- **Binary Cross-Entropy (BCE)**: For binary classification[3]

## Examples

The repository includes two comprehensive examples:

1. **Basic Autodiff Example** (`examples/autograd_example/main.cpp`): Demonstrates basic tensor operations and gradient computation[2]

2. **Iris Classification** (`examples/iris_classification/iris_example.cpp`): Complete neural network training example using the famous Iris dataset[2]

## Performance Characteristics

Grail is designed for educational purposes and clarity of implementation. Key characteristics:

- **Reverse-Mode Efficiency**: O(1) complexity for gradients regardless of input dimension[12]
- **Memory Management**: Automatic gradient accumulation and storage
- **Computational Graph**: Dynamic graph construction during forward pass

## Comparison with Other Libraries

Unlike production libraries like PyTorch or TensorFlow, Grail focuses on:

- **Educational Value**: Clear, readable implementation of autodiff concepts[15]
- **Minimal Dependencies**: Self-contained C++ implementation
- **Simplicity**: Straightforward API without complex optimizations
- **Transparency**: Easy to understand computational graph mechanics

For comparison, other C++ autodiff libraries include:
- **Adept**: High-performance library with expression templates[9][10]
- **AutoGrad**: Another educational autodiff implementation[20][23]
- **CppAD**: Mature automatic differentiation package

## Contributing

Contributions are welcome! Areas for improvement include:

1. **Build System**: Add comprehensive CMake configuration
2. **Documentation**: Expand API documentation
3. **Performance**: Optimize memory usage and computation speed
4. **Operations**: Add more mathematical functions and operations
5. **Testing**: Implement unit tests for all components

### Development Setup

```bash
git clone https://github.com/ewilipsic/Grail-A_autograd_library_from_scatch.git
cd Grail-A_autograd_library_from_scatch

# Compile examples
g++ -std=c++17 -I src examples/autograd_example/main.cpp -o autograd_example
g++ -std=c++17 -I src examples/iris_classification/iris_example.cpp -o iris_example
```

## License

This project is available as open source. Please check the repository for specific license information.

## Theoretical Background

Automatic differentiation computes exact derivatives by applying the chain rule to elementary operations[12][15]. Reverse-mode AD works by:

1. **Forward Pass**: Build computational graph while computing function values
2. **Backward Pass**: Traverse graph in reverse, accumulating partial derivatives
3. **Chain Rule**: Combine partial derivatives to compute total derivatives

This approach is particularly efficient for functions with many inputs and few outputs, making it ideal for machine learning optimization[12].

## Further Reading

- [Automatic Differentiation Overview](https://auto-ed.readthedocs.io/en/latest/mod3.html)[12]
- [Reverse-Mode Autodiff Tutorial](https://sidsite.com/posts/autodiff/)[15]
- [Computational Graphs and Backpropagation](http://www.cs.columbia.edu/~mcollins/ff2.pdf)[13]