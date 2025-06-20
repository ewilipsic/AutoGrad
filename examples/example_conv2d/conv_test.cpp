#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include "src\grail.h"

using namespace grail;

void test_conv2d() {
    std::cout << "Testing Conv2D implementation..." << std::endl;
    
    // Test 1: Basic functionality test
    std::cout << "Test 1: Basic Conv2D forward pass" << std::endl;
    
    // Create input tensor: batch=1, channels=1, height=5, width=5
    tensor input({1, 1, 5, 5}, 0.0, true);
    
    // Fill input with simple pattern for easy verification
    for(int i = 0; i < 25; i++) {
        input.arr()[i] = i + 1; // Values 1-25
    }
    
    // Create weight tensor: out_channels=1, in_channels=1, kernel=3x3
    tensor weight({1, 1, 3, 3}, 0.0, true);
    
    // Simple edge detection kernel
    weight[{0,0,0,0}] = -1; weight[{0,0,0,1}] = -1; weight[{0,0,0,2}] = -1;
    weight[{0,0,1,0}] = -1; weight[{0,0,1,1}] =  8; weight[{0,0,1,2}] = -1;
    weight[{0,0,2,0}] = -1; weight[{0,0,2,1}] = -1; weight[{0,0,2,2}] = -1;
    
    // Create bias
    tensor bias({1,1,3,3}, 0.0, true);
    bias[0] = 0.5;
    
    // Create Conv2D layer
    Conv2D conv(weight, bias, 1, 1, 0, 0, 3, 3, 1, 1);
    
    // Forward pass
    tensor output = conv.forward(input);
    
    // Check output shape
    assert(output.shape()[0] == 1); // batch
    assert(output.shape()[1] == 1); // out_channels
    assert(output.shape()[2] == 3); // height: (5-3)/1+1 = 3
    assert(output.shape()[3] == 3); // width: (5-3)/1+1 = 3
    
    std::cout << "✓ Output shape correct: [1,1,3,3]" << std::endl;
    
    // Test 2: Gradient computation test
    std::cout << "\nTest 2: Gradient computation" << std::endl;
    
    // Create simple loss (sum of outputs)
    tensor loss({1}, 0.0, true);
    float loss_val = 0.0;
    for(int i = 0; i < output.arr().size(); i++) {
        loss_val += output.arr()[i];
    }
    loss[0] = loss_val;
    
    // Set output gradients to 1 for backprop
    output.grad()->fill(1.0);
    
    // Backward pass
    backward(output);
    
    // Check if gradients exist
    assert(weight.grad() != nullptr);
    assert(bias.grad() != nullptr);
    
    std::cout << "✓ Gradients computed successfully" << std::endl;
    
    // Test 3: Parameter update test
    std::cout << "\nTest 3: Parameter updates" << std::endl;
    
    // Store original values
    float original_weight = weight[0];
    float original_bias = bias[0];
    
    // Update parameters
    float learning_rate = 0.01;
    conv.update(learning_rate);
    
    // Check if parameters changed
    assert(weight[0] != original_weight);
    assert(bias[0] != original_bias);
    
    std::cout << "✓ Parameters updated correctly" << std::endl;
    
    // Test 4: Zero grad test
    std::cout << "\nTest 4: Zero gradients" << std::endl;
    
    conv.zero_grad();
    
    // Check if gradients are zeroed
    bool weight_grad_zero = true;
    bool bias_grad_zero = true;
    
    for(int i = 0; i < weight.grad()->arr.size(); i++) {
        if((*weight.grad())[i] != 0.0) weight_grad_zero = false;
    }
    
    for(int i = 0; i < bias.grad()->arr.size(); i++) {
        if((*bias.grad())[i] != 0.0) bias_grad_zero = false;
    }
    
    assert(weight_grad_zero);
    assert(bias_grad_zero);
    
    std::cout << "✓ Gradients zeroed successfully" << std::endl;
    
    // Test 5: Multiple channels test
    std::cout << "\nTest 5: Multiple channels" << std::endl;
    
    // Input: 1 batch, 3 channels, 4x4
    tensor multi_input({1, 3, 4, 4}, 0.0, true);
    
    // Fill with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    
    for(int i = 0; i < multi_input.arr().size(); i++) {
        multi_input.arr()[i] = dis(gen);
    }
    
    // Weight: 2 out_channels, 3 in_channels, 2x2 kernel
    tensor multi_weight({2, 3, 2, 2}, 0.0, true);
    for(int i = 0; i < multi_weight.arr().size(); i++) {
        multi_weight.arr()[i] = dis(gen);
    }
    
    // Bias: 2 out_channels
    tensor multi_bias({1,2,3,3}, 0.0, true);
    multi_bias[0] = 0.1;
    multi_bias[1] = 0.2;
    
    Conv2D multi_conv(multi_weight, multi_bias, 1, 1, 0, 0, 2, 2, 3, 2);
    tensor multi_output = multi_conv.forward(multi_input);
    
    // Check output shape: [1, 2, 3, 3]
    assert(multi_output.shape()[0] == 1);
    assert(multi_output.shape()[1] == 2);
    assert(multi_output.shape()[2] == 3);
    assert(multi_output.shape()[3] == 3);
    
    std::cout << "✓ Multiple channels test passed" << std::endl;
    
    // Test 6: Padding test
    std::cout << "\nTest 6: Padding test" << std::endl;
    
    tensor pad_input({1, 1, 3, 3}, 0.0, true);
    for(int i = 0; i < 9; i++) {
        pad_input.arr()[i] = i + 1;
    }
    
    tensor pad_weight({1, 1, 3, 3}, 0.0, true);
    for(int i = 0; i < 9; i++) {
        pad_weight.arr()[i] = 1.0; // All ones kernel
    }
    
    tensor pad_bias({1,1,3,3}, 0.0, true);
    
    Conv2D pad_conv(pad_weight, pad_bias, 1, 1, 1, 1, 3, 3, 1, 1); // padding=1
    tensor pad_output = pad_conv.forward(pad_input);
    
    // With padding=1, output should be same size as input
    assert(pad_output.shape()[2] == 3);
    assert(pad_output.shape()[3] == 3);
    
    std::cout << "✓ Padding test passed" << std::endl;
    
    // Test 7: Stride test
    std::cout << "\nTest 7: Stride test" << std::endl;
    
    tensor stride_input({1, 1, 6, 6}, 0.0, true);
    for(int i = 0; i < 36; i++) {
        stride_input.arr()[i] = i + 1;
    }
    
    tensor stride_weight({1, 1, 2, 2}, 0.0, true);
    for(int i = 0; i < 4; i++) {
        stride_weight.arr()[i] = 1.0;
    }
    
    tensor stride_bias({1,1,3,3}, 0.0, true);
    
    Conv2D stride_conv(stride_weight, stride_bias, 2, 2, 0, 0, 2, 2, 1, 1); // stride=2
    tensor stride_output = stride_conv.forward(stride_input);
    
    // With stride=2, output should be smaller
    int expected_size = (6 - 2) / 2 + 1; // = 3
    assert(stride_output.shape()[2] == expected_size);
    assert(stride_output.shape()[3] == expected_size);
    
    std::cout << "✓ Stride test passed" << std::endl;
    
    std::cout << "\n🎉 All Conv2D tests passed successfully!" << std::endl;
}

// Additional helper function to print tensor values for debugging
void print_tensor_info(const tensor& t, const std::string& name) {
    std::cout << name << " shape: [";
    for(int i = 0; i < t.shape().size(); i++) {
        std::cout << t.shape()[i];
        if(i < t.shape().size() - 1) std::cout << ",";
    }
    std::cout << "]" << std::endl;
    
    std::cout << name << " first few values: ";
    for(int i = 0; i < std::min(5, (int)t.arr().size()); i++) {
        std::cout << t.arr()[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    try {
        test_conv2d();
    } catch (const std::exception& e) {
        std::cout << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
