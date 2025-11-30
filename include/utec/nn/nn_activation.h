#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include <utec/nn/nn_interfaces.h>
#include <utec/algebra/Tensor.h>
#include <cmath>
#include <algorithm>

namespace utec {
namespace neural_network {

// Función de activación ReLU: f(x) = max(0, x)
template<typename T>
class ReLU final : public ILayer<T> {
private:
    utec::algebra::Tensor<T, 2> input_;
    
public:
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& z) override {
        input_ = z;
        
        auto result = z;
        for (auto& val : result) {
            val = std::max(T{0}, val);
        }
        return result;
    }
    
    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& gradient) override {
        auto result = gradient;
        
        auto it_grad = result.begin();
        auto it_input = input_.cbegin();
        
        while (it_grad != result.end()) {
            if (*it_input <= T{0}) {  // ⭐ Cambio: <= en lugar de <
                *it_grad = T{0};
            }
            ++it_grad;
            ++it_input;
        }
        
        return result;
    }
    
    // No tiene parámetros, así que update_params no hace nada (heredado de ILayer)
};

// Función de activación Sigmoid: f(x) = 1 / (1 + e^(-x))
template<typename T>
class Sigmoid final : public ILayer<T> {
private:
    utec::algebra::Tensor<T, 2> output_;
    
    // ⭐ CRÍTICO: Epsilon para prevenir valores exactos de 0 o 1
    static constexpr T EPSILON = T{1e-7};
    
public:
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& z) override {
        auto result = z;
        
        for (auto& val : result) {
            // Sigmoid básico
            T sigmoid_val = T{1} / (T{1} + std::exp(-val));
            
            // ⭐ CLIP para evitar exactamente 0 o 1 (causa problemas con BCE)
            val = std::max(EPSILON, std::min(T{1} - EPSILON, sigmoid_val));
        }
        
        output_ = result;
        return result;
    }
    
    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& gradient) override {
        auto result = gradient;
        
        auto it_grad = result.begin();
        auto it_output = output_.cbegin();
        
        while (it_grad != result.end()) {
            T sigmoid_val = *it_output;
            // Derivada: sigmoid(x) * (1 - sigmoid(x))
            *it_grad = (*it_grad) * sigmoid_val * (T{1} - sigmoid_val);
            ++it_grad;
            ++it_output;
        }
        
        return result;
    }
    
    // No tiene parámetros, así que update_params no hace nada (heredado de ILayer)
};

} // namespace neural_network
} // namespace utec

#endif // NN_ACTIVATION_H