#ifndef NN_LOSS_H
#define NN_LOSS_H

#include <utec/nn/nn_interfaces.h>
#include <utec/algebra/Tensor.h>
#include <cmath>

namespace utec {
namespace neural_network {

// Mean Squared Error Loss: MSE = (1/n) * Σ(y_pred - y_true)²
template<typename T>
class MSELoss final : public ILoss<T, 2> {
private:
    utec::algebra::Tensor<T, 2> y_predicted_;
    utec::algebra::Tensor<T, 2> y_true_;
    
public:
    MSELoss(const utec::algebra::Tensor<T, 2>& y_prediction, 
            const utec::algebra::Tensor<T, 2>& y_true) 
        : y_predicted_(y_prediction), y_true_(y_true) {
        
        if (y_predicted_.shape() != y_true_.shape()) {
            throw std::invalid_argument("Predictions and true values must have the same shape");
        }
    }
    
    T loss() const override {
        auto diff = y_predicted_ - y_true_;
        
        T sum = T{0};
        for (const auto& val : diff) {
            sum += val * val;
        }
        
        return sum / static_cast<T>(y_predicted_.size());
    }
    
    utec::algebra::Tensor<T, 2> loss_gradient() const override {
        auto gradient = y_predicted_ - y_true_;
        T factor = T{2} / static_cast<T>(y_predicted_.size());
        
        gradient = gradient * factor;
        return gradient;
    }
};

// Binary Cross Entropy Loss: BCE = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]
template<typename T>
class BinaryCrossEntropyLoss final : public ILoss<T, 2> {
private:
    utec::algebra::Tensor<T, 2> y_predicted_;
    utec::algebra::Tensor<T, 2> y_true_;
    
    // ⭐ CAMBIO CRÍTICO: Epsilon aumentado de 1e-15 a 1e-7
    // 1e-15 es demasiado pequeño y puede causar problemas numéricos
    static constexpr T epsilon = T{1e-7};
    
public:
    BinaryCrossEntropyLoss(const utec::algebra::Tensor<T, 2>& y_prediction,
            const utec::algebra::Tensor<T, 2>& y_true) 
        : y_predicted_(y_prediction), y_true_(y_true) {
        
        if (y_predicted_.shape() != y_true_.shape()) {
            throw std::invalid_argument("Predictions and true values must have the same shape");
        }
    }
    
    T loss() const override {
        T sum = T{0};
        
        auto it_pred = y_predicted_.cbegin();
        auto it_true = y_true_.cbegin();
        
        while (it_pred != y_predicted_.cend()) {
            // Clip para estabilidad numérica
            T p = std::max(epsilon, std::min(T{1} - epsilon, *it_pred));
            T y = *it_true;
            
            // BCE: -[y*log(p) + (1-y)*log(1-p)]
            sum += -(y * std::log(p) + (T{1} - y) * std::log(T{1} - p));
            
            ++it_pred;
            ++it_true;
        }
        
        return sum / static_cast<T>(y_predicted_.size());
    }
    
    // ⭐ GRADIENTE CORREGIDO
    // Fórmula correcta: dL/dp = -(y/p - (1-y)/(1-p)) / n
    // Simplificado: dL/dp = (p - y) / (p * (1-p) * n)
    // Pero la forma más estable es: dL/dp = -(y/p - (1-y)/(1-p)) / n
    utec::algebra::Tensor<T, 2> loss_gradient() const override {
        auto gradient = y_predicted_;
        
        auto it_grad = gradient.begin();
        auto it_pred = y_predicted_.cbegin();
        auto it_true = y_true_.cbegin();
        
        T n = static_cast<T>(y_predicted_.size());
        
        while (it_grad != gradient.end()) {
            // Clip para evitar división por cero
            T p = std::max(epsilon, std::min(T{1} - epsilon, *it_pred));
            T y = *it_true;
            
            // ⭐ GRADIENTE CORRECTO: -(y/p - (1-y)/(1-p)) / n
            *it_grad = -(y / p - (T{1} - y) / (T{1} - p)) / n;
            
            ++it_grad;
            ++it_pred;
            ++it_true;
        }
        
        return gradient;
    }
};

} // namespace neural_network
} // namespace utec

#endif // NN_LOSS_H