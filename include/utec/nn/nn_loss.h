#ifndef NN_LOSS_H
#define NN_LOSS_H

#include <utec/nn/nn_interfaces.h>
#include <utec/algebra/Tensor.h>
#include <cmath>

/**
 * =================================================================
 * NOTACIÓN DE COMPLEJIDAD ALGORÍTMICA (O)
 * =================================================================
 * S_batch: Número de muestras en el lote actual.
 * M_out: Número de neuronas de salida.
 * N_elements: Número total de elementos en el tensor de salida.
 * N_elements = S_batch * M_out.
 * =================================================================
 */

namespace utec {
namespace neural_network {

// Mean Squared Error Loss: MSE = (1/n) * Σ(y_pred - y_true)²
template<typename T>
class MSELoss final : public ILoss<T, 2> {
private:
    utec::algebra::Tensor<T, 2> y_predicted_; // (S_batch x M_out)
    utec::algebra::Tensor<T, 2> y_true_;      // (S_batch x M_out)
    
public:
    // Constructor
    // Complejidad: O(1). Solo almacena referencias/copias.
    MSELoss(const utec::algebra::Tensor<T, 2>& y_prediction, 
            const utec::algebra::Tensor<T, 2>& y_true) 
        : y_predicted_(y_prediction), y_true_(y_true) {
        
        if (y_predicted_.shape() != y_true_.shape()) {
            throw std::invalid_argument("Predictions and true values must have the same shape");
        }
    }
    
    // --- Algoritmo de Cálculo de Pérdida (MSE) ---
    // Complejidad: O(N_elements).
    // Se realiza resta y suma de cuadrados sobre todos los N_elements.
    T loss() const override {
        // Resta elemento a elemento: O(N_elements)
        auto diff = y_predicted_ - y_true_; 
        
        T sum = T{0};
        for (const auto& val : diff) { // Bucle O(N_elements)
            sum += val * val; // O(1)
        }
        
        // División final O(1)
        return sum / static_cast<T>(y_predicted_.size());
    }
    
    // --- Algoritmo de Cálculo del Gradiente de Pérdida (MSE) ---
    // dL/dY_pred = (2/n) * (Y_pred - Y_true)
    // Complejidad: O(N_elements).
    utec::algebra::Tensor<T, 2> loss_gradient() const override {
        // Resta elemento a elemento: O(N_elements)
        auto gradient = y_predicted_ - y_true_; 
        T factor = T{2} / static_cast<T>(y_predicted_.size());
        
        // Multiplicación por factor: O(N_elements)
        gradient = gradient * factor; 
        return gradient;
    }
};

// Binary Cross Entropy Loss: BCE = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]
template<typename T>
class BinaryCrossEntropyLoss final : public ILoss<T, 2> {
private:
    utec::algebra::Tensor<T, 2> y_predicted_; // (S_batch x M_out)
    utec::algebra::Tensor<T, 2> y_true_;      // (S_batch x M_out)
    
    static constexpr T epsilon = T{1e-7};
    
public:
    // Constructor
    // Complejidad: O(1).
    BinaryCrossEntropyLoss(const utec::algebra::Tensor<T, 2>& y_prediction,
            const utec::algebra::Tensor<T, 2>& y_true) 
        : y_predicted_(y_prediction), y_true_(y_true) {
        
        if (y_predicted_.shape() != y_true_.shape()) {
            throw std::invalid_argument("Predictions and true values must have the same shape");
        }
    }
    
    // --- Algoritmo de Cálculo de Pérdida (BCE) ---
    // Complejidad: O(N_elements).
    // Se realiza una iteración y cálculo logarítmico para cada elemento.
    T loss() const override {
        T sum = T{0};
        
        auto it_pred = y_predicted_.cbegin();
        auto it_true = y_true_.cbegin();
        
        while (it_pred != y_predicted_.cend()) { // Bucle O(N_elements)
            // Operaciones de clip y logaritmo O(1)
            T p = std::max(epsilon, std::min(T{1} - epsilon, *it_pred));
            T y = *it_true;
            
            sum += -(y * std::log(p) + (T{1} - y) * std::log(T{1} - p));
            
            ++it_pred;
            ++it_true;
        }
        
        // División final O(1)
        return sum / static_cast<T>(y_predicted_.size());
    }
    
    // --- Algoritmo de Cálculo del Gradiente de Pérdida (BCE) ---
    // dL/dP = -(y/p - (1-y)/(1-p)) / n
    // Complejidad: O(N_elements).
    utec::algebra::Tensor<T, 2> loss_gradient() const override {
        auto gradient = y_predicted_; // O(N_elements) para la copia
        
        auto it_grad = gradient.begin();
        auto it_pred = y_predicted_.cbegin();
        auto it_true = y_true_.cbegin();
        
        T n = static_cast<T>(y_predicted_.size());
        
        while (it_grad != gradient.end()) { // Bucle O(N_elements)
            // Operaciones de clip y división O(1)
            T p = std::max(epsilon, std::min(T{1} - epsilon, *it_pred));
            T y = *it_true;
            
            // Cálculo del gradiente dL/dP
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
