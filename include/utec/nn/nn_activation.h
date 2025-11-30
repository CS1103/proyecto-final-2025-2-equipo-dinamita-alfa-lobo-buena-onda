#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include <utec/nn/nn_interfaces.h>
#include <utec/algebra/Tensor.h>
#include <cmath>
#include <algorithm>

/**
 * =================================================================
 * NOTACIÓN DE COMPLEJIDAD ALGORÍTMICA (O)
 * =================================================================
 * S_batch: Número de muestras en el lote actual.
 * M_out: Número de características/neuronas en la capa de salida.
 * N_elements: Número total de elementos en el tensor de entrada/salida.
 * N_elements = S_batch * M_out.
 * =================================================================
 */

namespace utec {
namespace neural_network {

// Función de activación ReLU: f(x) = max(0, x)
template<typename T>
class ReLU final : public ILayer<T> {
private:
    // Almacena la entrada (Z) para usarla en la retropropagación (backward)
    utec::algebra::Tensor<T, 2> input_;
    
public:
    // --- Algoritmo Forward Pass (ReLU) ---
    // Complejidad: O(N_elements).
    // Operación elemento a elemento, lineal con el tamaño del tensor de entrada.
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& z) override {
        input_ = z; // Copia el tensor de entrada para usarlo en backward. O(N_elements)
        
        auto result = z;
        for (auto& val : result) { // Bucle O(N_elements)
            // Aplicación de ReLU (max(0, x)). O(1)
            val = std::max(T{0}, val);
        }
        return result;
    }
    
    // --- Algoritmo Backward Pass (ReLU) ---
    // Complejidad: O(N_elements).
    // Aplica la derivada elemento a elemento (1 si input > 0, 0 si input <= 0).
    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& gradient) override {
        auto result = gradient; // O(N_elements) para copiar el gradiente de la siguiente capa.
        
        auto it_grad = result.begin();
        auto it_input = input_.cbegin();
        
        while (it_grad != result.end()) { // Bucle O(N_elements)
            // Si la entrada original (antes de ReLU) fue <= 0, el gradiente de la pérdida es 0.
            if (*it_input <= T{0}) {
                *it_grad = T{0}; // Gradiente que pasa a la capa anterior es 0.
            }
            ++it_grad;
            ++it_input;
        }
        
        return result;
    }
    
    // Las funciones de activación no tienen parámetros (pesos/sesgos) que actualizar.
};

// Función de activación Sigmoid: f(x) = 1 / (1 + e^(-x))
template<typename T>
class Sigmoid final : public ILayer<T> {
private:
    // Almacena el resultado de la activación (A = sigmoid(Z)) para usarlo en backward.
    utec::algebra::Tensor<T, 2> output_;
    
    // Épsilon para prevenir la salida de valores exactos de 0 o 1, que causan
    // inestabilidad logarítmica en funciones de pérdida como BCE (Binary Cross-Entropy).
    static constexpr T EPSILON = T{1e-7};
    
public:
    // --- Algoritmo Forward Pass (Sigmoid) ---
    // Complejidad: O(N_elements).
    // Operación elemento a elemento, lineal con el tamaño del tensor de entrada.
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& z) override {
        auto result = z;
        
        for (auto& val : result) { // Bucle O(N_elements)
            // Cálculo de Sigmoid: O(1) debido a std::exp()
            T sigmoid_val = T{1} / (T{1} + std::exp(-val));
            
            // Limita el valor para estabilidad. O(1)
            val = std::max(EPSILON, std::min(T{1} - EPSILON, sigmoid_val));
        }
        
        output_ = result; // Almacenar salida para backward. O(N_elements)
        return result;
    }
    
    // --- Algoritmo Backward Pass (Sigmoid) ---
    // Complejidad: O(N_elements).
    // Aplica la derivada elemento a elemento: A * (1 - A).
    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& gradient) override {
        auto result = gradient; // O(N_elements) para copiar el gradiente de la siguiente capa.
        
        auto it_grad = result.begin();
        auto it_output = output_.cbegin(); // Usamos la salida almacenada (A)
        
        while (it_grad != result.end()) { // Bucle O(N_elements)
            T sigmoid_val = *it_output;
            // La retropropagación es la multiplicación del gradiente de la capa
            // siguiente por la derivada de Sigmoid: dL/dZ = dL/dA * (A * (1 - A)). O(1)
            *it_grad = (*it_grad) * sigmoid_val * (T{1} - sigmoid_val);
            ++it_grad;
            ++it_output;
        }
        
        return result;
    }
    
    // Las funciones de activación no tienen parámetros (pesos/sesgos) que actualizar.
};

} // namespace neural_network
} // namespace utec

#endif // NN_ACTIVATION_H
