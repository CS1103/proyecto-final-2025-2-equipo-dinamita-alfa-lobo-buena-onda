#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#include <utec/nn/nn_interfaces.h>
#include <utec/algebra/Tensor.h>
#include <cmath>
#include <map>

/**
 * =================================================================
 * NOTACIÓN DE COMPLEJIDAD ALGORÍTMICA (O)
 * =================================================================
 * P_layer: Número total de parámetros (pesos o sesgos) en el tensor
 * que se está actualizando (tamaño de 'params' y 'grads').
 * L_dense: Número de capas Dense (que tienen parámetros) en la red.
 * t: Contador de pasos global del optimizador.
 * =================================================================
 */

namespace utec {
namespace neural_network {

// Stochastic Gradient Descent: θ = θ - lr * ∇θ
template<typename T>
class SGD final : public IOptimizer<T> {
private:
    T learning_rate_;
    
public:
    // Constructor
    // Complejidad: O(1).
    explicit SGD(T learning_rate = T{0.01}) 
        : learning_rate_(learning_rate) {}
    
    // --- Algoritmo de Actualización (SGD) ---
    // Complejidad: O(P_layer).
    // Lineal con el número de parámetros a actualizar.
    void update(utec::algebra::Tensor<T, 2>& params, 
                const utec::algebra::Tensor<T, 2>& grads) override {
        
        auto it_param = params.begin();
        auto it_grad = grads.cbegin();
        
        while (it_param != params.end()) { // Bucle O(P_layer)
            // Operación O(1): θ = θ - lr * ∇θ
            *it_param -= learning_rate_ * (*it_grad);
            ++it_param;
            ++it_grad;
        }
    }
};

// Adam Optimizer: Combina momentum y RMSprop
template<typename T>
class Adam final : public IOptimizer<T> {
private:
    T learning_rate_;
    T beta1_;       // Decay rate para el primer momento
    T beta2_;       // Decay rate para el segundo momento
    T epsilon_;     // Para estabilidad numérica
    size_t t_;      // Contador de pasos
    
    // Mapas para guardar momentos de cada parámetro (clave: puntero al tensor)
    std::map<void*, utec::algebra::Tensor<T, 2>> m_;  // Primer momento (media)
    std::map<void*, utec::algebra::Tensor<T, 2>> v_;  // Segundo momento (varianza)
    
public:
    // Constructor
    // Complejidad: O(1).
    explicit Adam(T learning_rate = T{0.001}, 
                  T beta1 = T{0.9}, 
                  T beta2 = T{0.999}, 
                  T epsilon = T{1e-8}) 
        : learning_rate_(learning_rate), 
          beta1_(beta1), 
          beta2_(beta2), 
          epsilon_(epsilon),
          t_(0) {}
    
    // --- Algoritmo de Actualización (Adam) ---
    // Complejidad: O(P_layer).
    // Lineal con el número de parámetros a actualizar.
    void update(utec::algebra::Tensor<T, 2>& params, 
                const utec::algebra::Tensor<T, 2>& grads) override {
        
        void* key = static_cast<void*>(&params);
        
        // Inicializar momentos. La búsqueda y el acceso son O(log(L_dense)) o O(1) amortizado.
        // La inicialización de los tensores es O(P_layer) si ocurre la primera vez.
        if (m_.find(key) == m_.end()) {
            m_[key] = utec::algebra::Tensor<T, 2>(params.shape()); 
            m_[key].fill(T{0});
            v_[key] = utec::algebra::Tensor<T, 2>(params.shape()); 
            v_[key].fill(T{0});
        }
        
        auto& m = m_[key];
        auto& v = v_[key];
        
        // Bucle para actualizar momentos m y v
        // Complejidad: O(P_layer)
        auto it_m = m.begin();
        auto it_v = v.begin();
        auto it_grad = grads.cbegin();
        
        while (it_grad != grads.cend()) {
            T g = *it_grad;
            
            // m = ... O(1)
            *it_m = beta1_ * (*it_m) + (T{1} - beta1_) * g;
            
            // v = ... O(1)
            *it_v = beta2_ * (*it_v) + (T{1} - beta2_) * g * g;
            
            ++it_m;
            ++it_v;
            ++it_grad;
        }
        
        // Cálculo de corrección de bias. std::pow es O(1).
        T bias_correction1 = T{1} - std::pow(beta1_, t_ + 1);
        T bias_correction2 = T{1} - std::pow(beta2_, t_ + 1);
        
        // Bucle para actualizar parámetros
        // Complejidad: O(P_layer)
        auto it_param = params.begin();
        it_m = m.begin();
        it_v = v.begin();
        
        while (it_param != params.end()) {
            // Cálculo de momentos corregidos m_hat y v_hat. O(1)
            T m_hat = (*it_m) / bias_correction1;
            T v_hat = (*it_v) / bias_correction2;
            
            // params = params - lr * m_hat / (sqrt(v_hat) + epsilon). O(1)
            *it_param -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            
            ++it_param;
            ++it_m;
            ++it_v;
        }
    }
    
    // --- Algoritmo de Paso Global ---
    // Complejidad: O(1).
    // Solo incrementa un contador.
    void step() override {
        ++t_;
    }
};

} // namespace neural_network
} // namespace utec

#endif // NN_OPTIMIZER_H
