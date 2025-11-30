#ifndef NN_INTERFACES_H
#define NN_INTERFACES_H

#include <utec/algebra/Tensor.h>

/**
 * =================================================================
 * NOTACIÓN DE COMPLEJIDAD ALGORÍTMICA (O)
 * =================================================================
 * S_batch: Tamaño del batch actual (número de muestras).
 * M_in: Número de características de entrada.
 * M_out: Número de neuronas de salida.
 * P_layer: Número total de parámetros de la capa.
 * N_elements: Número total de elementos en el tensor de salida/gradiente
 * (generalmente S_batch * M_out).
 * C_mat_op: Costo de operaciones matriciales (ej. C_mat_mul).
 * =================================================================
 */

namespace utec {
namespace neural_network {

template<typename T> class IOptimizer;

// Interfaz base para capas de la red neuronal
template<typename T>
class ILayer {
public:
    virtual ~ILayer() = default;
    
    // --- Algoritmo de Propagación hacia Adelante (Forward) ---
    // Complejidad esperada: O(C_mat_op) o O(N_elements).
    // Depende del tipo de capa. Para Dense es O(C_mat_mul); para Activaciones es O(N_elements).
    virtual utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& input) = 0;
    
    // --- Algoritmo de Propagación hacia Atrás (Backward) ---
    // Complejidad esperada: O(C_mat_op) o O(N_elements).
    // Debe calcular dX. Su costo es comparable al forward.
    virtual utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& gradient) = 0;
    
    // --- Algoritmo de Actualización de Parámetros ---
    // Complejidad esperada: O(P_layer) para capas con parámetros (ej. Dense).
    // O(1) para capas sin parámetros (ej. Activación).
    virtual void update_params(IOptimizer<T>& optimizer) {}
};

// Interfaz base para funciones de pérdida (loss functions)
template<typename T, int N>
class ILoss {
public:
    virtual ~ILoss() = default;
    
    // --- Algoritmo de Cálculo de Pérdida (Loss) ---
    // Complejidad esperada: O(N_elements).
    // Requiere iterar sobre todos los elementos de la salida (predicciones) y el objetivo (Y).
    virtual T loss() const = 0;
    
    // --- Algoritmo de Cálculo del Gradiente de Pérdida ---
    // Complejidad esperada: O(N_elements).
    // Requiere calcular el gradiente dL/dA o dL/dZ para cada elemento de salida.
    virtual utec::algebra::Tensor<T, N> loss_gradient() const = 0;
};

// Interfaz base para optimizadores
template<typename T>
class IOptimizer {
public:
    virtual ~IOptimizer() = default;
    
    // --- Algoritmo de Actualización de Parámetros ---
    // Complejidad esperada: O(P_layer).
    // Lineal con el número de parámetros de la capa que se está actualizando.
    virtual void update(utec::algebra::Tensor<T, 2>& params, 
                       const utec::algebra::Tensor<T, 2>& grads) = 0;
    
    // --- Algoritmo de Paso (Step) Global ---
    // Complejidad esperada: O(1).
    // Generalmente incrementa un contador de paso global. Podría ser O(P) si
    // se manejan los momentos globales para todos los parámetros (como en Adam).
    virtual void step() {}
};

} // namespace neural_network
} // namespace utec

#endif // NN_INTERFACES_H
