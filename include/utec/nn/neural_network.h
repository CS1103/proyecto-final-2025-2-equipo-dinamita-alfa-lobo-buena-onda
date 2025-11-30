#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <utec/nn/nn_interfaces.h>
#include <utec/nn/nn_activation.h>
#include <utec/nn/nn_dense.h>
#include <utec/nn/nn_loss.h>
#include <utec/nn/nn_optimizer.h>
#include <utec/algebra/Tensor.h>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream> 
#include <typeinfo> 

/**
 * =================================================================
 * NOTACIÓN DE COMPLEJIDAD ALGORÍTMICA (O)
 * =================================================================
 * L: Número de capas en la red.
 * E: Número de épocas de entrenamiento.
 * N: Número total de muestras de entrenamiento.
 * B: Tamaño del batch (batch_size).
 * P: Número total de parámetros (pesos y sesgos) en la red.
 * S_batch: Tamaño del batch actual (variable, <= B).
 * F: Costo computacional de la propagación de una sola muestra
 * a través de toda la red, F ≈ Σ (M_in * M_out) para capas densas.
 * C_layer^op: Costo de una operación (forward, backward, update) en una única capa.
 * F_input: Número de características (features) en el set de datos.
 * =================================================================
 */

namespace utec {
namespace neural_network {

using namespace utec::algebra;

template<typename T>
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<ILayer<T>>> layers_;

    // --- Algoritmo Forward Pass ---
    // Complejidad: O(L * C_layer^forward) o O(S_batch * F).
    // La complejidad es lineal con el número de capas (L) y con el costo F de propagación por muestra.
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& input) {
        auto output = input;
        for (auto& layer : layers_) { // Bucle O(L)
            output = layer->forward(output); // O(C_layer^forward)
        }
        return output;
    }

    // --- Algoritmo Backward Pass (Backpropagation) ---
    // Complejidad: O(L * C_layer^backward) o O(S_batch * F).
    // Similar al forward, lineal con L y F.
    void backward(const utec::algebra::Tensor<T, 2>& gradient) {
        auto grad = gradient;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) { // Bucle O(L)
            grad = (*it)->backward(grad); // O(C_layer^backward)
        }
    }

    // --- Algoritmo de Actualización de Parámetros ---
    // Complejidad: O(P + C_optimizer^step).
    // Lineal con el número total de parámetros (P) de la red.
    void update_parameters(IOptimizer<T>& optimizer) {
        for (auto& layer : layers_) { // Bucle O(L)
            // La complejidad de update_params es O(P_layer), donde P_layer son los
            // parámetros de esa capa. Sumado a todas las capas es O(P).
            layer->update_params(optimizer); 
        }
        optimizer.step();  // O(C_optimizer^step), generalmente O(1) o O(P) para optimizadores complejos como Adam.
    }

    // --- Algoritmo para Extraer Batch ---
    // Complejidad: O(S_batch * F_input).
    // Se copian S_batch filas, cada una con F_input características.
    utec::algebra::Tensor<T, 2> extract_batch(const utec::algebra::Tensor<T, 2>& data,
                                               size_t start,
                                               size_t size) {
        const auto shape = data.shape();
        size_t features = shape[1]; // F_input

        utec::algebra::Tensor<T, 2> batch(size, features); // O(S_batch * F_input) para inicializar

        for (size_t i = 0; i < size; ++i) { // Bucle O(S_batch)
            for (size_t j = 0; j < features; ++j) { // Bucle O(F_input)
                batch(i, j) = data(start + i, j); // O(1) acceso a elemento del tensor 2D
            }
        }

        return batch;
    }

public:
    // --- Algoritmo para Añadir Capa ---
    // Complejidad: O(1) amortizado.
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers_.push_back(std::move(layer));
    }

    // --- Algoritmo de Entrenamiento (Bucle Completo) ---
    // Complejidad Total: O(E * N * F).
    // La complejidad es lineal con Épocas (E), Muestras Totales (N) y el Costo de Propagación por Muestra (F).
    template<template<typename> class LossType,
             template<typename> class OptimizerType = SGD>
    void train(const utec::algebra::Tensor<T, 2>& X,
               const utec::algebra::Tensor<T, 2>& Y,
               const size_t epochs,
               const size_t batch_size,
               T learning_rate) {

        OptimizerType<T> optimizer(learning_rate);
        const auto shape_X = X.shape();
        const size_t total_samples = shape_X[0]; // N

        for (size_t epoch = 0; epoch < epochs; ++epoch) { // Bucle O(E)
            T total_loss = T{0};
            size_t num_batches = 0;

            // Bucle Batch: O(N / B) iteraciones
            for (size_t batch_start = 0; batch_start < total_samples; batch_start += batch_size) {
                size_t batch_end = std::min(batch_start + batch_size, total_samples);
                size_t current_batch_size = batch_end - batch_start; // S_batch

                // Costo: O(S_batch * F_input)
                auto X_batch = extract_batch(X, batch_start, current_batch_size);
                auto Y_batch = extract_batch(Y, batch_start, current_batch_size);

                // Costo: O(S_batch * F)
                auto predictions = forward(X_batch);

                LossType<T> loss_fn(predictions, Y_batch);
                // Costo de loss() y loss_gradient() es O(S_batch * F_output)
                T loss = loss_fn.loss(); 
                // ...
                auto gradient = loss_fn.loss_gradient();

                // Costo: O(S_batch * F)
                backward(gradient);

                // Costo: O(P)
                update_parameters(optimizer);
            }
            // ... (comentarios de impresión) ...
        }
    }

    // --- Algoritmo de Predicción ---
    // Complejidad: O(N_pred * F).
    // Lineal con el número de muestras a predecir (N_pred) y el costo F.
    utec::algebra::Tensor<T, 2> predict(const utec::algebra::Tensor<T, 2>& X) {
        return forward(X); // O(N_pred * F)
    }

    // -----------------------------------------------------------------
    //  MÉTODOS DE SERIALIZACIÓN
    // -----------------------------------------------------------------

    /**
     * @brief Serializa el estado de la red.
     * Complejidad: O(P_dense).
     * Lineal con el número de parámetros (P_dense) a guardar.
     */
    void save_state(const std::string& filepath) const {
        std::ofstream ofs(filepath, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("Error al abrir el archivo para guardar el modelo.");
        }

        for (const auto& layer : layers_) { // Bucle O(L)
            if (auto dense_layer = dynamic_cast<Dense<T>*>(layer.get())) {
                // Asume que save_parameters es O(P_layer)
                dense_layer->save_parameters(ofs);
            }
        }
        ofs.close();
    }

    /**
     * @brief Deserializa y carga el estado de la red.
     * Complejidad: O(P_dense).
     * Lineal con el número de parámetros (P_dense) a cargar.
     */
    void load_state(const std::string& filepath) {
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error("Error al abrir el archivo para cargar el modelo. ¿Está entrenado?");
        }

        for (auto& layer : layers_) { // Bucle O(L)
            if (auto dense_layer = dynamic_cast<Dense<T>*>(layer.get())) {
                // Asume que load_parameters es O(P_layer)
                dense_layer->load_parameters(ifs);
            }
        }
        ifs.close();
    }
};

} // namespace neural_network
} // namespace utec

#endif // NEURAL_NETWORK_H
