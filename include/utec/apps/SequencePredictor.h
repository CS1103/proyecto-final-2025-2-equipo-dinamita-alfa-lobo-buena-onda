#ifndef UTAP_SEQUENCE_PREDICTOR_H
#define UTAP_SEQUENCE_PREDICTOR_H

#include <iostream>
#include <memory>
#include <cmath>
#include <random>
#include <string>
#include <utec/algebra/Tensor.h>
#include <utec/nn/neural_network.h>
#include <utec/nn/nn_dense.h>
#include <utec/nn/nn_activation.h>
#include <utec/nn/nn_loss.h>
#include <utec/nn/nn_optimizer.h>

namespace utec {
namespace apps {

using namespace utec::algebra;
using namespace utec::neural_network;

template<typename T>
class SequencePredictor {
private:
    NeuralNetwork<T> nn_;

    // --- Algoritmo de Inicialización Xavier/Glorot ---
    // Complejidad: O(W_layer), donde W_layer es el número de pesos de la capa.
    void init_weights_xavier(Tensor<T, 2>& t) {
        std::random_device rd;
        std::mt19937 gen(rd());
        size_t fan_in = t.shape()[0];
        size_t fan_out = t.shape()[1];
        T limit = std::sqrt(6.0 / (static_cast<T>(fan_in) + static_cast<T>(fan_out)));
        std::uniform_real_distribution<T> dis(-limit, limit);
        for (auto& val : t) {
            val = dis(gen);
        }
    }

    // --- Algoritmo de Inicialización de Bias ---
    // Complejidad: O(B_layer), donde B_layer es el número de bias de la capa.
    void init_bias_zero(Tensor<T, 2>& t) {
        t.fill(T{0});
    }

    // --- Algoritmo de Construcción de Red ---
    // Complejidad: O(W_total), dominado por la inicialización de pesos.
    template <typename InitWFun, typename InitBFun>
    void init_network(InitWFun init_w_fun, InitBFun init_b_fun) {
        // Capa oculta con más capacidad (16 neuronas en lugar de 4)
        nn_.add_layer(std::make_unique<Dense<T>>(
            1, 16, init_w_fun, init_b_fun
        ));
        nn_.add_layer(std::make_unique<ReLU<T>>());

        // CRÍTICO: Capa de salida SIN activación para regresión
        nn_.add_layer(std::make_unique<Dense<T>>(
            16, 1, init_w_fun, init_b_fun
        ));
        // NO agregar ReLU/Sigmoid aquí
    }

public:
    SequencePredictor() {
        // Complejidad: O(W_total)
        init_network(
            [this](Tensor<T, 2>& t){ init_weights_xavier(t); },
            [this](Tensor<T, 2>& t){ init_bias_zero(t); }
        );
    }

    // -----------------------------------------------------------------
    // MÉTODOS DE SERIALIZACIÓN (Portabilidad - Requisito Epic 3)
    // -----------------------------------------------------------------

    // Complejidad: O(W_total)
    void save_weights(const std::string& filepath) const {
        nn_.save_state(filepath);
    }

    // Complejidad: O(W_total)
    void load_weights(const std::string& filepath) {
        nn_.load_state(filepath);
    }

    // --- Algoritmo de Predicción (Forward Propagation) ---
    // Complejidad: O(D * W_total), donde D es el número de muestras.
    utec::algebra::Tensor<T, 2> predict(const utec::algebra::Tensor<T, 2>& X) {
        return nn_.predict(X);
    }

    /**
     * @brief Expone el método de entrenamiento de la red para pruebas.
     */
    // --- Algoritmo de Entrenamiento (Forward, Loss, Backpropagation, Optimizer) ---
    // Complejidad: O(Epochs * D * W_total).
    template<template<typename> class LossType,
             template<typename> class OptimizerType = SGD>
    void train(const utec::algebra::Tensor<T, 2>& X,
               const utec::algebra::Tensor<T, 2>& Y,
               const size_t epochs,
               const size_t batch_size,
               T learning_rate) {
        nn_.template train<LossType, OptimizerType>(X, Y, epochs, batch_size, learning_rate);
    }

    void run_series_experiment() {
        std::cout << "\n--- Ejecutando SequencePredictor (Regresión Simple) ---" << std::endl;

        // --- Algoritmo de Inicialización de Dataset ---
        // Complejidad: O(D) = O(5) = O(1)
        // Datos de entrenamiento: Y = 2X + 1
        Tensor<T, 2> X(5, 1);
        X = {1.0, 2.0, 3.0, 4.0, 5.0};

        Tensor<T, 2> Y(5, 1);
        Y = {3.0, 5.0, 7.0, 9.0, 11.0};

        // HIPERPARÁMETROS OPTIMIZADOS
        size_t epochs = 15000;        
        T learning_rate = 0.01;       
        size_t batch_size = 5;        

        std::cout << "Configuración del entrenamiento:" << std::endl;
        // ... (impresión de configuración omitida) ...
        std::cout << "\nEntrenando..." << std::endl;

        // --- Algoritmo de Entrenamiento (Adam + MSELoss) ---
        // Complejidad: O(Epochs * D * W_total) = O(15000 * 5 * W_total).
        nn_.template train<utec::neural_network::MSELoss, utec::neural_network::Adam>(
            X, Y, epochs, batch_size, learning_rate
        );

        // --- Algoritmo de Predicción (Forward Propagation) ---
        // Complejidad: O(D * W_total) = O(5 * W_total).
        auto predictions = nn_.predict(X);

        std::cout << "\n=== Resultados de Validación (Datos de Entrenamiento) ===" << std::endl;
        // ... (impresión de encabezado omitida) ...
        
        T total_error = 0.0;
        // --- Algoritmo de Cálculo de Error (Validación de Entrenamiento) ---
        // Complejidad: O(D) = O(5) = O(1)
        for(size_t i = 0; i < 5; ++i) {
            T error = std::abs(Y(i, 0) - predictions(i, 0));
            total_error += error;
            // ... (impresión de resultados omitida) ...
        }
        
        T mean_error = total_error / 5.0;
        std::cout << "\nError Promedio en Entrenamiento: " << mean_error << std::endl;

        // --- Algoritmo de Predicción (Prueba de Generalización) ---
        // Complejidad: O(D_test * W_total) = O(2 * W_total).
        // Prueba de generalización con valores no vistos
        Tensor<T, 2> X_test(2, 1);
        X_test = {6.0, 10.0};
        auto test_predictions = nn_.predict(X_test);
        
        std::cout << "\n=== Prueba de Generalización (Datos NO Vistos) ===" << std::endl;
        // ... (impresión de encabezado omitida) ...
        
        // --- Algoritmo de Cálculo de Error (Validación de Generalización) ---
        // Complejidad: O(D_test) = O(2) = O(1)
        T expected_6 = 13.0;  
        T error_6 = std::abs(expected_6 - test_predictions(0, 0));
        // ... (impresión de resultados omitida) ...
        
        T expected_10 = 21.0;  
        T error_10 = std::abs(expected_10 - test_predictions(1, 0));
        // ... (impresión de resultados omitida) ...
        
        T test_mean_error = (error_6 + error_10) / 2.0;
        std::cout << "\nError Promedio en Generalización: " << test_mean_error << std::endl;
        
        // Evaluación de calidad
        // ... (evaluación final omitida) ...
    }
};

} // namespace apps
} // namespace utec

#endif // UTAP_SEQUENCE_PREDICTOR_H
