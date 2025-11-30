#ifndef UTAP_PATTERN_CLASSIFIER_H
#define UTAP_PATTERN_CLASSIFIER_H

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
class PatternClassifier {
private:
    NeuralNetwork<T> nn_;

    // --- Algoritmo de Construcción de Red ---
    // Complejidad: O(W_total), dominado por la inicialización de pesos.
    template <typename InitWFun, typename InitBFun>
    void init_network(InitWFun init_w_fun, InitBFun init_b_fun) {
        // Arquitectura mejorada: 2 -> 8 -> 8 -> 1
        nn_.add_layer(std::make_unique<Dense<T>>(
            2, 8, init_w_fun, init_b_fun  // Primera capa oculta
        ));
        nn_.add_layer(std::make_unique<ReLU<T>>());
        
        nn_.add_layer(std::make_unique<Dense<T>>(
            8, 8, init_w_fun, init_b_fun  // Segunda capa oculta
        ));
        nn_.add_layer(std::make_unique<ReLU<T>>());
        
        nn_.add_layer(std::make_unique<Dense<T>>(
            8, 1, init_w_fun, init_b_fun  // Capa de salida
        ));
        nn_.add_layer(std::make_unique<Sigmoid<T>>());  // Sigmoid para clasificación binaria
    }

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

public:
    PatternClassifier() {
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

    // -----------------------------------------------------------------
    // LÓGICA DE ENTRENAMIENTO (XOR - Problema No Lineal)
    // -----------------------------------------------------------------

    void run_xor_experiment() {
        std::cout << "\n--- Ejecutando PatternClassifier (XOR) ---" << std::endl;

        // --- Algoritmo de Inicialización de Dataset ---
        // Complejidad: O(D) = O(4) = O(1)
        Tensor<T, 2> X(4, 2);
        X = {0.0, 0.0,   // XOR(0,0) = 0
             0.0, 1.0,   // XOR(0,1) = 1
             1.0, 0.0,   // XOR(1,0) = 1
             1.0, 1.0};  // XOR(1,1) = 0

        Tensor<T, 2> Y(4, 1);
        Y = {0.0, 1.0, 1.0, 0.0};

        // HIPERPARÁMETROS OPTIMIZADOS PARA XOR
        size_t epochs = 20000;      
        T learning_rate = 0.05;     
        size_t batch_size = 4;      

        std::cout << "Configuración del entrenamiento:" << std::endl;
        // ... (impresión de configuración omitida) ...
        std::cout << "\nEntrenando con Adam..." << std::endl;

        // --- Algoritmo de Entrenamiento (Adam + Binary Cross-Entropy) ---
        // Complejidad: O(Epochs * D * W_total) = O(20000 * 4 * W_total).
        nn_.template train<utec::neural_network::BinaryCrossEntropyLoss, 
                          utec::neural_network::Adam>(
            X, Y, epochs, batch_size, learning_rate
        );

        // --- Algoritmo de Predicción (Forward Propagation) ---
        // Complejidad: O(D * W_total) = O(4 * W_total).
        auto predictions = nn_.predict(X);

        std::cout << "\n=== Resultados de Validación (Threshold = 0.5) ===" << std::endl;
        // ... (impresión de encabezado omitida) ...
        
        // --- Algoritmo de Validación de Precisión ---
        // Complejidad: O(D) = O(4) = O(1)
        int correct = 0;
        for(size_t i = 0; i < 4; ++i) {
            T pred_val = predictions(i, 0);
            int predicted_class = (pred_val > 0.5) ? 1 : 0;
            int expected_class = static_cast<int>(Y(i, 0) + 0.5);
            bool is_correct = (predicted_class == expected_class);
            
            if (is_correct) correct++;

            // ... (impresión de resultados omitida) ...
        }
        
        T accuracy = (static_cast<T>(correct) / 4.0) * 100.0;
        // ... (impresión de resumen omitida) ...

        // --- Algoritmo de Predicción (Prueba de Robustez) ---
        // Complejidad: O(D_noisy * W_total) = O(4 * W_total).
        std::cout << "\n=== Prueba de Robustez (Inputs con Ruido) ===" << std::endl;
        Tensor<T, 2> X_noisy(4, 2);
        X_noisy = {
            0.05, 0.02,   
            0.02, 0.95,   
            0.98, 0.03,   
            0.97, 0.96    
        };
        
        auto noisy_predictions = nn_.predict(X_noisy);
        
        // ... (impresión de encabezado omitida) ...
        
        std::array<int, 4> expected_noisy = {0, 1, 1, 0};
        int robust_correct = 0;
        
        // --- Algoritmo de Validación de Robustez ---
        // Complejidad: O(D_noisy) = O(4) = O(1)
        for(size_t i = 0; i < 4; ++i) {
            T pred_val = noisy_predictions(i, 0);
            int predicted_class = (pred_val > 0.5) ? 1 : 0;
            bool is_correct = (predicted_class == expected_noisy[i]);
            
            if (is_correct) robust_correct++;
            
            // ... (impresión de resultados omitida) ...
        }
        
        // ... (impresión de robustez y evaluación final omitida) ...
    }
};

} // namespace apps
} // namespace utec

#endif // UTAP_PATTERN_CLASSIFIER_H
