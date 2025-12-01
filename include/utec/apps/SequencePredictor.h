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

    void init_bias_zero(Tensor<T, 2>& t) {
        t.fill(T{0});
    }

    template <typename InitWFun, typename InitBFun>
    void init_network(InitWFun init_w_fun, InitBFun init_b_fun) {
        // Capa oculta con más capacidad (16 neuronas en lugar de 4)
        nn_.add_layer(std::make_unique<Dense<T>>(
            1, 16, init_w_fun, init_b_fun
        ));
        nn_.add_layer(std::make_unique<ReLU<T>>());

        // CRÍTICO: Capa de salida SIN activación para regresión
        // Necesitamos valores reales ilimitados, no solo positivos
        nn_.add_layer(std::make_unique<Dense<T>>(
            16, 1, init_w_fun, init_b_fun
        ));
        // NO agregar ReLU/Sigmoid aquí - limitaría el rango de salida
    }

public:
    SequencePredictor() {
        init_network(
            [this](Tensor<T, 2>& t){ init_weights_xavier(t); },
            [this](Tensor<T, 2>& t){ init_bias_zero(t); }
        );
    }

    // -----------------------------------------------------------------
    // MÉTODOS DE SERIALIZACIÓN (Portabilidad - Requisito Epic 3)
    // -----------------------------------------------------------------

    void save_weights(const std::string& filepath) const {
        nn_.save_state(filepath);
    }

    void load_weights(const std::string& filepath) {
        nn_.load_state(filepath);
    }

    utec::algebra::Tensor<T, 2> predict(const utec::algebra::Tensor<T, 2>& X) {
        return nn_.predict(X);
    }

    /**
     * @brief Expone el método de entrenamiento de la red para pruebas.
     */
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

        // Datos de entrenamiento: Y = 2X + 1
        Tensor<T, 2> X(5, 1);
        X = {1.0, 2.0, 3.0, 4.0, 5.0};

        Tensor<T, 2> Y(5, 1);
        Y = {3.0, 5.0, 7.0, 9.0, 11.0};

        // HIPERPARÁMETROS OPTIMIZADOS
        size_t epochs = 15000;        // Aumentado para mejor convergencia
        T learning_rate = 0.01;       // Duplicado para convergencia más rápida
        size_t batch_size = 5;        // Batch completo (todo el dataset)

        std::cout << "Configuración del entrenamiento:" << std::endl;
        std::cout << "  - Épocas: " << epochs << std::endl;
        std::cout << "  - Learning Rate: " << learning_rate << std::endl;
        std::cout << "  - Batch Size: " << batch_size << std::endl;
        std::cout << "  - Optimizador: Adam" << std::endl;
        std::cout << "  - Loss: MSE (Mean Squared Error)" << std::endl;
        std::cout << "\nEntrenando..." << std::endl;

        nn_.template train<utec::neural_network::MSELoss, utec::neural_network::Adam>(
            X, Y, epochs, batch_size, learning_rate
        );

        auto predictions = nn_.predict(X);

        std::cout << "\n=== Resultados de Validación (Datos de Entrenamiento) ===" << std::endl;
        std::cout << "Input\tEsperado\tPredicho\tError Absoluto" << std::endl;
        std::cout << "-----\t--------\t--------\t--------------" << std::endl;
        
        T total_error = 0.0;
        for(size_t i = 0; i < 5; ++i) {
            T error = std::abs(Y(i, 0) - predictions(i, 0));
            total_error += error;
            std::cout << X(i, 0) << "\t" << Y(i, 0) << "\t\t" 
                      << predictions(i, 0) << "\t\t" << error << std::endl;
        }
        
        T mean_error = total_error / 5.0;
        std::cout << "\nError Promedio en Entrenamiento: " << mean_error << std::endl;

        // Prueba de generalización con valores no vistos
        Tensor<T, 2> X_test(2, 1);
        X_test = {6.0, 10.0};
        auto test_predictions = nn_.predict(X_test);
        
        std::cout << "\n=== Prueba de Generalización (Datos NO Vistos) ===" << std::endl;
        std::cout << "Input\tEsperado\tPredicho\tError Absoluto" << std::endl;
        std::cout << "-----\t--------\t--------\t--------------" << std::endl;
        
        T expected_6 = 13.0;  // f(6) = 2*6 + 1 = 13
        T error_6 = std::abs(expected_6 - test_predictions(0, 0));
        std::cout << X_test(0, 0) << "\t" << expected_6 << "\t\t" 
                  << test_predictions(0, 0) << "\t\t" << error_6 << std::endl;
        
        T expected_10 = 21.0;  // f(10) = 2*10 + 1 = 21
        T error_10 = std::abs(expected_10 - test_predictions(1, 0));
        std::cout << X_test(1, 0) << "\t" << expected_10 << "\t\t" 
                  << test_predictions(1, 0) << "\t\t" << error_10 << std::endl;
        
        T test_mean_error = (error_6 + error_10) / 2.0;
        std::cout << "\nError Promedio en Generalización: " << test_mean_error << std::endl;
        
        // Evaluación de calidad
        if (mean_error < 0.5 && test_mean_error < 1.0) {
            std::cout << "\n✅ ENTRENAMIENTO EXITOSO: La red aprendió la función Y = 2X + 1" << std::endl;
        } else if (mean_error < 1.0 && test_mean_error < 2.0) {
            std::cout << "\n⚠️  ENTRENAMIENTO ACEPTABLE: La red necesita más épocas o ajuste de hiperparámetros" << std::endl;
        } else {
            std::cout << "\n❌ ENTRENAMIENTO INSUFICIENTE: La red no ha convergido correctamente" << std::endl;
            std::cout << "   Sugerencias:" << std::endl;
            std::cout << "   - Aumentar épocas a 20000-30000" << std::endl;
            std::cout << "   - Ajustar learning rate (probar 0.02 o 0.005)" << std::endl;
            std::cout << "   - Verificar implementación del optimizador Adam" << std::endl;
        }
    }
};

} // namespace apps
} // namespace utec

#endif // UTAP_SEQUENCE_PREDICTOR_H