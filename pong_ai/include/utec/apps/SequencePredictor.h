#ifndef UTAP_SEQUENCE_PREDICTOR_H
#define UTAP_SEQUENCE_PREDICTOR_H

#include <iostream>
#include <memory>
#include <cmath>
#include <random>
#include <utec/algebra/tensor.h>
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

    // Funciones de inicialización (Xavier y Zero) - Debes copiarlas en todos los .h de apps o usar un .h de utils
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
        // Topología 1-4-1
        // Entrada: 1 número
        nn_.add_layer(std::make_unique<Dense<T>>(
            1, 4, init_w_fun, init_b_fun
        ));
        nn_.add_layer(std::make_unique<ReLU<T>>());
        // Salida: 1 número (sin activación final para regresión)
        nn_.add_layer(std::make_unique<Dense<T>>(
            4, 1, init_w_fun, init_b_fun
        ));
        // NOTA: No se usa Sigmoid, es una salida lineal/de identidad para regresión.
    }

public:
    SequencePredictor() {
        init_network(
            [this](Tensor<T, 2>& t){ init_weights_xavier(t); },
            [this](Tensor<T, 2>& t){ init_bias_zero(t); }
        );
    }

    void run_series_experiment() {
        std::cout << "\n--- Ejecutando SequencePredictor (Regresión Simple) ---" << std::endl;

        // Datos: Intentamos enseñar a la red la función f(x) = x * 2 + 1
        Tensor<T, 2> X(5, 1);
        X = {1.0, 2.0, 3.0, 4.0, 5.0}; // Entradas
        
        Tensor<T, 2> Y(5, 1);
        Y = {3.0, 5.0, 7.0, 9.0, 11.0}; // Salidas esperadas (X*2 + 1)

        size_t epochs = 5000;
        T learning_rate = 0.005;
        
        std::cout << "Entrenando con Adam y MSELoss..." << std::endl;

        // Entrenamiento: Adam y MSELoss (ideal para regresión)
        nn_.train<MSELoss, Adam>(X, Y, epochs, 5, learning_rate);

        // Validación
        auto predictions = nn_.predict(X);
        
        std::cout << "\nResultados de la validación:" << std::endl;
        std::cout << "Input\tEsperado\tPredicho" << std::endl;
        for(size_t i = 0; i < 5; ++i) {
            std::cout << X(i, 0) << "\t" << Y(i, 0) << "\t\t" << predictions(i, 0) << std::endl;
        }

        // Prueba de Generalización (predicción fuera del rango de entrenamiento)
        Tensor<T, 2> X_test(2, 1);
        X_test = {6.0, 10.0};
        auto test_predictions = nn_.predict(X_test);
        
        std::cout << "\nPrueba de Generalización:" << std::endl;
        std::cout << "Input\tEsperado\tPredicho" << std::endl;
        std::cout << X_test(0, 0) << "\t" << 13.0 << "\t\t" << test_predictions(0, 0) << std::endl;
        std::cout << X_test(1, 0) << "\t" << 21.0 << "\t\t" << test_predictions(1, 0) << std::endl;
    }
};

} // namespace apps
} // namespace utec

#endif // UTAP_SEQUENCE_PREDICTOR_H