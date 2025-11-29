#ifndef UTAP_PATTERN_CLASSIFIER_H
#define UTAP_PATTERN_CLASSIFIER_H

#include <iostream>
#include <memory>
#include <cmath>
#include <random>
#include <string> // Necesario para std::string en save/load
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
class PatternClassifier {
private:
    NeuralNetwork<T> nn_;

    template <typename InitWFun, typename InitBFun>
    void init_network(InitWFun init_w_fun, InitBFun init_b_fun) {
        // Topología 2-4-1 (Entrada: 2, Capa Oculta: 4, Salida: 1)
        nn_.add_layer(std::make_unique<Dense<T>>(
            2, 4, init_w_fun, init_b_fun
        ));
        nn_.add_layer(std::make_unique<ReLU<T>>());
        nn_.add_layer(std::make_unique<Dense<T>>(
            4, 1, init_w_fun, init_b_fun
        ));
        nn_.add_layer(std::make_unique<Sigmoid<T>>());
    }

    // Funciones de inicialización
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

public:
    PatternClassifier() {
        init_network(
            [this](Tensor<T, 2>& t){ init_weights_xavier(t); },
            [this](Tensor<T, 2>& t){ init_bias_zero(t); }
        );
    }

    // -----------------------------------------------------------------
    // MÉTODOS DE SERIALIZACIÓN (Portabilidad - Requisito Epic 3)
    // -----------------------------------------------------------------

    /**
     * @brief Guarda los pesos entrenados de la red en un archivo binario.
     */
    void save_weights(const std::string& filepath) const {
        nn_.save_state(filepath);
    }

    /**
     * @brief Carga los pesos de un archivo para restaurar una red previamente entrenada.
     */
    void load_weights(const std::string& filepath) {
        nn_.load_state(filepath);
    }

    /**
     * @brief Realiza una predicción con la red neuronal.
     */
    utec::algebra::Tensor<T, 2> predict(const utec::algebra::Tensor<T, 2>& X) {
        return nn_.predict(X);
    }

    // -----------------------------------------------------------------
    // LÓGICA DE ENTRENAMIENTO (Existente, con corrección)
    // -----------------------------------------------------------------

    void run_xor_experiment() {
        std::cout << "--- Ejecutando PatternClassifier (XOR) ---" << std::endl;

        // Datos XOR
        Tensor<T, 2> X(4, 2);
        X = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};
        Tensor<T, 2> Y(4, 1);
        Y = {0.0, 1.0, 1.0, 0.0};

        size_t epochs = 10000;
        T learning_rate = 0.01;

        std::cout << "Entrenando con Adam..." << std::endl;

        // Entrenamiento: Adam y BCE
        nn_.template train<utec::neural_network::BinaryCrossEntropyLoss, utec::neural_network::Adam>(X, Y, epochs, 4, learning_rate);

        // Validación
        auto predictions = nn_.predict(X);

        std::cout << "\nResultados de Validación:" << std::endl;
        std::cout << "Input\t\tEsperado\tClase Predicha" << std::endl;
        for(size_t i = 0; i < 4; ++i) {
            // LÍNEA CORREGIDA: Acceso al valor de la predicción
            double pred_val = predictions(i, 0);
            int final_class = (pred_val > 0.5) ? 1 : 0;

            std::cout << "(" << X(i, 0) << "," << X(i, 1) << ")\t"
                      << Y(i, 0) << "\t\t" << final_class << std::endl;
        }
    }
};

} // namespace apps
} // namespace utec

#endif // UTAP_PATTERN_CLASSIFIER_H