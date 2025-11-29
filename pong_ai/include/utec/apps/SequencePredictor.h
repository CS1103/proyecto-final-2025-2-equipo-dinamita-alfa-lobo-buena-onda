#ifndef UTAP_SEQUENCE_PREDICTOR_H
#define UTAP_SEQUENCE_PREDICTOR_H

#include <iostream>
#include <memory>
#include <cmath>
#include <random>
#include <string>
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

        nn_.add_layer(std::make_unique<Dense<T>>(
            1, 4, init_w_fun, init_b_fun
        ));
        nn_.add_layer(std::make_unique<ReLU<T>>());

        nn_.add_layer(std::make_unique<Dense<T>>(
            4, 1, init_w_fun, init_b_fun
        ));
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

    void run_series_experiment() {
        std::cout << "\n--- Ejecutando SequencePredictor (Regresión Simple) ---" << std::endl;

        Tensor<T, 2> X(5, 1);
        X = {1.0, 2.0, 3.0, 4.0, 5.0};

        Tensor<T, 2> Y(5, 1);
        Y = {3.0, 5.0, 7.0, 9.0, 11.0};

        size_t epochs = 5000;
        T learning_rate = 0.005;

        std::cout << "Entrenando con Adam y MSELoss..." << std::endl;

        nn_.template train<utec::neural_network::MSELoss, utec::neural_network::Adam>(X, Y, epochs, 5, learning_rate);

        auto predictions = nn_.predict(X);

        std::cout << "\nResultados de la validación:" << std::endl;
        std::cout << "Input\tEsperado\tPredicho" << std::endl;
        for(size_t i = 0; i < 5; ++i) {
            std::cout << X(i, 0) << "\t" << Y(i, 0) << "\t\t" << predictions(i, 0) << std::endl;
        }

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