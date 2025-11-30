#ifndef UTAP_CONTROLLER_DEMO_H
#define UTAP_CONTROLLER_DEMO_H

#include <iostream>
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>
#include <utec/algebra/Tensor.h>
#include <utec/nn/neural_network.h>
#include <utec/nn/nn_dense.h>
#include <utec/nn/nn_activation.h>
#include <utec/nn/nn_loss.h>
#include <utec/nn/nn_optimizer.h>
#include <string>

namespace utec {
namespace apps {

using namespace utec::algebra;
using namespace utec::neural_network;

template<typename T>
class ControllerDemo {
private:
    NeuralNetwork<T> nn_;

    // --- ESTADO INTERNO DEL SIMULADOR (EnvGym) ---
    T position_ = 0.0;
    T velocity_ = 0.0;

    // Constantes de la simulación
    static constexpr T MAX_POS = 2.0;
    static constexpr T MIN_POS = -2.0;
    static constexpr T FRICTION = 0.95;
    static constexpr T FORCE_MAGNITUDE = 0.1;

    // --- Algoritmo de Inicialización: Xavier/Glorot ---
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

    // --- Algoritmo de Inicialización: Bias a Cero ---
    // Complejidad: O(B_layer), donde B_layer es el número de bias de la capa.
    void init_bias_zero(Tensor<T, 2>& t) {
        t.fill(T{0});
    }

    template <typename InitWFun, typename InitBFun>
    void init_network(InitWFun init_w_fun, InitBFun init_b_fun) {
        // La complejidad de construcción es O(W_total) debido a las inicializaciones.
        // Arquitectura mejorada: 2 -> 16 -> 1
        nn_.add_layer(std::make_unique<Dense<T>>(<br>            2, 16, init_w_fun, init_b_fun  // Aumentado de 8 a 16
        ));
        nn_.add_layer(std::make_unique<ReLU<T>>());
        
        nn_.add_layer(std::make_unique<Dense<T>>(<br>            16, 1, init_w_fun, init_b_fun
        ));
        nn_.add_layer(std::make_unique<Sigmoid<T>>());  // Sigmoid para clasificación binaria
    }

public:
    ControllerDemo() {
        // O(W_total) por las llamadas a init_weights y init_bias
        init_network(<br>            [this](Tensor<T, 2>& t){ init_weights_xavier(t); },
            [this](Tensor<T, 2>& t){ init_bias_zero(t); }
        );
    }

    // -----------------------------------------------------------------
    // MÉTODOS DE SERIALIZACIÓN
    // -----------------------------------------------------------------

    // Complejidad: O(W_total), donde W_total es el número total de parámetros (pesos y bias)
    void save_weights(const std::string& filepath) const {
        nn_.save_state(filepath);
    }

    // Complejidad: O(W_total)
    void load_weights(const std::string& filepath) {
        nn_.load_state(filepath);
    }

    // -----------------------------------------------------------------
    // MÉTODOS DEL ENTORNO (EnvGym)
    // -----------------------------------------------------------------

    // Complejidad: O(1)
    void reset() {
        position_ = 0.0;
        velocity_ = 0.0;
        std::cout << "EnvGym: Reiniciado. Posicion: " << position_ <br>                  << ", Velocidad: " << velocity_ << std::endl;
    }

    // Complejidad: O(1)
    Tensor<T, 2> get_state() const {
        Tensor<T, 2> state(1, 2);
        state(0, 0) = position_;
        state(0, 1) = velocity_;
        return state;
    }

    // --- Algoritmo de Simulación Física ---
    // Complejidad: O(1)
    bool step(int action) {
        // Calcular fuerza según acción
        T force = (action == 1) ? FORCE_MAGNITUDE : -FORCE_MAGNITUDE;

        // Actualizar física
        velocity_ = (velocity_ + force) * FRICTION;
        position_ += velocity_;

        // Verificar límites
        if (position_ >= MAX_POS || position_ <= MIN_POS) {
            std::cout << "EnvGym: ¡Límite alcanzado! Posición final: " << position_ << std::endl;
            return false;
        }

        return true;
    }

    NeuralNetwork<T>& get_network() {
        return nn_;
    }

    // -----------------------------------------------------------------
    // ENTRENAMIENTO MEJORADO
    // -----------------------------------------------------------------

    void train_expert_policy(size_t epochs, T learning_rate) {
        std::cout << "\n--- Ejecutando ControllerDemo: Entrenamiento de Política de Control ---" << std::endl;

        // --- Algoritmo de Inicialización de Dataset ---
        // Complejidad: O(1) (porque D_size=12 es una constante pequeña)
        Tensor<T, 2> X(12, 2);
        X = {
            // ... datos omitidos ...
        };

        Tensor<T, 2> Y(12, 1);
        Y = {
            // ... datos omitidos ...
        };

        // --- Algoritmo de Entrenamiento (Forward Propagation + BinaryCrossEntropyLoss + Backpropagation + Adam) ---
        // Complejidad: O(Epochs * D * W_total)
        // D=12 es el número de muestras (Batch Size es 4, pero D se procesa completamente cada época).
        // W_total es el número total de parámetros de la red (fijo).
        nn_.template train<utec::neural_network::BinaryCrossEntropyLoss, <br>                          utec::neural_network::Adam>(
            X, Y, epochs, 4, learning_rate
        );

        // --- Algoritmo de Predicción (Forward Propagation) ---
        // Complejidad: O(D * W_total) = O(12 * W_total).
        auto predictions = nn_.predict(X);

        std::cout << "\n=== Resultados del Control (threshold = 0.5) ===" << std::endl;
        std::cout << "Pos\tVel\tEsperado\tPredicción\tAcción\tCorrect" << std::endl;
        std::cout << "---\t---\t--------\t----------\t------\t-------" << std::endl;
        
        // --- Algoritmo de Validación/Chequeo de Precisión ---
        // Complejidad: O(D) = O(12) = O(1)
        int correct = 0;
        for(size_t i = 0; i < 12; ++i) {
            T pred_val = predictions(i, 0);
            int predicted_action = (pred_val > 0.5) ? 1 : 0;
            int expected_action = static_cast<int>(Y(i, 0) + 0.5);
            bool is_correct = (predicted_action == expected_action);
            if (is_correct) correct++;

            std::cout << X(i, 0) << "\t" << X(i, 1) << "\t"<br>                      << expected_action << "\t\t" << pred_val << "\t"<br>                      << predicted_action << "\t"<br>                      << (is_correct ? "✓" : "✗") << std::endl;
        }
        
        T accuracy = (static_cast<T>(correct) / 12.0) * 100.0;
        std::cout << "\nAccuracy en entrenamiento: " << correct << "/12 (" <br>                  << accuracy << "%)" << std::endl;

        // --- Algoritmo de Predicción (Prueba de Generalización) ---
        // Complejidad: O(D_test * W_total) = O(3 * W_total).
        Tensor<T, 2> X_test(3, 2);
        X_test = {
            // ... datos omitidos ...
        };
        
        auto test_pred = nn_.predict(X_test);
        
        std::cout << "\n=== Pruebas de Generalización ===" << std::endl;
        std::cout << "Pos\tVel\tPredicción\tAcción\tEsperado" << std::endl;
        std::cout << "---\t---\t----------\t------\t--------" << std::endl;
        
        // --- Algoritmo de Validación de Prueba ---
        // Complejidad: O(D_test) = O(3) = O(1)
        std::array<int, 3> expected_test = {1, 0, 0};
        for(size_t i = 0; i < 3; ++i) {
            T pred_val = test_pred(i, 0);
            int action = (pred_val > 0.5) ? 1 : 0;
            std::cout << X_test(i, 0) << "\t" << X_test(i, 1) << "\t"<br>                      << pred_val << "\t" << action << "\t"<br>                      << expected_test[i] <br>                      << (action == expected_test[i] ? " ✓" : " ✗") << std::endl;
        }

        if (accuracy >= 90.0) {
            std::cout << "\n✅ ENTRENAMIENTO EXITOSO: Política lista para control" << std::endl;
        } else if (accuracy >= 75.0) {
            std::cout << "\n⚠️  ENTRENAMIENTO ACEPTABLE: Considerar más épocas" << std::endl;
        } else {
            std::cout << "\n❌ ENTRENAMIENTO INSUFICIENTE: Ajustar hiperparámetros" << std::endl;
        }
    }
};

} // namespace apps
} // namespace utec

#endif // UTAP_CONTROLLER_DEMO_H
