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
        // Arquitectura mejorada: 2 -> 16 -> 1
        nn_.add_layer(std::make_unique<Dense<T>>(
            2, 16, init_w_fun, init_b_fun  // Aumentado de 8 a 16
        ));
        nn_.add_layer(std::make_unique<ReLU<T>>());
        
        nn_.add_layer(std::make_unique<Dense<T>>(
            16, 1, init_w_fun, init_b_fun
        ));
        nn_.add_layer(std::make_unique<Sigmoid<T>>());  // Sigmoid para clasificación binaria
    }

public:
    ControllerDemo() {
        init_network(
            [this](Tensor<T, 2>& t){ init_weights_xavier(t); },
            [this](Tensor<T, 2>& t){ init_bias_zero(t); }
        );
    }

    // -----------------------------------------------------------------
    // MÉTODOS DE SERIALIZACIÓN
    // -----------------------------------------------------------------

    void save_weights(const std::string& filepath) const {
        nn_.save_state(filepath);
    }

    void load_weights(const std::string& filepath) {
        nn_.load_state(filepath);
    }

    // -----------------------------------------------------------------
    // MÉTODOS DEL ENTORNO (EnvGym)
    // -----------------------------------------------------------------

    void reset() {
        position_ = 0.0;
        velocity_ = 0.0;
        std::cout << "EnvGym: Reiniciado. Posicion: " << position_ 
                  << ", Velocidad: " << velocity_ << std::endl;
    }

    Tensor<T, 2> get_state() const {
        Tensor<T, 2> state(1, 2);
        state(0, 0) = position_;
        state(0, 1) = velocity_;
        return state;
    }

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

        // Dataset mejorado con más ejemplos y mejor balanceo
        Tensor<T, 2> X(12, 2);
        X = {
            // Casos críticos: muy a la izquierda -> ir a derecha (1)
            -2.0,  0.5,   // Borde izquierdo, moviéndose derecha -> mantener derecha
            -1.8, -0.2,   // Cerca del borde izq, moviéndose izq -> URGENTE derecha
            -1.5,  0.0,   // Lejos izquierda, parado -> ir derecha
            -1.0, -0.3,   // Izquierda, moviéndose más izq -> derecha
            
            // Zona central: depende de velocidad
            -0.5,  0.5,   // Centro-izq, moviéndose derecha -> dejar ir (0)
            -0.2, -0.1,   // Centro-izq, moviéndose izq lento -> compensar derecha (1)
             0.0,  0.0,   // Centro perfecto -> mantener (cualquiera, usemos 0)
             0.2,  0.1,   // Centro-der, moviéndose der lento -> compensar izquierda (0)
             
            // Casos críticos: muy a la derecha -> ir a izquierda (0)
             1.0,  0.3,   // Derecha, moviéndose más der -> izquierda
             1.5,  0.0,   // Lejos derecha, parado -> ir izquierda
             1.8,  0.2,   // Cerca del borde der, moviéndose der -> URGENTE izquierda
             2.0, -0.5    // Borde derecho, moviéndose izq -> mantener izquierda
        };

        // Acciones correctas (0 = izquierda, 1 = derecha)
        Tensor<T, 2> Y(12, 1);
        Y = {
            1.0,  // -2.0,  0.5  -> derecha
            1.0,  // -1.8, -0.2  -> derecha
            1.0,  // -1.5,  0.0  -> derecha
            1.0,  // -1.0, -0.3  -> derecha
            0.0,  // -0.5,  0.5  -> izquierda
            1.0,  // -0.2, -0.1  -> derecha
            0.0,  //  0.0,  0.0  -> izquierda
            0.0,  //  0.2,  0.1  -> izquierda
            0.0,  //  1.0,  0.3  -> izquierda
            0.0,  //  1.5,  0.0  -> izquierda
            0.0,  //  1.8,  0.2  -> izquierda
            0.0   //  2.0, -0.5  -> izquierda
        };

        std::cout << "Configuración:" << std::endl;
        std::cout << "  - Épocas: " << epochs << std::endl;
        std::cout << "  - Learning Rate: " << learning_rate << std::endl;
        std::cout << "  - Muestras de entrenamiento: 12" << std::endl;
        std::cout << "  - Optimizador: Adam" << std::endl;
        std::cout << "  - Loss: Binary Cross-Entropy" << std::endl;
        std::cout << "\nEntrenando la política de control..." << std::endl;

        // Entrenamiento con hiperparámetros mejorados
        nn_.template train<utec::neural_network::BinaryCrossEntropyLoss, 
                          utec::neural_network::Adam>(
            X, Y, epochs, 4, learning_rate
        );

        // Validación
        auto predictions = nn_.predict(X);

        std::cout << "\n=== Resultados del Control (threshold = 0.5) ===" << std::endl;
        std::cout << "Pos\tVel\tEsperado\tPredicción\tAcción\tCorrect" << std::endl;
        std::cout << "---\t---\t--------\t----------\t------\t-------" << std::endl;
        
        int correct = 0;
        for(size_t i = 0; i < 12; ++i) {
            T pred_val = predictions(i, 0);
            int predicted_action = (pred_val > 0.5) ? 1 : 0;
            int expected_action = static_cast<int>(Y(i, 0) + 0.5);
            bool is_correct = (predicted_action == expected_action);
            if (is_correct) correct++;

            std::cout << X(i, 0) << "\t" << X(i, 1) << "\t"
                      << expected_action << "\t\t" << pred_val << "\t"
                      << predicted_action << "\t"
                      << (is_correct ? "✓" : "✗") << std::endl;
        }
        
        T accuracy = (static_cast<T>(correct) / 12.0) * 100.0;
        std::cout << "\nAccuracy en entrenamiento: " << correct << "/12 (" 
                  << accuracy << "%)" << std::endl;

        // Prueba de generalización
        Tensor<T, 2> X_test(3, 2);
        X_test = {
            -1.2, -0.15,  // Izquierda, moviéndose más izq -> debería ir derecha (1)
             0.5,  0.3,   // Derecha, moviéndose más der -> debería ir izquierda (0)
            -0.8,  0.1    // Izquierda, moviéndose derecha -> debería mantener/ir izq (0)
        };
        
        auto test_pred = nn_.predict(X_test);
        
        std::cout << "\n=== Pruebas de Generalización ===" << std::endl;
        std::cout << "Pos\tVel\tPredicción\tAcción\tEsperado" << std::endl;
        std::cout << "---\t---\t----------\t------\t--------" << std::endl;
        
        std::array<int, 3> expected_test = {1, 0, 0};
        for(size_t i = 0; i < 3; ++i) {
            T pred_val = test_pred(i, 0);
            int action = (pred_val > 0.5) ? 1 : 0;
            std::cout << X_test(i, 0) << "\t" << X_test(i, 1) << "\t"
                      << pred_val << "\t" << action << "\t"
                      << expected_test[i] 
                      << (action == expected_test[i] ? " ✓" : " ✗") << std::endl;
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