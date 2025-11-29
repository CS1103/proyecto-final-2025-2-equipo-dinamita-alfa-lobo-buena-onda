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
    // Estas variables definen el estado actual del "carrito" en el simulador.
    T position_ = 0.0;
    T velocity_ = 0.0;

    // Constantes de la simulación
    static constexpr T MAX_POS = 2.0;    // Límite de posición
    static constexpr T MIN_POS = -2.0;
    static constexpr T FRICTION = 0.95;  // Factor de fricción
    static constexpr T FORCE_MAGNITUDE = 0.1; // Magnitud de la fuerza aplicada

    // Funciones de inicialización (del código anterior)
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
        // Topología 2-8-1
        nn_.add_layer(std::make_unique<Dense<T>>(
            2, 8, init_w_fun, init_b_fun
        ));
        nn_.add_layer(std::make_unique<ReLU<T>>());
        nn_.add_layer(std::make_unique<Dense<T>>(
            8, 1, init_w_fun, init_b_fun
        ));
        nn_.add_layer(std::make_unique<Sigmoid<T>>());
    }

public:
    ControllerDemo() {
        init_network(
            [this](Tensor<T, 2>& t){ init_weights_xavier(t); },
            [this](Tensor<T, 2>& t){ init_bias_zero(t); }
        );
    }

    // -----------------------------------------------------------------
    // 1. MÉTODOS DE SERIALIZACIÓN (Portabilidad - Requisito Epic 3)
    // -----------------------------------------------------------------

    /**
     * @brief Guarda los pesos entrenados de la red en un archivo.
     * Este método asume que NeuralNetwork<T> implementa save_state.
     */
    void save_weights(const std::string& filepath) const {
        // Debes implementar nn_.save_state(filepath) en neural_network.h/cpp
        nn_.save_state(filepath);
    }

    /**
     * @brief Carga los pesos de un archivo para restaurar una red previamente entrenada.
     * Este método asume que NeuralNetwork<T> implementa load_state.
     */
    void load_weights(const std::string& filepath) {
        // Debes implementar nn_.load_state(filepath) en neural_network.h/cpp
        nn_.load_state(filepath);
    }

    // -----------------------------------------------------------------
    // 2. MÉTODOS DEL ENTORNO DE PRUEBA (EnvGym - Requisito Epic 3)
    // -----------------------------------------------------------------

    /**
     * @brief Reinicia el entorno a un estado inicial (el EnvGym).
     */
    void reset() {
        position_ = 0.0; // Posición inicial en el centro
        velocity_ = 0.0; // Velocidad inicial cero
        std::cout << "EnvGym: Reiniciado. Posicion: " << position_ << ", Velocidad: " << velocity_ << std::endl;
    }

    /**
     * @brief Devuelve el estado actual del simulador (Input para la NN).
     * @return Tensor<T, 2> con [posición, velocidad].
     */
    Tensor<T, 2> get_state() const {
        Tensor<T, 2> state(1, 2); // Un tensor (1 fila, 2 columnas)
        state(0, 0) = position_;
        state(0, 1) = velocity_;
        return state;
    }

    /**
     * @brief Ejecuta la acción del agente y avanza el simulador un paso (step).
     * @param action La acción decidida por la red (0: Izquierda, 1: Derecha).
     * @return true si la simulación continúa, false si se alcanzó el límite.
     */
    bool step(int action) {
        // 1. Calcular la fuerza basada en la acción de la NN
        T force = (action == 1) ? FORCE_MAGNITUDE : -FORCE_MAGNITUDE;

        // 2. Actualizar la velocidad (fuerza y fricción)
        velocity_ = (velocity_ + force) * FRICTION;

        // 3. Actualizar la posición
        position_ += velocity_;

        // 4. Aplicar límites y verificar fin de simulación
        if (position_ >= MAX_POS || position_ <= MIN_POS) {
             // El sistema se salió de los límites (falla el control)
             std::cout << "EnvGym: ¡Límite alcanzado! Simulación terminada." << std::endl;
             return false;
        }

        return true;
    }

    /**
     * @brief Método para obtener la referencia a la red neuronal (útil para predict/forward).
     */
    NeuralNetwork<T>& get_network() {
        return nn_;
    }

    // -----------------------------------------------------------------
    // 3. MÉTODO DE ENTRENAMIENTO (Lógica existente, renombrado)
    // -----------------------------------------------------------------

    void train_expert_policy(size_t epochs, T learning_rate) {
        std::cout << "\n--- Ejecutando ControllerDemo: Entrenamiento de Política de Control ---" << std::endl;

        // 1. Dataset de Experto (Clonación de Comportamiento)
        Tensor<T, 2> X(8, 2);
        X = {-2.0, 0.5, -1.5, 0.0, -0.1, -0.2, 0.0, 0.0, 0.1, 0.2, 1.5, -0.5, 2.0, 0.5, -0.5, 1.0};

        // Acciones deseadas (Y)
        Tensor<T, 2> Y(8, 1);
        Y = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};

        std::cout << "Entrenando la política de control..." << std::endl;

        // Entrenamiento
        nn_.template train<utec::neural_network::BinaryCrossEntropyLoss, utec::neural_network::Adam>(X, Y, epochs, 4, learning_rate);

        // 2. Validación de resultados (Generalización)
        auto predictions = nn_.predict(X);

        std::cout << "\nResultados del Control (Acción Predicha > 0.5 = Derecha (1)):" << std::endl;
        std::cout << "Posicion\tVelocidad\tAcción Esperada\tAcción Predicha" << std::endl;
        for(size_t i = 0; i < 8; ++i) {
            double pred_val = predictions(i, 0);
            int final_action = (pred_val > 0.5) ? 1 : 0;

            std::cout << X(i, 0) << "\t\t" << X(i, 1) << "\t\t"
                      << Y(i, 0) << "\t\t" << final_action << std::endl;
        }

        Tensor<T, 2> X_new(1, 2);
        X_new = {-1.0, -0.1};
        auto test_pred = nn_.predict(X_new);
        int test_action = (test_pred(0, 0) > 0.5) ? 1 : 0;
        
        std::cout << "\nPrueba Generalización (Pos: -1.0, Vel: -0.1): Acción = " << test_action << std::endl;
    }
};

} // namespace apps
} // namespace utec

#endif // UTAP_CONTROLLER_DEMO_H