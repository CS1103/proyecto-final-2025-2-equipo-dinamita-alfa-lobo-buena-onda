#ifndef UTAP_CONTROLLER_DEMO_H
#define UTAP_CONTROLLER_DEMO_H

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
class ControllerDemo {
private:
    NeuralNetwork<T> nn_;
    
    // Funciones de inicialización (replicadas para simplicidad)
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
        // Topología 2-8-1 (2 entradas: Posición, Velocidad; 1 salida: Acción)
        nn_.add_layer(std::make_unique<Dense<T>>(
            2, 8, init_w_fun, init_b_fun
        ));
        nn_.add_layer(std::make_unique<ReLU<T>>());
        nn_.add_layer(std::make_unique<Dense<T>>(
            8, 1, init_w_fun, init_b_fun
        ));
        // Usamos Sigmoid para salida binaria: 0 (Izquierda) o 1 (Derecha)
        nn_.add_layer(std::make_unique<Sigmoid<T>>());
    }

public:
    ControllerDemo() {
        init_network(
            [this](Tensor<T, 2>& t){ init_weights_xavier(t); },
            [this](Tensor<T, 2>& t){ init_bias_zero(t); }
        );
    }

    void run_control_experiment() {
        std::cout << "\n--- Ejecutando ControllerDemo (Control Binario) ---" << std::endl;

        // 1. Dataset de Experto (Clonación de Comportamiento)
        // Regla: Si Posición < 0 (Izquierda), la acción es 1 (Derecha).
        //        Si Posición >= 0 (Derecha), la acción es 0 (Izquierda).
        // El estado del sistema: [Posición, Velocidad]
        Tensor<T, 2> X(8, 2);
        X = {-2.0, 0.5,   // Izquierda -> Derecha (1)
             -1.5, 0.0,   // Izquierda -> Derecha (1)
             -0.1, -0.2,  // Izquierda -> Derecha (1)
              0.0, 0.0,   // Centro    -> Izquierda (0)
              0.1, 0.2,   // Derecha   -> Izquierda (0)
              1.5, -0.5,  // Derecha   -> Izquierda (0)
              2.0, 0.5,   // Derecha   -> Izquierda (0)
             -0.5, 1.0};  // Izquierda -> Derecha (1)

        // Acciones deseadas (Y)
        Tensor<T, 2> Y(8, 1);
        Y = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};

        size_t epochs = 5000;
        T learning_rate = 0.01;
        
        std::cout << "Entrenando la política de control..." << std::endl;

        // Entrenamiento: Adam y BCE (ya que es una decisión binaria)
        nn_.train<BinaryCrossEntropyLoss, Adam>(X, Y, epochs, 8, learning_rate);

        // 2. Validación y Prueba de Generalización
        auto predictions = nn_.predict(X);
        
        std::cout << "\nResultados del Control (Acción Predicha > 0.5 = Derecha (1)):" << std::endl;
        std::cout << "Posicion\tVelocidad\tAcción Esperada\tAcción Predicha" << std::endl;
        for(size_t i = 0; i < 8; ++i) {
            double pred_val = predictions(i, 0);
            int final_action = (pred_val > 0.5) ? 1 : 0;
            
            std::cout << X(i, 0) << "\t\t" << X(i, 1) << "\t\t" 
                      << Y(i, 0) << "\t\t" << final_action << std::endl;
        }

        // Prueba en un nuevo estado (Generalización)
        Tensor<T, 2> X_new(1, 2);
        X_new = {-1.0, -0.1}; // Posición muy a la izquierda -> Debería ir a la Derecha (1)
        auto test_pred = nn_.predict(X_new);
        int test_action = (test_pred(0, 0) > 0.5) ? 1 : 0;
        
        std::cout << "\nPrueba Generalización (Pos: -1.0, Vel: -0.1): Acción = " << test_action << std::endl;
    }
};

} // namespace apps
} // namespace utec

#endif // UTAP_CONTROLLER_DEMO_H