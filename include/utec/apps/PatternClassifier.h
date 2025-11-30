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

    template <typename InitWFun, typename InitBFun>
    void init_network(InitWFun init_w_fun, InitBFun init_b_fun) {
        // Arquitectura mejorada: 2 -> 8 -> 8 -> 1
        // XOR requiere al menos 2 neuronas ocultas, pero más es mejor para robustez
        nn_.add_layer(std::make_unique<Dense<T>>(
            2, 8, init_w_fun, init_b_fun  // Primera capa oculta
        ));
        nn_.add_layer(std::make_unique<ReLU<T>>());
        
        nn_.add_layer(std::make_unique<Dense<T>>(
            8, 8, init_w_fun, init_b_fun  // Segunda capa oculta
        ));
        nn_.add_layer(std::make_unique<ReLU<T>>());
        
        nn_.add_layer(std::make_unique<Dense<T>>(
            8, 1, init_w_fun, init_b_fun  // Capa de salida
        ));
        nn_.add_layer(std::make_unique<Sigmoid<T>>());  // Sigmoid para clasificación binaria
    }

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

    // -----------------------------------------------------------------
    // LÓGICA DE ENTRENAMIENTO (XOR - Problema No Lineal)
    // -----------------------------------------------------------------

    void run_xor_experiment() {
        std::cout << "\n--- Ejecutando PatternClassifier (XOR) ---" << std::endl;

        // Datos XOR clásicos
        Tensor<T, 2> X(4, 2);
        X = {0.0, 0.0,   // XOR(0,0) = 0
             0.0, 1.0,   // XOR(0,1) = 1
             1.0, 0.0,   // XOR(1,0) = 1
             1.0, 1.0};  // XOR(1,1) = 0

        Tensor<T, 2> Y(4, 1);
        Y = {0.0, 1.0, 1.0, 0.0};

        // HIPERPARÁMETROS OPTIMIZADOS PARA XOR
        size_t epochs = 20000;      // Aumentado significativamente
        T learning_rate = 0.05;     // Learning rate más alto para convergencia rápida
        size_t batch_size = 4;      // Batch completo

        std::cout << "Configuración del entrenamiento:" << std::endl;
        std::cout << "  - Problema: XOR (No Lineal)" << std::endl;
        std::cout << "  - Arquitectura: 2 -> 8 -> 8 -> 1" << std::endl;
        std::cout << "  - Épocas: " << epochs << std::endl;
        std::cout << "  - Learning Rate: " << learning_rate << std::endl;
        std::cout << "  - Batch Size: " << batch_size << std::endl;
        std::cout << "  - Optimizador: Adam" << std::endl;
        std::cout << "  - Loss: Binary Cross-Entropy" << std::endl;
        std::cout << "\nEntrenando con Adam..." << std::endl;

        // Entrenamiento
        nn_.template train<utec::neural_network::BinaryCrossEntropyLoss, 
                          utec::neural_network::Adam>(
            X, Y, epochs, batch_size, learning_rate
        );

        // Validación detallada
        auto predictions = nn_.predict(X);

        std::cout << "\n=== Resultados de Validación (Threshold = 0.5) ===" << std::endl;
        std::cout << "Input\t\tEsperado\tPredicción\tClase\tResultado" << std::endl;
        std::cout << "-----\t\t--------\t----------\t-----\t---------" << std::endl;
        
        int correct = 0;
        for(size_t i = 0; i < 4; ++i) {
            T pred_val = predictions(i, 0);
            int predicted_class = (pred_val > 0.5) ? 1 : 0;
            int expected_class = static_cast<int>(Y(i, 0) + 0.5);
            bool is_correct = (predicted_class == expected_class);
            
            if (is_correct) correct++;

            std::cout << "(" << X(i, 0) << "," << X(i, 1) << ")\t"
                      << expected_class << "\t\t" << pred_val << "\t"
                      << predicted_class << "\t"
                      << (is_correct ? "✓ CORRECTO" : "✗ ERROR") << std::endl;
        }
        
        T accuracy = (static_cast<T>(correct) / 4.0) * 100.0;
        std::cout << "\n=== Resumen ===" << std::endl;
        std::cout << "Accuracy: " << correct << "/4 (" << accuracy << "%)" << std::endl;

        // Prueba de robustez con ruido
        std::cout << "\n=== Prueba de Robustez (Inputs con Ruido) ===" << std::endl;
        Tensor<T, 2> X_noisy(4, 2);
        X_noisy = {
            0.05, 0.02,   // ~(0,0) con ruido
            0.02, 0.95,   // ~(0,1) con ruido
            0.98, 0.03,   // ~(1,0) con ruido
            0.97, 0.96    // ~(1,1) con ruido
        };
        
        auto noisy_predictions = nn_.predict(X_noisy);
        
        std::cout << "Input\t\tEsperado\tPredicción\tClase\tResultado" << std::endl;
        std::cout << "-----\t\t--------\t----------\t-----\t---------" << std::endl;
        
        std::array<int, 4> expected_noisy = {0, 1, 1, 0};
        int robust_correct = 0;
        
        for(size_t i = 0; i < 4; ++i) {
            T pred_val = noisy_predictions(i, 0);
            int predicted_class = (pred_val > 0.5) ? 1 : 0;
            bool is_correct = (predicted_class == expected_noisy[i]);
            
            if (is_correct) robust_correct++;
            
            std::cout << "(" << X_noisy(i, 0) << "," << X_noisy(i, 1) << ")\t"
                      << expected_noisy[i] << "\t\t" << pred_val << "\t"
                      << predicted_class << "\t"
                      << (is_correct ? "✓ ROBUSTO" : "✗ SENSIBLE") << std::endl;
        }
        
        T robustness = (static_cast<T>(robust_correct) / 4.0) * 100.0;
        std::cout << "\nRobustez: " << robust_correct << "/4 (" << robustness << "%)" << std::endl;

        // Evaluación final
        if (correct == 4 && robust_correct >= 3) {
            std::cout << "\n✅ ENTRENAMIENTO EXITOSO: XOR aprendido correctamente y robusto" << std::endl;
        } else if (correct == 4) {
            std::cout << "\n⚠️  ENTRENAMIENTO PARCIAL: XOR aprendido pero sensible al ruido" << std::endl;
        } else if (correct >= 3) {
            std::cout << "\n⚠️  ENTRENAMIENTO INCOMPLETO: Casi aprendido, necesita más épocas" << std::endl;
            std::cout << "   Sugerencia: Aumentar épocas a 30000 o ajustar learning rate" << std::endl;
        } else {
            std::cout << "\n❌ ENTRENAMIENTO FALLIDO: La red no convergió" << std::endl;
            std::cout << "   Sugerencias:" << std::endl;
            std::cout << "   - Aumentar épocas a 40000-50000" << std::endl;
            std::cout << "   - Probar learning rate = 0.1" << std::endl;
            std::cout << "   - Verificar implementación de BinaryCrossEntropyLoss" << std::endl;
            std::cout << "   - Verificar implementación del optimizador Adam" << std::endl;
        }
    }
};

} // namespace apps
} // namespace utec

#endif // UTAP_PATTERN_CLASSIFIER_H