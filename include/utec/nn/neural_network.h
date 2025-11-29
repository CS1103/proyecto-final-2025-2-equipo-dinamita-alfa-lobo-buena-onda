#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <utec/nn/nn_interfaces.h>
#include <utec/nn/nn_activation.h>
#include <utec/nn/nn_dense.h>
#include <utec/nn/nn_loss.h>
#include <utec/nn/nn_optimizer.h>
#include <utec/algebra/Tensor.h>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream> // ¡Necesario para la serialización!
#include <typeinfo> // Necesario para identificar el tipo de capa

namespace utec {
namespace neural_network {

using namespace utec::algebra;

template<typename T>
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<ILayer<T>>> layers_;

    // --- MÉTODOS PRIVADOS EXISTENTES ---

    // Forward pass a través de todas las capas
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& input) {
        auto output = input;
        for (auto& layer : layers_) {
            output = layer->forward(output);
        }
        return output;
    }

    // Backward pass a través de todas las capas (en orden inverso)
    void backward(const utec::algebra::Tensor<T, 2>& gradient) {
        auto grad = gradient;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
    }

    // Actualizar parámetros de todas las capas
    void update_parameters(IOptimizer<T>& optimizer) {
        for (auto& layer : layers_) {
            layer->update_params(optimizer);
        }
        optimizer.step();  // Incrementar paso del optimizador (para Adam)
    }

    // Extraer un batch del tensor
    utec::algebra::Tensor<T, 2> extract_batch(const utec::algebra::Tensor<T, 2>& data,
                                               size_t start,
                                               size_t size) {
        const auto shape = data.shape();
        size_t features = shape[1];

        utec::algebra::Tensor<T, 2> batch(size, features);

        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < features; ++j) {
                batch(i, j) = data(start + i, j);
            }
        }

        return batch;
    }

public:
    // Agregar una capa a la red
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers_.push_back(std::move(layer));
    }

    // Entrenamiento de la red
    template<template<typename> class LossType,
             template<typename> class OptimizerType = SGD>
    void train(const utec::algebra::Tensor<T, 2>& X,
               const utec::algebra::Tensor<T, 2>& Y,
               const size_t epochs,
               const size_t batch_size,
               T learning_rate) {

        OptimizerType<T> optimizer(learning_rate);
        const auto shape_X = X.shape();
        const size_t total_samples = shape_X[0];

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            T total_loss = T{0};
            size_t num_batches = 0;

            for (size_t batch_start = 0; batch_start < total_samples; batch_start += batch_size) {
                size_t batch_end = std::min(batch_start + batch_size, total_samples);
                size_t current_batch_size = batch_end - batch_start;

                auto X_batch = extract_batch(X, batch_start, current_batch_size);
                auto Y_batch = extract_batch(Y, batch_start, current_batch_size);

                auto predictions = forward(X_batch);

                LossType<T> loss_fn(predictions, Y_batch);
                T loss = loss_fn.loss();
                total_loss += loss;
                ++num_batches;

                auto gradient = loss_fn.loss_gradient();
                backward(gradient);

                update_parameters(optimizer);
            }
            // Comentado para no interrumpir el flujo del entrenamiento, puedes descomentarlo si lo necesitas:
            /*
            if ((epoch + 1) % 100 == 0) {
                T avg_loss = total_loss / static_cast<T>(num_batches);
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                          << " - Loss: " << avg_loss << std::endl;
            }
            */
        }
    }

    // Realizar predicciones
    utec::algebra::Tensor<T, 2> predict(const utec::algebra::Tensor<T, 2>& X) {
        return forward(X);
    }

    // -----------------------------------------------------------------
    //  MÉTODOS DE SERIALIZACIÓN (Requisito Epic 3: Portabilidad)
    // -----------------------------------------------------------------

    /**
     * @brief Serializa el estado de la red (pesos y sesgos) a un archivo binario.
     * @param filepath Ruta del archivo donde guardar los pesos.
     */
    void save_state(const std::string& filepath) const {
        std::ofstream ofs(filepath, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("Error al abrir el archivo para guardar el modelo.");
        }

        for (const auto& layer : layers_) {
            // Solo serializamos las capas Dense (las que contienen pesos W y sesgos B)
            if (auto dense_layer = dynamic_cast<Dense<T>*>(layer.get())) {
                // Asume que Dense<T> tiene un método para guardar sus parámetros
                dense_layer->save_parameters(ofs);
            }
        }
        ofs.close();
    }

    /**
     * @brief Deserializa y carga el estado de la red desde un archivo.
     * @param filepath Ruta del archivo desde donde cargar los pesos.
     */
    void load_state(const std::string& filepath) {
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error("Error al abrir el archivo para cargar el modelo. ¿Está entrenado?");
        }

        for (auto& layer : layers_) {
            // Solo deserializamos las capas Dense
            if (auto dense_layer = dynamic_cast<Dense<T>*>(layer.get())) {
                // Asume que Dense<T> tiene un método para cargar sus parámetros
                dense_layer->load_parameters(ifs);
            }
        }
        ifs.close();
    }
};

} // namespace neural_network
} // namespace utec

#endif // NEURAL_NETWORK_H