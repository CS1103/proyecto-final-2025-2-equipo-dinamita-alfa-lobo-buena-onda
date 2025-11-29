//
// Created by HP on 28/11/2025.
//
#include <iostream>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

// Cabeceras de la biblioteca NN
#include <utec/algebra/tensor.h>
#include <utec/nn/nn_interfaces.h>
#include <utec/nn/nn_dense.h>
#include <utec/nn/nn_activation.h>
#include <utec/nn/nn_loss.h>
#include <utec/nn/nn_optimizer.h>
#include <utec/nn/neural_network.h>

using namespace utec::algebra;
using namespace utec::neural_network;

using T = float;

// ----------------------------------------------------------------------
// Función de Ayuda para Ejecutar Pruebas
// ----------------------------------------------------------------------
template<typename Func>
void run_test(const std::string& name, Func test_func) {
    std::cout << "-> Ejecutando prueba: " << name << "..." << std::flush;
    try {
        test_func();
        std::cout << "\t\t[PASSED] ✅" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\t\t[FAILED] ❌ - Error: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "\t\t[FAILED] ❌ - Error desconocido." << std::endl;
    }
}

// Funciones de inicialización dummy para las pruebas
void init_dummy(Tensor<T, 2>& t) {
    // Inicialización simple, por ejemplo, ones
    t.fill(1.0f);
}

// ----------------------------------------------------------------------
// TEST 1: Validación de Dimensiones en Capa DENSE
// ----------------------------------------------------------------------

void test_dense_dimensions() {
    // Definimos dimensiones: (batch x in_features) * (in_features x out_features)
    size_t batch_size = 3;
    size_t in_f = 5;
    size_t out_f = 2;

    // Crear la capa Dense
    Dense<T> dense_layer(
        in_f, out_f, init_dummy, init_dummy
    );

    // Entrada (Batch x in_features)
    Tensor<T, 2> X(batch_size, in_f);
    X.fill(2.0f);

    // --- Forward Pass ---
    auto Z = dense_layer.forward(X);
    if (Z.shape()[0] != batch_size || Z.shape()[1] != out_f) {
        throw std::runtime_error("Dimensiones incorrectas en Forward. Esperado: (3, 2). Obtenido: (" +
                                 std::to_string(Z.shape()[0]) + ", " + std::to_string(Z.shape()[1]) + ")");
    }

    // --- Backward Pass ---
    // Gradiente de la función de pérdida respecto a la salida (Batch x out_features)
    Tensor<T, 2> dZ(batch_size, out_f);
    dZ.fill(0.1f);

    auto dX = dense_layer.backward(dZ);

    // 1. Verificar el gradiente de la entrada (dX)
    if (dX.shape()[0] != batch_size || dX.shape()[1] != in_f) {
        throw std::runtime_error("Dimensiones incorrectas en dX. Esperado: (3, 5). Obtenido: (" +
                                 std::to_string(dX.shape()[0]) + ", " + std::to_string(dX.shape()[1]) + ")");
    }

    // 2. Verificar el gradiente del peso (dW)
    // Se asume que el gradiente se almacena internamente y se puede inspeccionar
    // Se requiere acceso al gradiente interno, si no es posible, se asume que la próxima prueba de optimización lo validará.
    // Para simplificar, solo verificamos el flujo.
}

// ----------------------------------------------------------------------
// TEST 2: Validación de Funciones de Activación
// ----------------------------------------------------------------------

void test_activation_functionality() {
    T tolerance = 1e-6f;

    // --- ReLU ---
    ReLU<T> relu_layer;
    Tensor<T, 2> X_relu(1, 4);
    X_relu = {-1.0f, 0.0f, 2.0f, -5.0f};

    // Forward: {0, 0, 2, 0}
    auto Z_relu = relu_layer.forward(X_relu);
    if (std::abs(Z_relu(0, 2) - 2.0f) > tolerance || std::abs(Z_relu(0, 0) - 0.0f) > tolerance) {
        throw std::runtime_error("ReLU Forward incorrecto.");
    }

    // Backward: dZ = {1, 1, 1, 1} -> dX = {0, 0/1, 1, 0} (depende de la subgradiente)
    Tensor<T, 2> dZ_relu(1, 4);
    dZ_relu.fill(1.0f);
    auto dX_relu = relu_layer.backward(dZ_relu);
    if (std::abs(dX_relu(0, 2) - 1.0f) > tolerance || std::abs(dX_relu(0, 0) - 0.0f) > tolerance) {
        throw std::runtime_error("ReLU Backward incorrecto.");
    }

    // --- Sigmoid ---
    Sigmoid<T> sigmoid_layer;
    Tensor<T, 2> X_sigmoid(1, 1);
    X_sigmoid = {0.0f}; // Sigmoid(0) = 0.5

    // Forward: {0.5}
    auto Z_sigmoid = sigmoid_layer.forward(X_sigmoid);
    if (std::abs(Z_sigmoid(0, 0) - 0.5f) > tolerance) {
        throw std::runtime_error("Sigmoid Forward incorrecto. Esperado: 0.5. Obtenido: " + std::to_string(Z_sigmoid(0, 0)));
    }

    // Backward: dSigmoid(x)/dx = sigmoid(x) * (1 - sigmoid(x)). Para x=0, es 0.5 * 0.5 = 0.25
    Tensor<T, 2> dZ_sigmoid(1, 1);
    dZ_sigmoid = {1.0f};
    auto dX_sigmoid = sigmoid_layer.backward(dZ_sigmoid);
    if (std::abs(dX_sigmoid(0, 0) - 0.25f) > 0.01f) { // Tolerancia un poco más alta
        throw std::runtime_error("Sigmoid Backward incorrecto. Esperado: 0.25. Obtenido: " + std::to_string(dX_sigmoid(0, 0)));
    }
}

// ----------------------------------------------------------------------
// TEST 3: Sanity Check de Entrenamiento (La pérdida debe disminuir)
// ----------------------------------------------------------------------

void test_training_sanity_check() {
    // Topología 1 -> 4 -> 1 (Simple regresión lineal)
    NeuralNetwork<T> nn;
    nn.add_layer(std::make_unique<Dense<T>>(1, 4, init_dummy, init_dummy));
    nn.add_layer(std::make_unique<ReLU<T>>());
    nn.add_layer(std::make_unique<Dense<T>>(4, 1, init_dummy, init_dummy));

    // Datos: y = 2x
    Tensor<T, 2> X(2, 1);
    X = {1.0f, 2.0f};
    Tensor<T, 2> Y(2, 1);
    Y = {2.0f, 4.0f};

    // 1. Obtener la pérdida inicial
    auto initial_predictions = nn.predict(X);
    MSELoss<T> loss_fn;
    T initial_loss = loss_fn.forward(initial_predictions, Y);

    // 2. Entrenar por pocos pasos (debe ser suficiente para reducir la pérdida)
    size_t epochs = 10;
    T learning_rate = 0.01f;

    // Usamos el método train de NeuralNetwork
    nn.template train<MSELoss, Adam>(X, Y, epochs, 2, learning_rate);

    // 3. Obtener la pérdida final
    auto final_predictions = nn.predict(X);
    T final_loss = loss_fn.forward(final_predictions, Y);

    // 4. Verificar que la pérdida ha disminuido significativamente
    if (final_loss >= initial_loss) {
        throw std::runtime_error("La pérdida no disminuyó después del entrenamiento. Inicial: " +
                                 std::to_string(initial_loss) + ", Final: " + std::to_string(final_loss));
    }
}

// ----------------------------------------------------------------------
// FUNCIÓN PRINCIPAL
// ----------------------------------------------------------------------

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "           TESTS DE LA BIBLIOTECA NEURAL NETWORK          " << std::endl;
    std::cout << "==========================================================" << std::endl;

    run_test("Dimensiones Capa Dense (Forward/Backward)", test_dense_dimensions);
    run_test("Funcionalidad Activaciones (ReLU/Sigmoid)", test_activation_functionality);
    run_test("Sanity Check de Entrenamiento (Pérdida)", test_training_sanity_check);

    std::cout << "==========================================================" << std::endl;
    return 0;
}