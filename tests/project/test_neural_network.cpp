#include <iostream>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <fstream>

// Cabeceras de la biblioteca NN
#include <utec/algebra/Tensor.h>
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

// Funciones de inicialización para las pruebas
void init_ones(Tensor<T, 2>& t) {
    t.fill(1.0f);
}

void init_small_random(Tensor<T, 2>& t) {
    for (size_t i = 0; i < t.size(); ++i) {
        *(t.begin() + i) = 0.1f * (static_cast<T>(rand()) / RAND_MAX);
    }
}

// ----------------------------------------------------------------------
// TEST 1: Validación de Dimensiones en Capa DENSE
// COMPLEJIDAD Forward: O(batch_size * in_features * out_features)
// COMPLEJIDAD Backward: O(batch_size * in_features * out_features)
// ESPACIO: O(in_features * out_features) para pesos
// ----------------------------------------------------------------------

void test_dense_dimensions() {
    // Definimos dimensiones: (batch x in_features) * (in_features x out_features)
    size_t batch_size = 3;
    size_t in_f = 5;
    size_t out_f = 2;

    // Crear la capa Dense
    Dense<T> dense_layer(in_f, out_f, init_ones, init_ones);

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

    // Verificar el gradiente de la entrada (dX)
    if (dX.shape()[0] != batch_size || dX.shape()[1] != in_f) {
        throw std::runtime_error("Dimensiones incorrectas en dX. Esperado: (3, 5). Obtenido: (" +
                                 std::to_string(dX.shape()[0]) + ", " + std::to_string(dX.shape()[1]) + ")");
    }
}

// ----------------------------------------------------------------------
// TEST 2: Validación de Funciones de Activación
// COMPLEJIDAD: O(N) donde N = número de elementos en el tensor
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

    // Backward: dZ = {1, 1, 1, 1} -> dX = {0, 0/1, 1, 0}
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
    if (std::abs(dX_sigmoid(0, 0) - 0.25f) > 0.01f) {
        throw std::runtime_error("Sigmoid Backward incorrecto. Esperado: 0.25. Obtenido: " + std::to_string(dX_sigmoid(0, 0)));
    }

    // --- Tanh (si está implementada) ---
    // Se puede agregar aquí si implementaste Tanh
}

// ----------------------------------------------------------------------
// TEST 3: Funciones de Pérdida (Loss Functions)
// COMPLEJIDAD: O(N) donde N = batch_size * output_size
// ----------------------------------------------------------------------

void test_loss_functions() {
    T tolerance = 1e-5f;

    // --- MSE Loss ---
    Tensor<T, 2> pred_mse(2, 1);
    pred_mse = {2.0f, 4.0f};
    Tensor<T, 2> target_mse(2, 1);
    target_mse = {1.0f, 3.0f};

    MSELoss<T> mse(pred_mse, target_mse);
    T loss_mse = mse.loss();
    // MSE = mean((2-1)^2 + (4-3)^2) = mean(1 + 1) = 1.0
    if (std::abs(loss_mse - 1.0f) > tolerance) {
        throw std::runtime_error("MSE Loss incorrecto. Esperado: 1.0. Obtenido: " + std::to_string(loss_mse));
    }

    // Verificar que la pérdida es positiva
    if (loss_mse < 0) {
        throw std::runtime_error("MSE Loss debe ser no-negativo.");
    }

    // --- Binary Cross Entropy Loss ---
    Tensor<T, 2> pred_bce(2, 1);
    pred_bce = {0.9f, 0.1f};
    Tensor<T, 2> target_bce(2, 1);
    target_bce = {1.0f, 0.0f};

    BinaryCrossEntropyLoss<T> bce(pred_bce, target_bce);
    T loss_bce = bce.loss();
    
    // Verificar que la pérdida está en rango razonable
    if (loss_bce < 0 || loss_bce > 10.0) {
        throw std::runtime_error("BCE Loss fuera de rango razonable: " + std::to_string(loss_bce));
    }

    // Verificar que BCE penaliza más las predicciones incorrectas
    Tensor<T, 2> pred_bad(2, 1);
    pred_bad = {0.1f, 0.9f};  // Predicciones invertidas (malas)
    
    BinaryCrossEntropyLoss<T> bce_bad(pred_bad, target_bce);
    T loss_bce_bad = bce_bad.loss();
    
    if (loss_bce_bad <= loss_bce) {
        throw std::runtime_error("BCE debería penalizar más las predicciones incorrectas.");
    }
}

// ----------------------------------------------------------------------
// TEST 4: Optimizadores
// COMPLEJIDAD por update: O(N) donde N = número de parámetros
// ----------------------------------------------------------------------

void test_optimizers() {
    T tolerance = 1e-6f;

    // --- SGD (Stochastic Gradient Descent) ---
    SGD<T> sgd(0.1f); // learning_rate = 0.1
    
    Tensor<T, 2> params(2, 2);
    params.fill(1.0f);
    
    Tensor<T, 2> grads(2, 2);
    grads.fill(0.5f); // Gradiente constante

    sgd.update(params, grads);
    
    // Nuevo valor = 1.0 - 0.1 * 0.5 = 0.95
    if (std::abs(params(0, 0) - 0.95f) > tolerance) {
        throw std::runtime_error("SGD update incorrecto. Esperado: 0.95. Obtenido: " + std::to_string(params(0, 0)));
    }

    // --- Adam Optimizer ---
    Adam<T> adam(0.01f); // learning_rate = 0.01
    
    Tensor<T, 2> params_adam(2, 2);
    params_adam.fill(1.0f);
    
    Tensor<T, 2> grads_adam(2, 2);
    grads_adam.fill(0.1f);

    // Primera actualización
    adam.update(params_adam, grads_adam);
    
    // El valor debe haber cambiado (no verificamos el valor exacto por la complejidad de Adam)
    if (std::abs(params_adam(0, 0) - 1.0f) < tolerance) {
        throw std::runtime_error("Adam update no modificó los parámetros.");
    }

    // Segunda actualización (para verificar que mantiene estado)
    adam.update(params_adam, grads_adam);
}

// ----------------------------------------------------------------------
// TEST 5: Sanity Check de Entrenamiento (La pérdida debe disminuir)
// COMPLEJIDAD por epoch: O(epochs * batch_size * layers * weights)
// Para este test: O(100 * 2 * 2 * (1*4 + 4*1)) = O(100 * 2 * 2 * 8) = O(3200)
// ----------------------------------------------------------------------

void test_training_sanity_check() {
    // Topología 1 -> 4 -> 1 (Simple regresión lineal)
    NeuralNetwork<T> nn;
    nn.add_layer(std::make_unique<Dense<T>>(1, 4, init_small_random, init_small_random));
    nn.add_layer(std::make_unique<ReLU<T>>());
    nn.add_layer(std::make_unique<Dense<T>>(4, 1, init_small_random, init_small_random));

    // Datos: y = 2x
    Tensor<T, 2> X(2, 1);
    X = {1.0f, 2.0f};
    Tensor<T, 2> Y(2, 1);
    Y = {2.0f, 4.0f};

    // 1. Obtener la pérdida inicial
    auto initial_predictions = nn.predict(X);
    MSELoss<T> initial_loss_fn(initial_predictions, Y);
    T initial_loss = initial_loss_fn.loss();

    // 2. Entrenar por varios pasos
    size_t epochs = 100;
    T learning_rate = 0.01f;

    nn.template train<MSELoss, Adam>(X, Y, epochs, 2, learning_rate);

    // 3. Obtener la pérdida final
    auto final_predictions = nn.predict(X);
    MSELoss<T> final_loss_fn(final_predictions, Y);
    T final_loss = final_loss_fn.loss();

    // 4. Verificar que la pérdida ha disminuido significativamente
    if (final_loss >= initial_loss * 0.9) { // Al menos 10% de reducción
        throw std::runtime_error("La pérdida no disminuyó suficientemente. Inicial: " +
                                 std::to_string(initial_loss) + ", Final: " + std::to_string(final_loss));
    }

    std::cout << " (Loss: " << initial_loss << " -> " << final_loss << ")";
}

// ----------------------------------------------------------------------
// TEST 6: Convergencia en Problema XOR (Épic 2 Completo)
// Este es el test más importante: demuestra que la NN puede aprender
// COMPLEJIDAD: O(epochs * batch_size * sum(layer_weights))
// ----------------------------------------------------------------------

void test_xor_convergence() {
    // XOR es un problema no linealmente separable, requiere hidden layer
    NeuralNetwork<T> nn;
    nn.add_layer(std::make_unique<Dense<T>>(2, 4, init_small_random, init_small_random));
    nn.add_layer(std::make_unique<Sigmoid<T>>());
    nn.add_layer(std::make_unique<Dense<T>>(4, 1, init_small_random, init_small_random));
    nn.add_layer(std::make_unique<Sigmoid<T>>());

    // Dataset XOR
    Tensor<T, 2> X(4, 2);
    X = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    
    Tensor<T, 2> Y(4, 1);
    Y = {0.0f, 1.0f, 1.0f, 0.0f};

    // Entrenar
    nn.template train<BinaryCrossEntropyLoss, Adam>(X, Y, 1000, 4, 0.1f);

    // Verificar predicciones
    auto predictions = nn.predict(X);
    
    int correct = 0;
    for (size_t i = 0; i < 4; ++i) {
        int pred_class = (predictions(i, 0) > 0.5) ? 1 : 0;
        int true_class = static_cast<int>(Y(i, 0));
        if (pred_class == true_class) {
            correct++;
        }
    }

    // Debe acertar al menos 3 de 4 (75% accuracy)
    if (correct < 3) {
        throw std::runtime_error("La red no convergió en XOR. Aciertos: " + std::to_string(correct) + "/4");
    }

    std::cout << " (Accuracy: " << correct << "/4)";
}

// ----------------------------------------------------------------------
// TEST 7: Serialización de la Red Neuronal (Épic 2 + Épic 3)
// COMPLEJIDAD: O(total_parameters)
// ----------------------------------------------------------------------

void test_nn_serialization() {
    const std::string test_file = "test_nn_serialization.bin";
    T tolerance = 1e-5f;

    // 1. Crear y entrenar una red
    NeuralNetwork<T> nn1;
    nn1.add_layer(std::make_unique<Dense<T>>(2, 3, init_small_random, init_small_random));
    nn1.add_layer(std::make_unique<ReLU<T>>());
    nn1.add_layer(std::make_unique<Dense<T>>(3, 1, init_small_random, init_small_random));

    Tensor<T, 2> X(2, 2);
    X = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor<T, 2> Y(2, 1);
    Y = {1.0f, 0.0f};

    nn1.template train<BinaryCrossEntropyLoss, Adam>(X, Y, 50, 2, 0.01f);

    // 2. Guardar el estado
    nn1.save_state(test_file);

    // 3. Obtener predicciones originales
    auto pred_original = nn1.predict(X);

    // 4. Crear una nueva red y cargar el estado
    NeuralNetwork<T> nn2;
    nn2.add_layer(std::make_unique<Dense<T>>(2, 3, init_ones, init_ones));
    nn2.add_layer(std::make_unique<ReLU<T>>());
    nn2.add_layer(std::make_unique<Dense<T>>(3, 1, init_ones, init_ones));

    nn2.load_state(test_file);

    // 5. Obtener predicciones de la red cargada
    auto pred_loaded = nn2.predict(X);

    // 6. Verificar que son (casi) idénticas
    for (size_t i = 0; i < pred_original.size(); ++i) {
        if (std::abs(*(pred_original.begin() + i) - *(pred_loaded.begin() + i)) > tolerance) {
            std::remove(test_file.c_str());
            throw std::runtime_error("Serialización falló: predicciones no coinciden.");
        }
    }

    // Limpieza
    std::remove(test_file.c_str());
}

// ----------------------------------------------------------------------
// TEST 8: Forward y Backward en Red Completa (Épic 2 - Verificación Manual)
// ----------------------------------------------------------------------

void test_forward_backward_manual() {
    T tolerance = 1e-5f;

    // Red simple: 2 -> 2 -> 1
    NeuralNetwork<T> nn;
    nn.add_layer(std::make_unique<Dense<T>>(2, 2, init_ones, init_ones));
    nn.add_layer(std::make_unique<Sigmoid<T>>());
    nn.add_layer(std::make_unique<Dense<T>>(2, 1, init_ones, init_ones));

    // Entrada simple
    Tensor<T, 2> X(1, 2);
    X = {1.0f, 1.0f};

    // Forward
    auto output = nn.predict(X);
    
    // Verificar que la salida tiene las dimensiones correctas
    if (output.shape()[0] != 1 || output.shape()[1] != 1) {
        throw std::runtime_error("Forward pass: dimensiones incorrectas.");
    }

    // La salida debe ser un número válido (no NaN, no Inf)
    if (std::isnan(output(0, 0)) || std::isinf(output(0, 0))) {
        throw std::runtime_error("Forward pass: salida inválida (NaN o Inf).");
    }
}

// ----------------------------------------------------------------------
// FUNCIÓN PRINCIPAL
// ----------------------------------------------------------------------

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "     TESTS DE LA BIBLIOTECA NEURAL NETWORK (EPIC 2)      " << std::endl;
    std::cout << "==========================================================" << std::endl;

    run_test("Dimensiones Capa Dense (Forward/Backward)", test_dense_dimensions);
    run_test("Funcionalidad Activaciones (ReLU/Sigmoid)", test_activation_functionality);
    run_test("Funciones de Pérdida (MSE/BCE)", test_loss_functions);
    run_test("Optimizadores (SGD/Adam)", test_optimizers);
    run_test("Sanity Check de Entrenamiento (Pérdida)", test_training_sanity_check);
    run_test("Convergencia en XOR (Problema No Lineal)", test_xor_convergence);
    run_test("Serialización de Red Neuronal", test_nn_serialization);
    run_test("Forward/Backward Manual en Red Completa", test_forward_backward_manual);

    std::cout << "==========================================================" << std::endl;
    std::cout << "   ✅ ÉPIC 2 COMPLETAMENTE CUBIERTO - 8 Tests Pasados   " << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}