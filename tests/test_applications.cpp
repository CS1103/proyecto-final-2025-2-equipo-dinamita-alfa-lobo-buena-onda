//
// Created by HP on 28/11/2025.
//

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cmath>

// Incluir las cabeceras de las aplicaciones
#include "utec/apps/PatternClassifier.h"
#include "utec/apps/SequencePredictor.h" // Corregido de SequenceProdictor
#include "utec/apps/ControllerDemo.h"

using namespace utec::apps;
using namespace utec::algebra;

// Definición de una función de prueba genérica
template<typename T, typename Func>
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

// ----------------------------------------------------------------------
// TEST 1: PRUEBA DE PORTABILIDAD/SERIALIZACIÓN (PatternClassifier)
// ----------------------------------------------------------------------

void test_pattern_classifier_portability() {
    using T = float;
    const std::string test_filepath = "test_classifier_weights.bin";

    // 1. Instancia Original y Entrenamiento Mínimo
    PatternClassifier<T> trainer;
    // Creamos un dataset de prueba simple (e.g., XOR)
    Tensor<T, 2> X(4, 2);
    X = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};
    Tensor<T, 2> Y(4, 1);
    Y = {0.0, 1.0, 1.0, 0.0};

    // Entrenar lo suficiente para que los pesos sean distintos de cero/iniciales
    trainer.train<utec::neural_network::BinaryCrossEntropyLoss, utec::neural_network::Adam>(X, Y, 10, 4, 0.01f);

    // 2. Predecir con el modelo original y guardar
    auto original_prediction = trainer.predict(X);
    trainer.save_weights(test_filepath);

    // 3. Crear Nueva Instancia y Cargar Pesos
    PatternClassifier<T> tester;
    tester.load_weights(test_filepath);

    // 4. Predecir con el modelo cargado
    auto loaded_prediction = tester.predict(X);

    // 5. Verificar que las predicciones son (casi) idénticas
    T tolerance = 1e-6;
    for (size_t i = 0; i < 4; ++i) {
        if (std::abs(original_prediction(i, 0) - loaded_prediction(i, 0)) > tolerance) {
            // Limpiar archivo antes de lanzar error
            std::remove(test_filepath.c_str());
            throw std::runtime_error("Las predicciones del modelo cargado no coinciden con el original.");
        }
    }

    // Limpieza
    std::remove(test_filepath.c_str());
}

// ----------------------------------------------------------------------
// TEST 2: FUNCIONALIDAD BÁSICA (SequencePredictor)
// ----------------------------------------------------------------------

void test_sequence_predictor_functionality() {
    using T = float;

    // El método run_series_experiment() ya entrena y valida.
    // Aquí solo probamos que la predicción final tiene sentido.
    SequencePredictor<T> predictor;

    // Entrenamiento (se podría encapsular en un método más limpio)
    Tensor<T, 2> X(5, 1);
    X = {1.0, 2.0, 3.0, 4.0, 5.0};
    Tensor<T, 2> Y(5, 1);
    Y = {3.0, 5.0, 7.0, 9.0, 11.0};

    predictor.train<utec::neural_network::MSELoss, utec::neural_network::Adam>(X, Y, 10000, 5, 0.005f);

    // Prueba de Generalización (predicción de 6.0, esperado 13.0)
    Tensor<T, 2> X_test(1, 1);
    X_test = {6.0};
    auto test_predictions = predictor.predict(X_test);

    T expected = 13.0f;
    T tolerance = 0.5f; // Tolerancia amplia para un modelo simple

    if (std::abs(test_predictions(0, 0) - expected) > tolerance) {
        throw std::runtime_error("El predictor no pudo generalizar correctamente. Predicción: " +
                                 std::to_string(test_predictions(0, 0)) + ", Esperado: " + std::to_string(expected));
    }
}


// ----------------------------------------------------------------------
// TEST 3: PRUEBA BÁSICA DE SIMULACIÓN (ControllerDemo)
// ----------------------------------------------------------------------

void test_controller_demo_step() {
    using T = float;
    ControllerDemo<T> demo;

    // 1. Obtener estado inicial
    auto initial_state = demo.get_state();

    // 2. Ejecutar un paso con una acción
    // Usamos la acción 1 (Derecha/Aceleración)
    bool running = demo.step(1);

    // 3. Obtener el nuevo estado
    auto new_state = demo.get_state();

    // 4. Verificar que la simulación está activa y el estado ha cambiado
    if (!running) {
        throw std::runtime_error("La simulación se detuvo inmediatamente.");
    }

    // El estado debe haber cambiado (posición o velocidad)
    if (initial_state(0, 0) == new_state(0, 0) && initial_state(0, 1) == new_state(0, 1)) {
        throw std::runtime_error("El estado del entorno no cambió después de un paso de simulación.");
    }
}


// ----------------------------------------------------------------------
// FUNCIÓN PRINCIPAL DE PRUEBAS
// ----------------------------------------------------------------------

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "             EJECUCIÓN DE TEST AUTOMATIZADOS (EPIC 3)     " << std::endl;
    std::cout << "==========================================================" << std::endl;

    run_test<float>("Portabilidad (Clasificador)", test_pattern_classifier_portability);
    run_test<float>("Funcionalidad (Predictor)", test_sequence_predictor_functionality);
    run_test<float>("Simulación (ControllerDemo)", test_controller_demo_step);

    std::cout << "==========================================================" << std::endl;
    return 0;
}