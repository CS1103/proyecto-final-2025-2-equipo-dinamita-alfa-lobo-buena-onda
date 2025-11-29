//
// Created by HP on 28/11/2025.
//
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include "utec/apps/PatternClassifier.h" // Incluye la clase de aplicación
#include "utec/algebra/tensor.h"         // Para la definición del dataset de prueba

using namespace utec::apps;
using namespace utec::algebra;

// Función para ejecutar la demostración completa del Epic 3: Serialización del Clasificador
void run_pattern_classifier_demo() {
    using T = float;
    const std::string filepath = "classifier_weights.bin";

    // ----------------------------------------------------------------------
    // FASE 1: ENTRENAMIENTO Y SERIALIZACIÓN (Portabilidad)
    // ----------------------------------------------------------------------

    std::cout << "==========================================================" << std::endl;
    std::cout << "        FASE 1: ENTRENAMIENTO y GUARDADO (Clasificador)   " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Crear una instancia para entrenar
    PatternClassifier<T> trainer;

    // 2. Ejecutar el experimento de clasificación (esto entrena la red)
    // El método run_xor_experiment() está definido en PatternClassifier.h
    trainer.run_xor_experiment();

    // 3. Serializar (Guardar) los pesos entrenados
    try {
        // Llama al save_weights que a su vez llama a nn_.save_state()
        trainer.save_weights(filepath);
        std::cout << "\n✅ SERIALIZACIÓN EXITOSA: Modelo guardado en '" << filepath << "'" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR DE SERIALIZACIÓN: " << e.what() << std::endl;
        return;
    }

    // ----------------------------------------------------------------------
    // FASE 2: DESERIALIZACIÓN Y PRUEBA DE PORTABILIDAD
    // ----------------------------------------------------------------------

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "        FASE 2: CARGA y PRUEBA DE PORTABILIDAD         " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Crear una nueva instancia (el tester tendrá pesos aleatorios iniciales)
    PatternClassifier<T> tester;

    // 2. Deserializar (Cargar) los pesos entrenados, reemplazando los aleatorios
    try {
        // Llama al load_weights que a su vez llama a nn_.load_state()
        tester.load_weights(filepath);
        std::cout << "✅ CARGA EXITOSA: Pesos restaurados en la nueva instancia del clasificador." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ ERROR DE CARGA: " << e.what() << std::endl;
        return;
    }

    // 3. Definir el conjunto de prueba (usamos XOR para verificar)
    Tensor<T, 2> X_test(4, 2);
    X_test = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};

    Tensor<T, 2> Y_expected(4, 1);
    Y_expected = {0.0, 1.0, 1.0, 0.0};

    // 4. Prueba de Predicción con el modelo cargado
    auto predictions = tester.predict(X_test);

    std::cout << "\n--- Resultados de Predicción con Modelo Cargado ---" << std::endl;
    std::cout << "Input 1\tInput 2\tEsperado\tPredicho\tResultado" << std::endl;
    for(size_t i = 0; i < 4; ++i) {
        T expected = Y_expected(i, 0);
        T pred_raw = predictions(i, 0);
        int pred_class = (pred_raw > 0.5) ? 1 : 0;

        std::string result_str = (expected == pred_class) ? "CORRECTO" : "ERROR";

        std::cout << X_test(i, 0) << "\t" << X_test(i, 1) << "\t"
                  << expected << "\t\t" << pred_class << "\t\t" << result_str << std::endl;
    }
}

int main() {
    try {
        run_pattern_classifier_demo();
    } catch (const std::exception& e) {
        std::cerr << "\nError Crítico en la Ejecución del Clasificador: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}