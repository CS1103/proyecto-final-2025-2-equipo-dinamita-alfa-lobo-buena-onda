//
// Created by HP on 28/11/2025.
//

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include "utec/apps/SequencePredictor.h"
#include "utec/algebra/Tensor.h"

using namespace utec::apps;
using namespace utec::algebra;

// Función para ejecutar la demostración completa del Epic 3: Serialización del Predictor
void run_sequence_predictor_demo() {
    using T = float;
    const std::string filepath = "sequence_predictor_weights.bin";

    // ----------------------------------------------------------------------
    // FASE 1: ENTRENAMIENTO Y SERIALIZACIÓN (Portabilidad)
    // ----------------------------------------------------------------------

    std::cout << "==========================================================" << std::endl;
    std::cout << "       FASE 1: ENTRENAMIENTO y GUARDADO (Predictor)       " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Crear una instancia para entrenar
    SequencePredictor<T> trainer;

    // 2. Ejecutar el experimento de regresión (entrena la red)
    trainer.run_series_experiment();

    // 3. Serializar (Guardar) los pesos entrenados
    try {
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

    // 1. Crear una nueva instancia (la red tendrá nuevos pesos aleatorios)
    SequencePredictor<T> tester;

    // 2. Deserializar (Cargar) los pesos entrenados
    try {
        tester.load_weights(filepath);
        std::cout << "✅ CARGA EXITOSA: Pesos restaurados en la nueva instancia del predictor." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ ERROR DE CARGA: " << e.what() << std::endl;
        return;
    }

    // 3. Definir el conjunto de prueba, incluyendo valores no vistos
    Tensor<T, 2> X_test(4, 1);
    X_test = {2.5, 5.0, 7.0, 10.0};

    // 4. Valores esperados (f(x) = 2x + 1)
    std::array<T, 4> Y_expected = {6.0, 11.0, 15.0, 21.0};

    // 5. Prueba de Predicción con el modelo cargado
    auto predictions = tester.predict(X_test);

    std::cout << "\n--- Resultados de Predicción con Modelo Cargado ---" << std::endl;
    std::cout << "Input\tEsperado\tPredicho" << std::endl;
    for(size_t i = 0; i < X_test.shape()[0]; ++i) {
        std::cout << X_test(i, 0) << "\t" << Y_expected[i] << "\t\t" << predictions(i, 0) << std::endl;
    }
}

int main() {
    try {
        run_sequence_predictor_demo();
    } catch (const std::exception& e) {
        std::cerr << "\nError Crítico en la Ejecución del Predictor de Secuencias: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}