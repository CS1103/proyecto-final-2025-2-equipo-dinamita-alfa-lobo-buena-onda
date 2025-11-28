#include <iostream>
// Nuevos includes para las tres aplicaciones
#include <utec/apps/PatternClassifier.h>
#include <utec/apps/SequencePredictor.h>
#include <utec/apps/ControllerDemo.h>     // <-- Nuevo include

using namespace utec::apps;

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "           EPIC 3: APLICACIONES DE RED NEURONAL           " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Clasificación de Patrones (XOR)
    PatternClassifier<double> classifier;
    classifier.run_xor_experiment();

    std::cout << "\n" << std::string(60, '=') << "\n";

    // 2. Predicción de Series (Regresión)
    SequencePredictor<double> predictor;
    predictor.run_series_experiment();

    std::cout << "\n" << std::string(60, '=') << "\n";

    // 3. Control Simplificado (Clonación de Comportamiento)
    ControllerDemo<double> controller;
    controller.run_control_experiment();

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Todos los experimentos de la Epic 3 completados." << std::endl;

    return 0;
}