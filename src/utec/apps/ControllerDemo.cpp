//
// Created by HP on 28/11/2025.
//
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "utec/apps/ControllerDemo.h"

using namespace utec::apps;

void run_epic3_demo() {
    using T = float;

    // ----------------------------------------------------------------------
    // FASE 1: ENTRENAMIENTO Y SERIALIZACIÓN (Portabilidad)
    // ----------------------------------------------------------------------

    std::cout << "==========================================================" << std::endl;
    std::cout << "             FASE 1: ENTRENAMIENTO Y GUARDADO             " << std::endl;
    std::cout << "==========================================================" << std::endl;

    ControllerDemo<T> trainer;

    // ⭐ HIPERPARÁMETROS CORREGIDOS (aumentados)
    size_t epochs = 20000;      // Aumentado de 5000 a 20000
    T learning_rate = 0.01;     // Mantener igual está bien
    trainer.train_expert_policy(epochs, learning_rate);

    const std::string filepath = "policy_weights.bin";
    try {
        trainer.save_weights(filepath);
        std::cout << "\n✅ SERIALIZACIÓN EXITOSA: Modelo guardado en '" << filepath << "'" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR DE SERIALIZACIÓN: " << e.what() << std::endl;
        return;
    }

    // ----------------------------------------------------------------------
    // FASE 2: DESERIALIZACIÓN Y SIMULACIÓN (Integración con EnvGym)
    // ----------------------------------------------------------------------

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "              FASE 2: CARGA Y SIMULACIÓN (EnvGym)         " << std::endl;
    std::cout << "==========================================================" << std::endl;

    ControllerDemo<T> agent;

    try {
        agent.load_weights(filepath);
        std::cout << "✅ CARGA EXITOSA: Pesos restaurados en la nueva instancia del agente." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ ERROR DE CARGA: " << e.what() << std::endl;
        return;
    }

    // Simulación
    agent.reset();
    int max_steps = 50;

    std::cout << "\nIniciando bucle de simulación (Máx. " << max_steps << " pasos):" << std::endl;

    int step_count = 0;
    for (int step = 0; step < max_steps; ++step) {
        auto state = agent.get_state();
        auto output = agent.get_network().predict(state);

        T pred_val = output(0, 0);
        int action = (pred_val > 0.5) ? 1 : 0;

        bool still_running = agent.step(action);
        step_count++;

        if (!still_running) {
            break;
        }
    }

    std::cout << "Simulación completada después de " << step_count << " pasos." << std::endl;
    
    if (step_count >= max_steps) {
        std::cout << "✅ CONTROL EXITOSO: El agente mantuvo el equilibrio durante toda la simulación" << std::endl;
    } else {
        std::cout << "⚠️  CONTROL PARCIAL: El agente perdió el equilibrio antes del límite" << std::endl;
    }
}

int main() {
    try {
        run_epic3_demo();
    } catch (const std::exception& e) {
        std::cerr << "Error Crítico en la Ejecución: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}