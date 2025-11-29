//
// Created by HP on 28/11/2025.
//
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "utec/apps/ControllerDemo.h" // Incluye la clase ControllerDemo (nuestro EnvGym)

using namespace utec::apps;

// Función auxiliar para correr la demostración completa del Epic 3
void run_epic3_demo() {
    using T = float;

    // ----------------------------------------------------------------------
    // FASE 1: ENTRENAMIENTO Y SERIALIZACIÓN (Portabilidad)
    // ----------------------------------------------------------------------

    std::cout << "==========================================================" << std::endl;
    std::cout << "             FASE 1: ENTRENAMIENTO Y GUARDADO             " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Crear una instancia para entrenar
    ControllerDemo<T> trainer;

    // 2. Entrenar la política del experto
    size_t epochs = 5000;
    T learning_rate = 0.01;
    trainer.train_expert_policy(epochs, learning_rate);

    // 3. Serializar (Guardar) los pesos entrenados
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

    // 1. Crear una nueva instancia (el Agente de prueba)
    ControllerDemo<T> agent;

    // 2. Deserializar (Cargar) los pesos entrenados en la nueva instancia
    try {
        agent.load_weights(filepath);
        std::cout << "✅ CARGA EXITOSA: Pesos restaurados en la nueva instancia del agente." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ ERROR DE CARGA: " << e.what() << std::endl;
        return;
    }

    // 3. Iniciar la simulación (EnvGym Loop)
    agent.reset();
    int max_steps = 50;

    std::cout << "\nIniciando bucle de simulación (Máx. " << max_steps << " pasos):" << std::endl;

    for (int step = 0; step < max_steps; ++step) {
        // a) El Entorno proporciona el Estado
        //    State = [posición, velocidad] (Tensor 1x2)
        auto state = agent.get_state();

        // b) El Agente (NN) predice la acción
        auto output = agent.get_network().predict(state);

        // c) Determinar la Acción (Decisión Binaria: 0 o 1)
        //    La red usa Sigmoid, el valor > 0.5 se mapea a acción = 1 (Derecha)
        T pred_val = output(0, 0);
        int action = (pred_val > 0.5) ? 1 : 0; // 1: Derecha, 0: Izquierda

        // d) Ejecutar la acción en el EnvGym
        bool still_running = agent.step(action);

        // e) Verificar si la simulación terminó
        if (!still_running) {
            break;
        }
    }
}

int main() {
    // Es buena práctica envolver todo en un try-catch para manejar errores de archivos
    try {
        run_epic3_demo();
    } catch (const std::exception& e) {
        std::cerr << "Error Crítico en la Ejecución: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}