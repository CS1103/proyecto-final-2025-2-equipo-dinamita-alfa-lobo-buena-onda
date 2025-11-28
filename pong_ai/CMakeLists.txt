    cmake_minimum_required(VERSION 3.12)
    project(PongAI VERSION 1.0)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # 1. Definir la librería (el motor de la IA)
    # Agregamos todos los archivos .h para que aparezcan en el IDE
    add_library(ai_core INTERFACE)

    target_include_directories(ai_core INTERFACE include)

    target_sources(ai_core INTERFACE
            include/utec/algebra/tensor.h
            include/utec/nn/neural_network.h
            include/utec/nn/nn_interfaces.h
            include/utec/nn/nn_dense.h
            include/utec/nn/nn_activation.h
            include/utec/nn/nn_loss.h
            include/utec/nn/nn_optimizer.h

            include/utec/apps/PatternClassifier.h
            include/utec/apps/SequencePredictor.h
            include/utec/apps/ControllerDemo.h
    )

    # 2. Definir el ejecutable (donde correrás los experimentos del Epic 3)
    add_executable(run_experiments
            src/main.cpp
    )

    # 3. Vincular el ejecutable con la librería
    target_link_libraries(run_experiments PRIVATE ai_core)

