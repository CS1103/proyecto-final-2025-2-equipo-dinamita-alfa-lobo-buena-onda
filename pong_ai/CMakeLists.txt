cmake_minimum_required(VERSION 3.12)
project(PongAI VERSION 1.0)

# Manteniendo el estándar C++17 que definiste.
# Nota: El proyecto requiere C++20 para algunas características de Tensor.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. Definir la librería INTERFACE (solo cabeceras y rutas de inclusión)
add_library(ai_core INTERFACE)
target_include_directories(ai_core INTERFACE include)

# Mantenemos las fuentes de cabecera como tú las tenías:
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

# ----------------------------------------------------
# 2. EJECUTABLES DE APLICACIONES (src/)
#    Convertimos el proyecto en drivers individuales.
# ----------------------------------------------------

# [COMENTADO/ELIMINADO]: El ejecutable genérico no es necesario
# add_executable(run_experiments src/main.cpp)
# target_link_libraries(run_experiments PRIVATE ai_core) 

# Driver para el Controlador (Simulación/EnvGym)
add_executable(ControllerDemo src/ControllerDemo.cpp)
target_link_libraries(ControllerDemo PRIVATE ai_core)

# Driver para el Clasificador de Patrones (XOR)
add_executable(PatternClassifier src/PatternClassifier.cpp)
target_link_libraries(PatternClassifier PRIVATE ai_core)

# Driver para el Predictor de Secuencias (Regresión)
add_executable(SequencePredictor src/SequencePredictor.cpp)
target_link_libraries(SequencePredictor PRIVATE ai_core)


# ----------------------------------------------------
# 3. EJECUTABLES DE PRUEBAS (tests/)
# ----------------------------------------------------

# Pruebas para la Biblioteca Tensor (Epic 1)
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE ai_core)

# Pruebas para la Red Neuronal (Epic 2)
add_executable(test_neural_network tests/test_neural_network.cpp)
target_link_libraries(test_neural_network PRIVATE ai_core)

# Pruebas de Integración de Aplicaciones (Epic 3)
add_executable(test_applications tests/test_applications.cpp)
target_link_libraries(test_applications PRIVATE ai_core)

# Opcional: Habilitar CTest para correr todas las pruebas fácilmente
enable_testing()
add_test(NAME TestTensor COMMAND test_tensor)
add_test(NAME TestNN COMMAND test_neural_network)
add_test(NAME TestApps COMMAND test_applications)