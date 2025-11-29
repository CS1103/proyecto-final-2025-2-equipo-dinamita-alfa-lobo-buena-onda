# 1. Versión mínima de CMake y nombre del proyecto
cmake_minimum_required(VERSION 3.16)
project(PongAI CXX)

# 2. Establecer el estándar de C++ (C++17 es suficiente para tu código)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 3. Indicar a CMake dónde encontrar los archivos de cabecera (.h)
# Esto permite usar #include "utec/algebra/Tensor.h" sin rutas relativas.
include_directories(include)

# 4. Definir los ejecutables para las APLICACIONES PRINCIPALES
# CMake buscará los archivos .cpp y los compilará en ejecutables con los nombres indicados.

add_executable(pattern_classifier_app src/utec/apps/PatternClassifier.cpp)
add_executable(sequence_predictor_app src/utec/apps/SequencePredictor.cpp)
add_executable(controller_demo_app src/utec/apps/ControllerDemo.cpp)

# 5. Definir los ejecutables para las PRUEBAS
# Habilitamos las pruebas para que se puedan ejecutar con CTest
enable_testing()

# Prueba para la clase Tensor (Epic 1)
add_executable(test_tensor tests/test_tensor.cpp)
add_test(NAME TestTensor COMMAND test_tensor)

# Prueba para la biblioteca de Red Neuronal (Epic 2)
add_executable(test_neural_network tests/test_neural_network.cpp)
add_test(NAME TestNeuralNetwork COMMAND test_neural_network)

# Prueba para las aplicaciones y la portabilidad (Epic 3)
add_executable(test_applications tests/test_applications.cpp)
add_test(NAME TestApplications COMMAND test_applications)

# 6. Mensajes informativos para el usuario
message(STATUS "Proyecto configurado. Ejecutables creados:")
message(STATUS "  - Aplicaciones: pattern_classifier_app, sequence_predictor_app, controller_demo_app")
message(STATUS "  - Pruebas: test_tensor, test_neural_network, test_applications")
message(STATUS "")
message(STATUS "Para compilar, ejecuta: make")
message(STATUS "Para ejecutar una aplicación, por ejemplo: ./pattern_classifier_app")
message(STATUS "Para correr todas las pruebas, ejecuta: ctest")
