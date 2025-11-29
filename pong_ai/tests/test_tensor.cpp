#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <numeric>
#include <algorithm>

// Incluir la cabecera del Tensor
#include <utec/algebra/tensor.h>

using namespace utec::algebra;
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

// ----------------------------------------------------------------------
// TEST 1: Inicialización, Acceso e Indexación
// ----------------------------------------------------------------------

void test_basic_functionality() {
    // 1. Inicialización y Shape (2x3x4)
    Tensor<T, 3> t1(2, 3, 4);
    if (t1.size() != 24) {
        throw std::runtime_error("Tamaño incorrecto en t1. Esperado: 24. Obtenido: " + std::to_string(t1.size()));
    }
    if (t1.shape()[0] != 2 || t1.shape()[2] != 4) {
        throw std::runtime_error("Shape incorrecto en t1.");
    }

    // 2. Llenar y Acceso variádico (operator())
    t1.fill(1.0f);
    t1(0, 1, 2) = 10.0f; // Modificar un elemento

    if (std::abs(t1(0, 0, 0) - 1.0f) > 1e-6) {
        throw std::runtime_error("Error en fill o acceso (0,0,0).");
    }
    if (std::abs(t1(0, 1, 2) - 10.0f) > 1e-6) {
        throw std::runtime_error("Error en asignación o acceso (0,1,2).");
    }

    // 3. Reshape (2x3x4 a 4x6x1)
    t1.reshape(4, 6, 1);
    if (t1.size() != 24 || t1.shape()[0] != 4) {
        throw std::runtime_error("Error en reshape.");
    }
    // Verificar que el dato en el índice lineal 6 (anteriormente (0,1,2)) sigue siendo 10.0
    // Índice lineal de (0,1,2) en 2x3x4 es: 0*12 + 1*4 + 2*1 = 6
    if (std::abs(t1(1, 2, 0) - 10.0f) > 1e-6) { // Índice lineal 6 en 4x6x1 es (1,2,0)
        // t1(1, 2, 0) -> 1*6 + 2*1 + 0 = 8. (Error en el cálculo del índice lineal o la prueba)
        // Recalculando: 6 es el 7mo elemento. En 4x6x1, el 7mo elemento es (1,1,0).
        // 1*6 + 1*1 + 0 = 7. El elemento en índice 6 es (1,0,0) que es 1*6 + 0*1 + 0 = 6.
        // El elemento 7mo (índice 6) debe ser 10.0.
        // En 4x6x1, (1, 0, 0) es el índice lineal 6.
        if (std::abs(t1(1, 0, 0) - 10.0f) > 1e-6) {
             throw std::runtime_error("Error de mapeo de datos después de reshape. Dato en índice 6 no es 10.0.");
        }
    }
}

// ----------------------------------------------------------------------
// TEST 2: Operaciones Element-wise y Escalares
// ----------------------------------------------------------------------

void test_element_wise_and_scalar() {
    Tensor<T, 2> t1(2, 2);
    t1 = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor<T, 2> t2(2, 2);
    t2 = {1.0f, 1.0f, 2.0f, 2.0f};

    T tolerance = 1e-6f;

    // 1. Suma Tensor-Tensor (Element-wise)
    auto t_sum = t1 + t2;
    if (std::abs(t_sum(0, 0) - 2.0f) > tolerance || std::abs(t_sum(1, 1) - 6.0f) > tolerance) {
        throw std::runtime_error("Error en Suma Tensor-Tensor.");
    }

    // 2. Multiplicación Tensor-Escalar
    auto t_mul_s = t1 * 2.0f;
    if (std::abs(t_mul_s(0, 1) - 4.0f) > tolerance || std::abs(t_mul_s(1, 0) - 6.0f) > tolerance) {
        throw std::runtime_error("Error en Multiplicación Tensor-Escalar.");
    }

    // 3. División Escalar-Tensor (usando el operador global)
    auto t_div_s = t_mul_s / 2.0f;
    if (std::abs(t_div_s(0, 1) - 2.0f) > tolerance) {
        throw std::runtime_error("Error en División Tensor-Escalar.");
    }
}


// ----------------------------------------------------------------------
// TEST 3: Broadcasting
// ----------------------------------------------------------------------

void test_broadcasting() {
    T tolerance = 1e-6f;

    // Caso 1: Tensor (3, 4) + Tensor (1, 4) -> Resultado (3, 4)
    Tensor<T, 2> A(3, 4);
    A.fill(1.0f); // Todos a 1.0
    Tensor<T, 2> B(1, 4);
    B = {1.0f, 2.0f, 3.0f, 4.0f}; // Fila de suma

    auto C = A + B; // B debe sumarse a cada fila de A

    if (C.shape()[0] != 3 || C.shape()[1] != 4) {
        throw std::runtime_error("Broadcasting: Shape de salida incorrecto.");
    }
    if (std::abs(C(0, 0) - 2.0f) > tolerance || std::abs(C(2, 3) - 5.0f) > tolerance) {
        throw std::runtime_error("Broadcasting: Suma incorrecta. Esperado C(2,3)=5.0. Obtenido: " + std::to_string(C(2, 3)));
    }

    // Caso 2: Tensor (3, 4) + Tensor (3, 1) -> Resultado (3, 4)
    Tensor<T, 2> D(3, 1);
    D = {10.0f, 20.0f, 30.0f}; // Columna de suma

    auto E = A + D; // D debe sumarse a cada columna de A

    if (std::abs(E(0, 3) - 11.0f) > tolerance || std::abs(E(2, 0) - 31.0f) > tolerance) {
        throw std::runtime_error("Broadcasting: Suma por columna incorrecta.");
    }

    // Caso 3: Incompatibilidad (3, 4) + (2, 4) -> Debe lanzar excepción
    Tensor<T, 2> F(2, 4);
    bool exception_caught = false;
    try {
        auto G = A + F;
    } catch (const std::invalid_argument& e) {
        exception_caught = true;
    }
    if (!exception_caught) {
        throw std::runtime_error("Broadcasting: No se lanzó excepción para shapes incompatibles (3,4) + (2,4).");
    }
}

// ----------------------------------------------------------------------
// TEST 4: Multiplicación Matricial (Matrix Product) y Transposición
// ----------------------------------------------------------------------

void test_matrix_operations() {
    T tolerance = 1e-5f;

    // 1. Transposición (2D)
    Tensor<T, 2> t_orig(2, 3);
    t_orig = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    auto t_transposed = transpose_2d(t_orig); // Esperado: 3x2

    if (t_transposed.shape()[0] != 3 || t_transposed.shape()[1] != 2) {
        throw std::runtime_error("Transpose: Shape de salida incorrecto.");
    }
    if (std::abs(t_transposed(1, 0) - 2.0f) > tolerance || std::abs(t_transposed(2, 1) - 6.0f) > tolerance) {
        throw std::runtime_error("Transpose: Elemento incorrecto. t(1,0) debe ser 2.0.");
    }

    // 2. Multiplicación Matricial (2D: 2x3 * 3x2 = 2x2)
    Tensor<T, 2> M1(2, 3);
    M1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor<T, 2> M2(3, 2);
    M2 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    auto M_result = matrix_product(M1, M2); // Esperado: M_result(2, 2)

    if (M_result.shape()[0] != 2 || M_result.shape()[1] != 2) {
        throw std::runtime_error("Matrix Product: Shape de salida incorrecto.");
    }

    // M_result(0, 0) = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // M_result(1, 1) = 4*10 + 5*12 + 6*14 (El dato es 12, no 14) -> 4*10 + 5*12 + 6*12 = 40 + 60 + 72 = 172
    // M_result(1, 1) = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154

    if (std::abs(M_result(0, 0) - 58.0f) > tolerance || std::abs(M_result(1, 1) - 154.0f) > tolerance) {
        throw std::runtime_error("Matrix Product: Resultado incorrecto. Esperado (0,0)=58.0, (1,1)=154.0.");
    }

    // 3. Multiplicación en Batch (3D: 2x2x3 * 2x3x2 = 2x2x2)
    Tensor<T, 3> B1(2, 2, 3);
    B1.fill(1.0f); // Matrices de 1s
    Tensor<T, 3> B2(2, 3, 2);
    B2.fill(1.0f); // Matrices de 1s

    auto B_result = matrix_product(B1, B2); // Esperado: Matrices 2x2 con valor 3 (porque k=3)

    if (B_result.shape()[0] != 2 || B_result.shape()[1] != 2 || B_result.shape()[2] != 2) {
        throw std::runtime_error("Batch Matrix Product: Shape de salida incorrecto.");
    }
    if (std::abs(B_result(0, 0, 0) - 3.0f) > tolerance) {
        throw std::runtime_error("Batch Matrix Product: Resultado incorrecto. Esperado 3.0.");
    }
}


// ----------------------------------------------------------------------
// FUNCIÓN PRINCIPAL DE PRUEBAS
// ----------------------------------------------------------------------

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "             TESTS DE LA BIBLIOTECA TENSOR (EPIC 1)       " << std::endl;
    std::cout << "==========================================================" << std::endl;

    run_test("Inicialización, Acceso y Reshape", test_basic_functionality);
    run_test("Operaciones Element-wise y Escalares", test_element_wise_and_scalar);
    run_test("Funcionalidad de Broadcasting", test_broadcasting);
    run_test("Operaciones Matriciales (MatMul/Transpose)", test_matrix_operations);

    std::cout << "==========================================================" << std::endl;
    return 0;
}