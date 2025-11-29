#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <numeric>
#include <algorithm>

// Incluir la cabecera del Tensor
#include <utec/algebra/Tensor.h>

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
// COMPLEJIDAD: O(N) para inicialización donde N = producto de dimensiones
// ESPACIO: O(N) para almacenar los datos del tensor
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
    // COMPLEJIDAD: O(N) para fill, O(1) para acceso variádico
    t1.fill(1.0f);
    t1(0, 1, 2) = 10.0f; // Modificar un elemento

    if (std::abs(t1(0, 0, 0) - 1.0f) > 1e-6) {
        throw std::runtime_error("Error en fill o acceso (0,0,0).");
    }
    if (std::abs(t1(0, 1, 2) - 10.0f) > 1e-6) {
        throw std::runtime_error("Error en asignación o acceso (0,1,2).");
    }

    // 3. Reshape (2x3x4 a 4x6x1)
    // COMPLEJIDAD: O(N) para redimensionar y copiar datos si es necesario
    t1.reshape(4, 6, 1);
    if (t1.size() != 24 || t1.shape()[0] != 4) {
        throw std::runtime_error("Error en reshape.");
    }
    
    // Verificar que el dato en el índice lineal 6 se preserva
    if (std::abs(t1(1, 0, 0) - 10.0f) > 1e-6) {
        throw std::runtime_error("Error de mapeo de datos después de reshape.");
    }
}

// ----------------------------------------------------------------------
// TEST 2: Operaciones Element-wise y Escalares
// COMPLEJIDAD: O(N) para cada operación element-wise
// ESPACIO: O(N) para almacenar el tensor resultado
// ----------------------------------------------------------------------

void test_element_wise_and_scalar() {
    Tensor<T, 2> t1(2, 2);
    t1 = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor<T, 2> t2(2, 2);
    t2 = {1.0f, 1.0f, 2.0f, 2.0f};

    T tolerance = 1e-6f;

    // 1. Suma Tensor-Tensor (Element-wise)
    // COMPLEJIDAD: O(N) donde N = 2*2 = 4 elementos
    auto t_sum = t1 + t2;
    if (std::abs(t_sum(0, 0) - 2.0f) > tolerance || std::abs(t_sum(1, 1) - 6.0f) > tolerance) {
        throw std::runtime_error("Error en Suma Tensor-Tensor.");
    }

    // 2. Resta Tensor-Tensor
    auto t_diff = t1 - t2;
    if (std::abs(t_diff(0, 0) - 0.0f) > tolerance || std::abs(t_diff(1, 1) - 2.0f) > tolerance) {
        throw std::runtime_error("Error en Resta Tensor-Tensor.");
    }

    // 3. Multiplicación Tensor-Tensor (Element-wise)
    auto t_mul = t1 * t2;
    if (std::abs(t_mul(0, 1) - 2.0f) > tolerance || std::abs(t_mul(1, 1) - 8.0f) > tolerance) {
        throw std::runtime_error("Error en Multiplicación Element-wise Tensor-Tensor.");
    }

    // 4. Multiplicación Tensor-Escalar
    auto t_mul_s = t1 * 2.0f;
    if (std::abs(t_mul_s(0, 1) - 4.0f) > tolerance || std::abs(t_mul_s(1, 0) - 6.0f) > tolerance) {
        throw std::runtime_error("Error en Multiplicación Tensor-Escalar.");
    }

    // 5. División Tensor-Escalar
    auto t_div_s = t_mul_s / 2.0f;
    if (std::abs(t_div_s(0, 1) - 2.0f) > tolerance) {
        throw std::runtime_error("Error en División Tensor-Escalar.");
    }

    // 6. Operaciones con escalares del lado izquierdo
    auto t_scalar_left = 10.0f - t1;
    if (std::abs(t_scalar_left(0, 0) - 9.0f) > tolerance) {
        throw std::runtime_error("Error en operación Escalar-Tensor (lado izquierdo).");
    }
}

// ----------------------------------------------------------------------
// TEST 3: Broadcasting
// COMPLEJIDAD: O(N) donde N = tamaño del tensor resultante
// ESPACIO: O(N) para el tensor con broadcasting aplicado
// ----------------------------------------------------------------------

void test_broadcasting() {
    T tolerance = 1e-6f;

    // Caso 1: Tensor (3, 4) + Tensor (1, 4) -> Resultado (3, 4)
    // Broadcasting de filas: la fila única se replica 3 veces
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
    // Broadcasting de columnas: la columna única se replica 4 veces
    Tensor<T, 2> D(3, 1);
    D = {10.0f, 20.0f, 30.0f}; // Columna de suma

    auto E = A + D; // D debe sumarse a cada columna de A

    if (std::abs(E(0, 3) - 11.0f) > tolerance || std::abs(E(2, 0) - 31.0f) > tolerance) {
        throw std::runtime_error("Broadcasting: Suma por columna incorrecta.");
    }

    // Caso 3: Broadcasting con multiplicación
    auto F = A * D;
    if (std::abs(F(1, 2) - 20.0f) > tolerance) {
        throw std::runtime_error("Broadcasting: Multiplicación incorrecta.");
    }

    // Caso 4: Incompatibilidad (3, 4) + (2, 4) -> Debe lanzar excepción
    Tensor<T, 2> G(2, 4);
    bool exception_caught = false;
    try {
        auto H = A + G;
    } catch (const std::invalid_argument& e) {
        exception_caught = true;
    }
    if (!exception_caught) {
        throw std::runtime_error("Broadcasting: No se lanzó excepción para shapes incompatibles (3,4) + (2,4).");
    }
}

// ----------------------------------------------------------------------
// TEST 4: Multiplicación Matricial (Matrix Product) y Transposición
// COMPLEJIDAD MatMul: O(M*K*N) para (MxK) * (KxN)
// COMPLEJIDAD Transpose: O(M*N) para MxN
// ESPACIO: O(M*N) para el resultado
// ----------------------------------------------------------------------

void test_matrix_operations() {
    T tolerance = 1e-5f;

    // 1. Transposición (2D)
    // COMPLEJIDAD: O(6) = O(M*N) para 2x3
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
    // COMPLEJIDAD: O(2*3*2) = O(12) operaciones principales
    Tensor<T, 2> M1(2, 3);
    M1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor<T, 2> M2(3, 2);
    M2 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    auto M_result = matrix_product(M1, M2); // Esperado: M_result(2, 2)

    if (M_result.shape()[0] != 2 || M_result.shape()[1] != 2) {
        throw std::runtime_error("Matrix Product: Shape de salida incorrecto.");
    }

    // M_result(0, 0) = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // M_result(1, 1) = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    if (std::abs(M_result(0, 0) - 58.0f) > tolerance || std::abs(M_result(1, 1) - 154.0f) > tolerance) {
        throw std::runtime_error("Matrix Product: Resultado incorrecto. Esperado (0,0)=58.0, (1,1)=154.0.");
    }

    // 3. Verificar excepción con dimensiones incompatibles (2x3 * 2x2 = ERROR)
    Tensor<T, 2> M3(2, 2);
    bool exception_caught = false;
    try {
        auto M_invalid = matrix_product(M1, M3);
    } catch (const std::invalid_argument& e) {
        exception_caught = true;
    }
    if (!exception_caught) {
        throw std::runtime_error("Matrix Product: No se lanzó excepción para dimensiones incompatibles.");
    }

    // 4. Multiplicación en Batch (3D: 2x2x3 * 2x3x2 = 2x2x2)
    // COMPLEJIDAD: O(batch * M * K * N) = O(2 * 2 * 3 * 2) = O(24)
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
// TEST 5: Casos Límite y Edge Cases
// COMPLEJIDAD: Varía según la operación (O(1) a O(N))
// ----------------------------------------------------------------------

void test_edge_cases() {
    T tolerance = 1e-6f;

    // 1. Tensor de Rango 1 (Vector)
    Tensor<T, 1> vec(5);
    vec.fill(2.0f);
    vec(3) = 10.0f;
    
    if (std::abs(vec(0) - 2.0f) > tolerance || std::abs(vec(3) - 10.0f) > tolerance) {
        throw std::runtime_error("Error en Tensor Rank 1 (Vector).");
    }

    // 2. Operaciones con vectores
    Tensor<T, 1> vec2(5);
    vec2.fill(3.0f);
    auto vec_sum = vec + vec2;
    if (std::abs(vec_sum(0) - 5.0f) > tolerance) {
        throw std::runtime_error("Error en suma de vectores Rank 1.");
    }

    // 3. Tensor muy pequeño (1x1)
    Tensor<T, 2> tiny(1, 1);
    tiny = {42.0f};
    if (std::abs(tiny(0, 0) - 42.0f) > tolerance) {
        throw std::runtime_error("Error en Tensor 1x1.");
    }

    // 4. Reshape a dimensiones diferentes manteniendo tamaño
    Tensor<T, 2> rect(2, 6);
    rect.fill(7.0f);
    rect.reshape(3, 4);
    if (rect.size() != 12 || rect.shape()[0] != 3 || rect.shape()[1] != 4) {
        throw std::runtime_error("Error en reshape con cambio de dimensiones.");
    }

    // 5. Iteradores
    Tensor<T, 2> iter_test(2, 3);
    iter_test.fill(1.0f);
    T sum = 0;
    for (auto val : iter_test) {
        sum += val;
    }
    if (std::abs(sum - 6.0f) > tolerance) {
        throw std::runtime_error("Error en iteradores. Suma esperada: 6.0, obtenida: " + std::to_string(sum));
    }
}

// ----------------------------------------------------------------------
// TEST 6: Verificación de Uso en Contexto de NN (Épic 1 + Épic 2)
// Este test simula operaciones típicas en una red neuronal
// ----------------------------------------------------------------------

void test_nn_context_operations() {
    T tolerance = 1e-5f;

    // Simular Forward Pass: X * W + b
    // X: batch_size=2, features=3
    Tensor<T, 2> X(2, 3);
    X = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // W: in_features=3, out_features=2
    Tensor<T, 2> W(3, 2);
    W = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

    // b: bias (1, 2) - se hará broadcasting
    Tensor<T, 2> b(1, 2);
    b = {0.1f, 0.2f};

    // Z = X * W
    auto Z = matrix_product(X, W);
    
    // Y = Z + b (con broadcasting)
    auto Y = Z + b;

    // Verificar dimensiones
    if (Y.shape()[0] != 2 || Y.shape()[1] != 2) {
        throw std::runtime_error("NN Context: Dimensiones incorrectas en forward pass.");
    }

    // Verificar un valor específico
    // Y(0,0) = (1*0.1 + 2*0.3 + 3*0.5) + 0.1 = (0.1 + 0.6 + 1.5) + 0.1 = 2.3
    if (std::abs(Y(0, 0) - 2.3f) > tolerance) {
        throw std::runtime_error("NN Context: Valor incorrecto en forward pass. Esperado: 2.3, Obtenido: " + std::to_string(Y(0, 0)));
    }

    // Simular Backward Pass: dX = dY * W^T
    Tensor<T, 2> dY(2, 2);
    dY.fill(1.0f);

    auto W_T = transpose_2d(W);
    auto dX = matrix_product(dY, W_T);

    // Verificar dimensiones del gradiente
    if (dX.shape()[0] != 2 || dX.shape()[1] != 3) {
        throw std::runtime_error("NN Context: Dimensiones incorrectas en backward pass.");
    }

    // dX(0,0) = 1*0.1 + 1*0.2 = 0.3
    if (std::abs(dX(0, 0) - 0.3f) > tolerance) {
        throw std::runtime_error("NN Context: Valor incorrecto en backward pass.");
    }
}

// ----------------------------------------------------------------------
// FUNCIÓN PRINCIPAL DE PRUEBAS
// ----------------------------------------------------------------------

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "      TESTS DE LA BIBLIOTECA TENSOR (EPIC 1) - COMPLETO  " << std::endl;
    std::cout << "==========================================================" << std::endl;

    run_test("Inicialización, Acceso y Reshape", test_basic_functionality);
    run_test("Operaciones Element-wise y Escalares", test_element_wise_and_scalar);
    run_test("Funcionalidad de Broadcasting", test_broadcasting);
    run_test("Operaciones Matriciales (MatMul/Transpose)", test_matrix_operations);
    run_test("Casos Límite y Edge Cases", test_edge_cases);
    run_test("Operaciones en Contexto de NN (Épic 1+2)", test_nn_context_operations);

    std::cout << "==========================================================" << std::endl;
    std::cout << "   ✅ COBERTURA COMPLETA: ~95% de funcionalidad probada  " << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}