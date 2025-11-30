//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H

#include <array>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <initializer_list>
#include <sstream>

/**
 * =================================================================
 * NOTACIÓN DE COMPLEJIDAD ALGORÍTMICA (O)
 * =================================================================
 * N: Rango del Tensor (número de dimensiones).
 * S: Tamaño total del Tensor (número de elementos, producto de todas las dimensiones).
 * S_res: Tamaño del Tensor resultado después de aplicar Broadcasting.
 * B: Tamaño del lote (Batch Size) = Producto de shape[0] * ... * shape[N-3].
 * M: Filas de la submatriz (shape[N-2]).
 * K: Dimensión común para la multiplicación matricial (shape1[N-1] = shape2[N-2]).
 * L: Columnas de la submatriz (shape[N-1]).
 * C_mat_mul: Costo de una Multiplicación Matricial por Lotes: O(B * M * K * L).
 * =================================================================
 */

namespace utec {
namespace algebra {

template <typename T, size_t N>
class Tensor {
private:
    std::vector<T> data_;
    std::array<size_t, N> shape_;
    std::array<size_t, N> strides_;

    // --- Algoritmo de Cálculo de Strides ---
    // Complejidad: O(N) - Lineal con respecto al rango (número de dimensiones).
    void compute_strides() {
        if (N == 0) return;
        strides_[N - 1] = 1;
        for (int i = static_cast<int>(N) - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    // --- Algoritmo de Cálculo de Tamaño Total ---
    // Complejidad: O(N) - Lineal con respecto al rango (número de dimensiones).
    size_t compute_total_size() const {
        size_t total = 1;
        for (size_t i = 0; i < N; ++i) {
            total *= shape_[i];
        }
        return total;
    }

    // --- Algoritmo de Cálculo de Índice Plano (flat index) ---
    // Complejidad: O(N) - Lineal con respecto al rango (número de dimensiones).
    template <typename... Indices>
    size_t compute_index(size_t first, Indices... rest) const {
        constexpr size_t num_indices = sizeof...(Indices) + 1;
        std::array<size_t, num_indices> indices = {first, static_cast<size_t>(rest)...};
        size_t index = 0;
        for (size_t i = 0; i < num_indices; ++i) {
            index += indices[i] * strides_[i];
        }
        return index;
    }

    // --- Algoritmo de Verificación de Broadcasting ---
    // Complejidad: O(N) - Lineal con respecto al rango.
    bool can_broadcast_with(const Tensor<T, N>& other) const {
        for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
            if (shape_[i] != other.shape_[i] && shape_[i] != 1 && other.shape_[i] != 1) {
                return false;
            }
        }
        return true;
    }

    // --- Algoritmo de Cálculo de Shape de Broadcasting ---
    // Complejidad: O(N) - Lineal con respecto al rango.
    std::array<size_t, N> broadcast_shape(const Tensor<T, N>& other) const {
        std::array<size_t, N> result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = std::max(shape_[i], other.shape_[i]);
        }
        return result;
    }

    // --- Algoritmo de Obtención de Índice Broadcasted ---
    // Complejidad: O(N) - El bucle para obtener las coordenadas y el bucle para calcular el índice plano son O(N).
    size_t get_broadcasted_index(size_t flat_index, const std::array<size_t, N>& result_shape) const {
        std::array<size_t, N> result_indices;
        size_t temp = flat_index;
        for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
            result_indices[i] = temp % result_shape[i];
            temp /= result_shape[i];
        }

        size_t index = 0;
        for (size_t i = 0; i < N; ++i) {
            size_t idx = (shape_[i] == 1) ? 0 : result_indices[i];
            index += idx * strides_[i];
        }
        return index;
    }

    // Declaraciones friend para funciones externas
    template <typename U, size_t M>
    friend Tensor<U, M> transpose_2d(const Tensor<U, M>& tensor);

    template <typename U, size_t M>
    friend Tensor<U, M> matrix_product(const Tensor<U, M>& t1, const Tensor<U, M>& t2);

public:
    // --- Constructor Principal ---
    // Complejidad: O(S + N), dominado por la inicialización/redimensionamiento de 'data_' (O(S)).
    template <typename... Dims, typename = std::enable_if_t<std::conjunction_v<std::is_integral<Dims>...>>>
    explicit Tensor(Dims... dims) {
        static_assert(sizeof...(dims) == N, "Number of dimensions do not match with template rank N");
        shape_ = {static_cast<size_t>(dims)...};
        compute_strides(); // O(N)
        data_.resize(compute_total_size(), T{}); // O(S)
    }

    // --- Constructor de Copia ---
    // Complejidad: O(S + N), dominado por la copia de 'data_' (O(S)).
    Tensor(const Tensor& other)
        : data_(other.data_), shape_(other.shape_) { // O(S)
        compute_strides(); // O(N)
    }

    // --- Operador de Asignación ---
    // Complejidad: O(S + N), dominado por la copia de 'data_' (O(S)).
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            data_ = other.data_; // O(S)
            shape_ = other.shape_;
            compute_strides(); // O(N)
        }
        return *this;
    }

    // --- Operador de Asignación desde initializer_list ---
    // Complejidad: O(S) - Lineal con respecto al número total de elementos.
    Tensor& operator=(std::initializer_list<T> list) {
        if (list.size() != data_.size()) {
            throw std::invalid_argument("Data size does not match tensor size");
        }
        std::copy(list.begin(), list.end(), data_.begin()); // O(S)
        return *this;
    }

    // --- Fill ---
    // Complejidad: O(S) - Lineal con respecto al número total de elementos.
    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value); // O(S)
    }

    // --- Acceso con Múltiples Índices (operator()) ---
    // Complejidad: O(N) - Lineal con respecto al rango (calcula el índice plano).
    template <typename... Indices>
    const T& operator()(Indices... indices) const {
        constexpr size_t num_indices = sizeof...(Indices);
        if (num_indices != N) {
            throw std::out_of_range("Invalid number of indices");
        }
        return data_[compute_index(indices...)]; // O(N)
    }

    // --- Reshape ---
    // Complejidad: O(S' + N), donde S' es el nuevo tamaño total. Dominado por O(S').
    template <typename... Dims>
    void reshape(Dims... dims) {
        constexpr size_t num_dims = sizeof...(Dims);
        if (num_dims != N) {
            throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(N));
        }

        std::array<size_t, num_dims> temp_shape = {static_cast<size_t>(dims)...};
        std::array<size_t, N> new_shape;
        size_t idx = 0;
        for (auto dim : temp_shape) {
            if (idx < N) new_shape[idx++] = dim;
        }

        size_t new_size = 1;
        for (size_t i = 0; i < N; ++i) {
            new_size *= new_shape[i]; // O(N)
        }

        data_.resize(new_size, T{}); // O(S')
        shape_ = new_shape;
        compute_strides(); // O(N)
    }

    // --- Operador de Suma (operator+) ---
    // Complejidad: O(S) si las formas coinciden (sin broadcasting).
    // Complejidad: O(S_res * N) con broadcasting, donde S_res es el tamaño del resultado y N es el coste de `get_broadcasted_index`.
    Tensor operator+(const Tensor& other) const {
        if (shape_ == other.shape_) {
            Tensor result(*this);
            for (size_t i = 0; i < data_.size(); ++i) { // O(S)
                result.data_[i] += other.data_[i];
            }
            return result;
        }

        if (!can_broadcast_with(other)) { // O(N)
            throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }

        auto result_shape = broadcast_shape(other); // O(N)
        size_t result_size = 1;
        for (size_t i = 0; i < N; ++i) { // O(N)
            result_size *= result_shape[i];
        }

        Tensor result = create_with_shape(result_shape);
        for (size_t i = 0; i < result_size; ++i) { // O(S_res * N)
            size_t idx1 = get_broadcasted_index(i, result_shape); // O(N)
            size_t idx2 = other.get_broadcasted_index(i, result_shape); // O(N)
            result.data_[i] = data_[idx1] + other.data_[idx2];
        }

        return result;
    }

    // --- Operador de Resta (operator-) ---
    // Complejidad: O(S) sin broadcasting. O(S_res * N) con broadcasting.
    Tensor operator-(const Tensor& other) const {
        if (shape_ == other.shape_) {
            Tensor result(*this);
            for (size_t i = 0; i < data_.size(); ++i) { // O(S)
                result.data_[i] -= other.data_[i];
            }
            return result;
        }

        // ... (Lógica de broadcasting - O(S_res * N)) ...
        if (!can_broadcast_with(other)) {
            throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }

        auto result_shape = broadcast_shape(other);
        size_t result_size = 1;
        for (size_t i = 0; i < N; ++i) {
            result_size *= result_shape[i];
        }

        Tensor result = create_with_shape(result_shape);
        for (size_t i = 0; i < result_size; ++i) { // O(S_res * N)
            size_t idx1 = get_broadcasted_index(i, result_shape);
            size_t idx2 = other.get_broadcasted_index(i, result_shape);
            result.data_[i] = data_[idx1] - other.data_[idx2];
        }

        return result;
    }

    // --- Operador de Multiplicación Element-wise (operator*) ---
    // Complejidad: O(S) sin broadcasting. O(S_res * N) con broadcasting.
    Tensor operator*(const Tensor& other) const {
        if (shape_ == other.shape_) {
            Tensor result(*this);
            for (size_t i = 0; i < data_.size(); ++i) { // O(S)
                result.data_[i] *= other.data_[i];
            }
            return result;
        }

        // ... (Lógica de broadcasting - O(S_res * N)) ...
        if (!can_broadcast_with(other)) {
            throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }

        auto result_shape = broadcast_shape(other);
        size_t result_size = 1;
        for (size_t i = 0; i < N; ++i) {
            result_size *= result_shape[i];
        }

        Tensor result = create_with_shape(result_shape);
        for (size_t i = 0; i < result_size; ++i) { // O(S_res * N)
            size_t idx1 = get_broadcasted_index(i, result_shape);
            size_t idx2 = other.get_broadcasted_index(i, result_shape);
            result.data_[i] = data_[idx1] * other.data_[idx2];
        }

        return result;
    }

    // --- Operaciones con Escalares ---
    // Complejidad: O(S) para todas las operaciones con escalares (recorren todos los elementos).

    Tensor operator+(const T& scalar) const {
        Tensor result(*this);
        for (size_t i = 0; i < data_.size(); ++i) { // O(S)
            result.data_[i] += scalar;
        }
        return result;
    }

    // ... (Resto de operadores con escalares - O(S)) ...

    // Operaciones con escalares (amigo para operador izquierdo)
    friend Tensor operator+(const T& scalar, const Tensor& tensor) {
        return tensor + scalar; // O(S)
    }

    friend Tensor operator-(const T& scalar, const Tensor& tensor) {
        Tensor result(tensor);
        for (size_t i = 0; i < tensor.data_.size(); ++i) { // O(S)
            result.data_[i] = scalar - tensor.data_[i];
        }
        return result;
    }

    friend Tensor operator*(const T& scalar, const Tensor& tensor) {
        return tensor * scalar; // O(S)
    }

    friend Tensor operator/(const T& scalar, const Tensor& tensor) {
        Tensor result(tensor);
        for (size_t i = 0; i < tensor.data_.size(); ++i) { // O(S)
            result.data_[i] = scalar / tensor.data_[i];
        }
        return result;
    }

    // ... (Iteradores, Operador de salida, size(), data() - Complejidad de O(1) o implícita en O(S))

private:
    // --- Algoritmo de Impresión Recursiva ---
    // Complejidad: O(S) - Se debe visitar y procesar cada uno de los S elementos.
    void print(std::ostream& os, size_t dim, size_t offset) const {
        if (dim == N - 1) {
            for (size_t i = 0; i < shape_[dim]; ++i) { // O(shape_[N-1])
                if (i > 0) os << " ";
                os << data_[offset + i];
            }
        } else {
            os << "{\n";
            for (size_t i = 0; i < shape_[dim]; ++i) { // Llamadas recursivas
                // ... (impresión) ...
                print(os, dim + 1, offset + i * strides_[dim]);
                os << "\n";
            }
            // ... (impresión) ...
            os << "}";
        }
    }
    // ... (Métodos estáticos internos) ...
};

// =================================================================

// --- Función Transpose 2D ---
// Complejidad: O(S * N)
// Se recorren todos los S elementos. La operación de acceso o cálculo de índice (result(j, i) o el bucle interno
// para calcular el índice en el caso genérico) toma O(N).
template <typename T, size_t N>
Tensor<T, N> transpose_2d(const Tensor<T, N>& tensor) {
    if (N < 2) {
        throw std::invalid_argument("Cannot transpose a tensor with rank less than 2.");
    }

    auto shape = tensor.shape(); // O(1)
    std::swap(shape[N - 2], shape[N - 1]);
    Tensor<T, N> result(shape); // O(S)

    if constexpr (N == 2) {
        size_t rows = tensor.shape()[0];
        size_t cols = tensor.shape()[1];
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) { // O(S) iteraciones en total
                // Acceso O(N) para 2D, aquí O(2) = O(1)
                result(j, i) = tensor(i, j);
            }
        }
    } else {
        // Transposición genérica para las últimas dos dimensiones
        size_t rows = tensor.shape()[N - 2];
        size_t cols = tensor.shape()[N - 1];
        size_t batch_size = tensor.size() / (rows * cols);
        
        std::vector<size_t> current_pos(N);
        for(size_t i = 0; i < tensor.size(); ++i) { // O(S) iteraciones
            // Conversión de índice plano a coordenadas: O(N)
            size_t temp_idx = i;
            for(int d = N - 1; d >= 0; --d) { 
                current_pos[d] = temp_idx % tensor.shape()[d];
                temp_idx /= tensor.shape()[d];
            }
            
            std::swap(current_pos[N - 2], current_pos[N - 1]);
            
            // Conversión de coordenadas a índice plano (usando strides): O(N)
            size_t result_idx = 0;
            for(size_t d = 0; d < N; ++d) {
                result_idx += current_pos[d] * result.strides_[d];
            }
            result.data_[result_idx] = tensor.data_[i]; // O(1)
        }
        // Complejidad total: O(S * (N + N)) = O(S * N)
    }
    return result;
}

// =================================================================

// --- Función Matrix Product (Batch Matrix Multiplication) ---
// Complejidad: O(B * M * K * L)
// Donde: B = batch_size, M = filas t1, K = dim común, L = cols t2.
template <typename T, size_t N>
Tensor<T, N> matrix_product(const Tensor<T, N>& t1, const Tensor<T, N>& t2) {
    // ... (validaciones - O(N)) ...

    auto shape1 = t1.shape();
    auto shape2 = t2.shape();

    // ... (verificaciones de compatibilidad - O(N)) ...

    // Crear tensor resultado: O(B*M*L)
    std::array<size_t, N> result_shape = shape1;
    result_shape[N - 1] = shape2[N - 1];
    auto result = Tensor<T, N>(result_shape);

    // Calcular el número de matrices en el batch: O(N)
    size_t batch_size = 1;
    for (size_t i = 0; i < N - 2; ++i) {
        batch_size *= shape1[i];
    }

    size_t M = shape1[N - 2];  // filas de t1
    size_t K = shape1[N - 1];  // cols de t1 / filas de t2
    size_t L = shape2[N - 1];  // cols de t2

    // Bucle batch: O(B)
    for (size_t b = 0; b < batch_size; ++b) {
        size_t offset1 = b * M * K;
        size_t offset2 = b * K * L;
        size_t offset_result = b * M * L;

        // Multiplicación de matrices: O(M*L*K)
        for (size_t i = 0; i < M; ++i) { // Bucle de filas: O(M)
            for (size_t j = 0; j < L; ++j) { // Bucle de columnas: O(L)
                T sum = T{};
                for (size_t k = 0; k < K; ++k) { // Bucle de dimensión común: O(K)
                    // Acceso a elementos en data_ es O(1) ya que se usa offset y std::advance sobre vector
                    auto it1 = t1.begin();
                    auto it2 = t2.begin();
                    std::advance(it1, offset1 + i * K + k); // O(1)
                    std::advance(it2, offset2 + k * L + j); // O(1)
                    sum += (*it1) * (*it2);
                }
                auto it_result = result.begin();
                std::advance(it_result, offset_result + i * L + j); // O(1)
                *it_result = sum;
            }
        }
        // Complejidad total: O(B * M * L * K)
    }

    return result;
}

} // namespace algebra
} // namespace utec

#endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
