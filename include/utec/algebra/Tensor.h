
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

namespace utec {
namespace algebra {

template <typename T, size_t N>
class Tensor {
private:
    std::vector<T> data_;
    std::array<size_t, N> shape_;
    std::array<size_t, N> strides_;

    void compute_strides() {
        if (N == 0) return;
        strides_[N - 1] = 1;
        for (int i = static_cast<int>(N) - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    size_t compute_total_size() const {
        size_t total = 1;
        for (size_t i = 0; i < N; ++i) {
            total *= shape_[i];
        }
        return total;
    }

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

    bool can_broadcast_with(const Tensor<T, N>& other) const {
        for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
            if (shape_[i] != other.shape_[i] && shape_[i] != 1 && other.shape_[i] != 1) {
                return false;
            }
        }
        return true;
    }

    std::array<size_t, N> broadcast_shape(const Tensor<T, N>& other) const {
        std::array<size_t, N> result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = std::max(shape_[i], other.shape_[i]);
        }
        return result;
    }

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
    // Constructor con dimensiones variables
    template <typename... Dims, typename = std::enable_if_t<std::conjunction_v<std::is_integral<Dims>...>>>
    explicit Tensor(Dims... dims) {
        static_assert(sizeof...(dims) == N, "Number of dimensions do not match with template rank N");
        shape_ = {static_cast<size_t>(dims)...};
        compute_strides();
        data_.resize(compute_total_size(), T{});
    }

    // Constructor desde un std::array para el shape
    explicit Tensor(const std::array<size_t, N>& shape)
        : shape_(shape) {
        compute_strides();
        data_.resize(compute_total_size(), T{});
    }

    // Default constructor
    Tensor() {
        shape_.fill(0);
        compute_strides();
        // data_ is implicitly empty, which is correct for a 0-sized tensor
    }

    // Constructor de copia
    Tensor(const Tensor& other)
        : data_(other.data_), shape_(other.shape_) {
        compute_strides();
    }

    // Operador de asignación
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            data_ = other.data_;
            shape_ = other.shape_;
            compute_strides();
        }
        return *this;
    }

    // Operador de asignación desde initializer_list
    Tensor& operator=(std::initializer_list<T> list) {
        if (list.size() != data_.size()) {
            throw std::invalid_argument("Data size does not match tensor size");
        }
        std::copy(list.begin(), list.end(), data_.begin());
        return *this;
    }

    // Obtener shape
    std::array<size_t, N> shape() const {
        return shape_;
    }

    // Fill
    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    // Acceso con múltiples índices (const)
    template <typename... Indices>
    const T& operator()(Indices... indices) const {
        constexpr size_t num_indices = sizeof...(Indices);
        if (num_indices != N) {
            throw std::out_of_range("Invalid number of indices");
        }
        return data_[compute_index(indices...)];
    }

    // Acceso con múltiples índices (no const)
    template <typename... Indices>
    T& operator()(Indices... indices) {
        constexpr size_t num_indices = sizeof...(Indices);
        if (num_indices != N) {
            throw std::out_of_range("Invalid number of indices");
        }
        return data_[compute_index(indices...)];
    }

    // Reshape con dimensiones variables
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
            new_size *= new_shape[i];
        }

        // El comportamiento correcto es redimensionar siempre
        data_.resize(new_size, T{});

        shape_ = new_shape;
        compute_strides();
    }

    // Operador de suma
    Tensor operator+(const Tensor& other) const {
        if (shape_ == other.shape_) {
            Tensor result(*this);
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] += other.data_[i];
            }
            return result;
        }

        if (!can_broadcast_with(other)) {
            throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }

        auto result_shape = broadcast_shape(other);
        size_t result_size = 1;
        for (size_t i = 0; i < N; ++i) {
            result_size *= result_shape[i];
        }

        Tensor result = create_with_shape(result_shape);
        for (size_t i = 0; i < result_size; ++i) {
            size_t idx1 = get_broadcasted_index(i, result_shape);
            size_t idx2 = other.get_broadcasted_index(i, result_shape);
            result.data_[i] = data_[idx1] + other.data_[idx2];
        }

        return result;
    }

    // Operador de resta
    Tensor operator-(const Tensor& other) const {
        if (shape_ == other.shape_) {
            Tensor result(*this);
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] -= other.data_[i];
            }
            return result;
        }

        if (!can_broadcast_with(other)) {
            throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }

        auto result_shape = broadcast_shape(other);
        size_t result_size = 1;
        for (size_t i = 0; i < N; ++i) {
            result_size *= result_shape[i];
        }

        Tensor result = create_with_shape(result_shape);
        for (size_t i = 0; i < result_size; ++i) {
            size_t idx1 = get_broadcasted_index(i, result_shape);
            size_t idx2 = other.get_broadcasted_index(i, result_shape);
            result.data_[i] = data_[idx1] - other.data_[idx2];
        }

        return result;
    }

    // Operador de multiplicación elemento a elemento
    Tensor operator*(const Tensor& other) const {
        if (shape_ == other.shape_) {
            Tensor result(*this);
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] *= other.data_[i];
            }
            return result;
        }

        if (!can_broadcast_with(other)) {
            throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }

        auto result_shape = broadcast_shape(other);
        size_t result_size = 1;
        for (size_t i = 0; i < N; ++i) {
            result_size *= result_shape[i];
        }

        Tensor result = create_with_shape(result_shape);
        for (size_t i = 0; i < result_size; ++i) {
            size_t idx1 = get_broadcasted_index(i, result_shape);
            size_t idx2 = other.get_broadcasted_index(i, result_shape);
            result.data_[i] = data_[idx1] * other.data_[idx2];
        }

        return result;
    }

    // Operaciones con escalares
    Tensor operator+(const T& scalar) const {
        Tensor result(*this);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] += scalar;
        }
        return result;
    }

    Tensor operator-(const T& scalar) const {
        Tensor result(*this);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] -= scalar;
        }
        return result;
    }

    Tensor operator*(const T& scalar) const {
        Tensor result(*this);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] *= scalar;
        }
        return result;
    }

    Tensor operator/(const T& scalar) const {
        Tensor result(*this);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] /= scalar;
        }
        return result;
    }

    // Operaciones con escalares (amigo para operador izquierdo)
    friend Tensor operator+(const T& scalar, const Tensor& tensor) {
        return tensor + scalar;
    }

    friend Tensor operator-(const T& scalar, const Tensor& tensor) {
        Tensor result(tensor);
        for (size_t i = 0; i < tensor.data_.size(); ++i) {
            result.data_[i] = scalar - tensor.data_[i];
        }
        return result;
    }

    friend Tensor operator*(const T& scalar, const Tensor& tensor) {
        return tensor * scalar;
    }

    friend Tensor operator/(const T& scalar, const Tensor& tensor) {
        Tensor result(tensor);
        for (size_t i = 0; i < tensor.data_.size(); ++i) {
            result.data_[i] = scalar / tensor.data_[i];
        }
        return result;
    }

    // Iteradores
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
    auto cbegin() const { return data_.cbegin(); }
    auto cend() const { return data_.cend(); }

    // Operador de salida
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        tensor.print(os, 0, 0);
        return os;
    }

    // Obtener tamaño total
    size_t size() const {
        return data_.size();
    }

    // Método para acceder a los datos crudos (necesario para serialización)
    T* data() {
        return data_.data();
    }

    // Versión const del método data()
    const T* data() const {
        return data_.data();
    }

private:
    void print(std::ostream& os, size_t dim, size_t offset) const {
        if (dim == N - 1) {
            for (size_t i = 0; i < shape_[dim]; ++i) {
                if (i > 0) os << " ";
                os << data_[offset + i];
            }
        } else {
            os << "{\n";
            for (size_t i = 0; i < shape_[dim]; ++i) {
                for (size_t j = 0; j < dim + 1; ++j) os << " ";
                print(os, dim + 1, offset + i * strides_[dim]);
                os << "\n";
            }
            for (size_t j = 0; j < dim; ++j) os << " ";
            os << "}";
        }
    }

    static Tensor create_with_shape(const std::array<size_t, N>& new_shape) {
        return create_with_shape_impl(new_shape, std::make_index_sequence<N>{});
    }

    template <size_t... Is>
    static Tensor create_with_shape_impl(const std::array<size_t, N>& new_shape, std::index_sequence<Is...>) {
        return Tensor(new_shape[Is]...); // Esta línea es correcta, pero la forma de llamarla no lo era.
    }
};

// Función transpose_2d
template <typename T, size_t N>
Tensor<T, N> transpose_2d(const Tensor<T, N>& tensor) {
    if (N < 2) {
        throw std::invalid_argument("Cannot transpose a tensor with rank less than 2.");
    }

    auto shape = tensor.shape();
    std::swap(shape[N - 2], shape[N - 1]);
    Tensor<T, N> result(shape);

    if constexpr (N == 2) {
        for (size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.shape()[1]; ++j) {
                result(j, i) = tensor(i, j);
            }
        }
    } else {
        // Generic transpose for last two dimensions
        size_t rows = tensor.shape()[N - 2];
        size_t cols = tensor.shape()[N - 1];
        size_t batch_size = tensor.size() / (rows * cols);
        
        std::vector<size_t> current_pos(N);
        for(size_t i = 0; i < tensor.size(); ++i) {
            size_t temp_idx = i;
            for(int d = N - 1; d >= 0; --d) {
                current_pos[d] = temp_idx % tensor.shape()[d];
                temp_idx /= tensor.shape()[d];
            }
            
            std::swap(current_pos[N - 2], current_pos[N - 1]);
            
            size_t result_idx = 0;
            for(size_t d = 0; d < N; ++d) {
                result_idx += current_pos[d] * result.strides_[d];
            }
            result.data_[result_idx] = tensor.data_[i];
        }
    }
    return result;
}

// Función matrix_product
template <typename T, size_t N>
Tensor<T, N> matrix_product(const Tensor<T, N>& t1, const Tensor<T, N>& t2) {
    if (N < 2) {
        throw std::invalid_argument("Matrix multiplication requires at least 2 dimensions");
    }

    auto shape1 = t1.shape();
    auto shape2 = t2.shape();

    // Verificar que las dimensiones de matriz sean compatibles
    if (shape1[N - 1] != shape2[N - 2]) {
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
    }

    // Verificar que las dimensiones batch coincidan
    for (size_t i = 0; i < N - 2; ++i) {
        if (shape1[i] != shape2[i]) {
            throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
        }
    }

    // Crear tensor resultado
    std::array<size_t, N> result_shape = shape1;
    result_shape[N - 1] = shape2[N - 1];

    auto result = Tensor<T, N>(result_shape);

    // Calcular el número de matrices en el batch
    size_t batch_size = 1;
    for (size_t i = 0; i < N - 2; ++i) {
        batch_size *= shape1[i];
    }

    size_t M = shape1[N - 2];  // filas de t1
    size_t K = shape1[N - 1];  // cols de t1 / filas de t2
    size_t L = shape2[N - 1];  // cols de t2

    // Para cada matriz en el batch
    for (size_t b = 0; b < batch_size; ++b) {
        size_t offset1 = b * M * K;
        size_t offset2 = b * K * L;
        size_t offset_result = b * M * L;

        // Multiplicación de matrices
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < L; ++j) {
                T sum = T{};
                for (size_t k = 0; k < K; ++k) {
                    auto it1 = t1.begin();
                    auto it2 = t2.begin();
                    std::advance(it1, offset1 + i * K + k);
                    std::advance(it2, offset2 + k * L + j);
                    sum += (*it1) * (*it2);
                }
                auto it_result = result.begin();
                std::advance(it_result, offset_result + i * L + j);
                *it_result = sum;
            }
        }
    }

    return result;
}

} // namespace algebra
} // namespace utec

#endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H