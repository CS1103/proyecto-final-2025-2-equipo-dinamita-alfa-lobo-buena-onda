#ifndef NN_DENSE_H
#define NN_DENSE_H

#include <utec/nn/nn_interfaces.h>
#include <utec/algebra/Tensor.h>
#include <fstream> 

/**
 * =================================================================
 * NOTACIÓN DE COMPLEJIDAD ALGORÍTMICA (O)
 * =================================================================
 * S_batch: Tamaño del batch actual (número de muestras).
 * M_in: Número de características de entrada (in_features_).
 * M_out: Número de neuronas de salida (out_features_).
 * P_layer: Número total de parámetros de la capa (pesos + sesgos).
 * C_mat_mul: Costo de una multiplicación matricial clave: O(S_batch * M_in * M_out).
 * =================================================================
 */

namespace utec {
namespace neural_network {

// Capa Dense (Fully Connected): Y = X * W + b
template<typename T>
class Dense final : public ILayer<T> {
private:
    utec::algebra::Tensor<T, 2> weights_;      // Matriz de pesos (M_in x M_out)
    utec::algebra::Tensor<T, 2> biases_;       // Vector de sesgos (1 x M_out)

    utec::algebra::Tensor<T, 2> input_;        // Guardamos entrada para backward (S_batch x M_in)
    utec::algebra::Tensor<T, 2> grad_weights_; // Gradiente de pesos (M_in x M_out)
    utec::algebra::Tensor<T, 2> grad_biases_;  // Gradiente de sesgos (1 x M_out)

    size_t in_features_; // M_in
    size_t out_features_; // M_out

    // --- FUNCIONES AUXILIARES DE SERIALIZACIÓN ---

    // Guarda las dimensiones y el contenido binario de un tensor 2D
    // Complejidad: O(N_elements). Para W es O(M_in * M_out). Para b es O(M_out).
    void save_tensor(std::ofstream& ofs, const utec::algebra::Tensor<T, 2>& t) const {
        // ... (Implementación de guardar dimensiones y datos binarios) ...
        size_t rows = t.shape()[0];
        size_t cols = t.shape()[1];
        size_t num_elements = rows * cols;

        ofs.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));

        if (num_elements > 0) {
            ofs.write(reinterpret_cast<const char*>(t.data()), num_elements * sizeof(T));
        }
    }

    // Carga un tensor 2D desde un archivo binario
    // Complejidad: O(N_elements). Para W es O(M_in * M_out). Para b es O(M_out).
    void load_tensor(std::ifstream& ifs, utec::algebra::Tensor<T, 2>& t) {
        // ... (Implementación de leer dimensiones, redimensionar y leer datos binarios) ...
        size_t rows, cols;
        ifs.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
        size_t num_elements = rows * cols;

        t.reshape(rows, cols);

        if (num_elements > 0) {
            ifs.read(reinterpret_cast<char*>(t.data()), num_elements * sizeof(T));
        }
    }

public:
    // Constructor
    // Complejidad: O(M_in * M_out). Domina la inicialización de W y grad_W.
    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun)
        : in_features_(in_f), out_features_(out_f),
          weights_(in_f, out_f), // O(M_in * M_out)
          biases_(1, out_f),
          grad_weights_(in_f, out_f), // O(M_in * M_out)
          grad_biases_(1, out_f) {

        init_w_fun(weights_);
        init_b_fun(biases_);

        grad_weights_.fill(T{0});
        grad_biases_.fill(T{0});
    }

    // Constructor vacío para Deserialización
    // Complejidad: O(1)
    Dense() : in_features_(0), out_features_(0), weights_(0, 0), biases_(0, 0),
              grad_weights_(0, 0), grad_biases_(0, 0) {}

    // --- Algoritmo Forward Pass ---
    // Y = X * W + b
    // Complejidad: O(S_batch * M_in * M_out) o C_mat_mul.
    // Dominado por la multiplicación matricial.
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& x) override {
        input_ = x; // Guardar entrada. O(S_batch * M_in)
        
        // Multiplicación: (S_batch x M_in) * (M_in x M_out) = (S_batch x M_out)
        auto output = utec::algebra::matrix_product(x, weights_); // O(C_mat_mul)
        
        // Suma de sesgos (Broadcasting). O(S_batch * M_out)
        output = output + biases_; 
        return output;
    }

    // --- Algoritmo Backward Pass ---
    // Calcula dW, db y dX
    // Complejidad: O(S_batch * M_in * M_out) o C_mat_mul.
    // Dominado por las dos multiplicaciones matriciales.
    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& dZ) override {
        // 1. Gradiente de Pesos (dW): X^T * dZ
        auto input_T = utec::algebra::transpose_2d(input_); // O(S_batch * M_in)
        // Multiplicación: (M_in x S_batch) * (S_batch x M_out) = (M_in x M_out)
        grad_weights_ = utec::algebra::matrix_product(input_T, dZ); // O(C_mat_mul)

        // 2. Gradiente de Sesgos (db): Suma de dZ a lo largo del batch
        grad_biases_.fill(T{0});
        // O(S_batch * M_out)
        for (size_t i = 0; i < dZ.shape()[0]; ++i) { 
            for (size_t j = 0; j < dZ.shape()[1]; ++j) { 
                grad_biases_(0, j) += dZ(i, j); 
            }
        }

        // 3. Gradiente para la capa anterior (dX): dZ * W^T
        auto weights_T = utec::algebra::transpose_2d(weights_); // O(M_in * M_out)
        // Multiplicación: (S_batch x M_out) * (M_out x M_in) = (S_batch x M_in)
        auto dX = utec::algebra::matrix_product(dZ, weights_T); // O(C_mat_mul)

        return dX;
    }

    // --- Algoritmo de Actualización de Parámetros ---
    // Complejidad: O(P_layer).
    // Lineal con el número de parámetros de la capa.
    void update_params(IOptimizer<T>& optimizer) override {
        // update(W, dW): O(M_in * M_out)
        optimizer.update(weights_, grad_weights_);
        // update(b, db): O(M_out)
        optimizer.update(biases_, grad_biases_);
    }

    // -----------------------------------------------------------------
    //  MÉTODOS DE SERIALIZACIÓN
    // -----------------------------------------------------------------

    /**
     * @brief Guarda la matriz de pesos (W) y el vector de sesgos (b).
     * Complejidad: O(P_layer).
     */
    void save_parameters(std::ofstream& ofs) const {
        save_tensor(ofs, weights_);
        save_tensor(ofs, biases_);
    }

    /**
     * @brief Carga la matriz de pesos (W) y el vector de sesgos (b).
     * Complejidad: O(P_layer).
     */
    void load_parameters(std::ifstream& ifs) {
        load_tensor(ifs, weights_);
        load_tensor(ifs, biases_);

        // Post-carga: O(1) y O(M_in * M_out) para redimensionar gradientes.
        in_features_ = weights_.shape()[0];
        out_features_ = weights_.shape()[1];

        grad_weights_.reshape(in_features_, out_features_);
        grad_weights_.fill(T{0});
        grad_biases_.reshape(1, out_features_);
        grad_biases_.fill(T{0});
    }
};

} // namespace neural_network
} // namespace utec

#endif // NN_DENSE_H
