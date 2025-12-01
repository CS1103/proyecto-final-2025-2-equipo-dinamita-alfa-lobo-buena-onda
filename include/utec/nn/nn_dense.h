#ifndef NN_DENSE_H
#define NN_DENSE_H

#include <utec/nn/nn_interfaces.h>
#include <utec/algebra/Tensor.h>
#include <fstream> // Necesario para la serialización

namespace utec {
namespace neural_network {

// Capa Dense (Fully Connected): Y = X * W + b
template<typename T>
class Dense final : public ILayer<T> {
private:
    utec::algebra::Tensor<T, 2> weights_;      // Matriz de pesos (in_features x out_features)
    utec::algebra::Tensor<T, 2> biases_;       // Vector de sesgos (1 x out_features)

    utec::algebra::Tensor<T, 2> input_;        // Guardamos entrada para backward
    utec::algebra::Tensor<T, 2> grad_weights_; // Gradiente de pesos
    utec::algebra::Tensor<T, 2> grad_biases_;  // Gradiente de sesgos

    size_t in_features_;
    size_t out_features_;

    // --- FUNCIONES AUXILIARES DE SERIALIZACIÓN ---

    // Guarda las dimensiones y el contenido binario de un tensor 2D
    void save_tensor(std::ofstream& ofs, const utec::algebra::Tensor<T, 2>& t) const {
        // 1. Guardar dimensiones (filas y columnas)
        size_t rows = t.shape()[0];
        size_t cols = t.shape()[1];
        size_t num_elements = rows * cols;

        ofs.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));

        // 2. Guardar los datos binarios del tensor
        if (num_elements > 0) {
            ofs.write(reinterpret_cast<const char*>(t.data()), num_elements * sizeof(T));
        }
    }

    // Carga un tensor 2D desde un archivo binario
    void load_tensor(std::ifstream& ifs, utec::algebra::Tensor<T, 2>& t) {
        // 1. Leer dimensiones (filas y columnas)
        size_t rows, cols;
        ifs.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
        size_t num_elements = rows * cols;

        // 2. Redimensionar el tensor (Importante: Asume que el Tensor tiene método resize)
        t.reshape(rows, cols);

        // 3. Leer los datos binarios
        if (num_elements > 0) {
            ifs.read(reinterpret_cast<char*>(t.data()), num_elements * sizeof(T));
        }
    }

public:
    // Constructor con inicializadores personalizados (se mantiene igual)
    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun)
        : in_features_(in_f), out_features_(out_f),
          weights_(in_f, out_f),
          biases_(1, out_f),
          grad_weights_(in_f, out_f),
          grad_biases_(1, out_f) {

        init_w_fun(weights_);
        init_b_fun(biases_);

        grad_weights_.fill(T{0});
        grad_biases_.fill(T{0});
    }

    // Constructor vacío para Deserialización (OPCIONAL, pero útil para cargar)
    // Se asume que este constructor será usado y luego se llamará a load_parameters
    Dense() : in_features_(0), out_features_(0), weights_(0, 0), biases_(0, 0),
              grad_weights_(0, 0), grad_biases_(0, 0) {}

    // Forward: Y = X * W + b (se mantiene igual)
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& x) override {
        input_ = x;
        auto output = utec::algebra::matrix_product(x, weights_);
        output = output + biases_;
        return output;
    }

    // Backward (se mantiene igual)
    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& dZ) override {
        // ... (Tu implementación existente de backward)
        auto input_T = utec::algebra::transpose_2d(input_);
        grad_weights_ = utec::algebra::matrix_product(input_T, dZ);

        // Sumar los gradientes a lo largo del batch (dim 0) para obtener el gradiente del sesgo
        grad_biases_.fill(T{0});
        for (size_t i = 0; i < dZ.shape()[0]; ++i) { // Para cada muestra en el batch
            for (size_t j = 0; j < dZ.shape()[1]; ++j) { // Para cada neurona de salida
                grad_biases_(0, j) += dZ(i, j); // Acumular el gradiente
            }
        }

        auto weights_T = utec::algebra::transpose_2d(weights_);
        auto dX = utec::algebra::matrix_product(dZ, weights_T);

        return dX;
    }

    // Actualiza pesos y sesgos usando el optimizador (se mantiene igual)
    void update_params(IOptimizer<T>& optimizer) override {
        optimizer.update(weights_, grad_weights_);
        optimizer.update(biases_, grad_biases_);
    }

    // -----------------------------------------------------------------
    //  MÉTODOS DE SERIALIZACIÓN (Portabilidad - Requisito Epic 3)
    // -----------------------------------------------------------------

    /**
     * @brief Guarda la matriz de pesos y el vector de sesgos en el stream.
     */
    void save_parameters(std::ofstream& ofs) const {
        save_tensor(ofs, weights_);
        save_tensor(ofs, biases_);
    }

    /**
     * @brief Carga la matriz de pesos y el vector de sesgos desde el stream.
     */
    void load_parameters(std::ifstream& ifs) {
        load_tensor(ifs, weights_);
        load_tensor(ifs, biases_);

        // Actualizar features después de cargar
        in_features_ = weights_.shape()[0];
        out_features_ = weights_.shape()[1];

        // Redimensionar gradientes para asegurar consistencia (o solo inicializarlos a cero)
        grad_weights_.reshape(in_features_, out_features_);
        grad_weights_.fill(T{0});
        grad_biases_.reshape(1, out_features_);
        grad_biases_.fill(T{0});
    }
};

} // namespace neural_network
} // namespace utec

#endif // NN_DENSE_H