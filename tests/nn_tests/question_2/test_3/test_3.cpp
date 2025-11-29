//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "nn_loss.h"

template<typename T, size_t DIMS>
using Tensor = utec::algebra::Tensor<T, DIMS>;

static void test_3() {
    using T = double;
    // Tensores
    constexpr int rows = 10;
    constexpr int cols = 12;
    Tensor<T,2> y_predicted(rows,cols);
    Tensor<T,2> y_expected(rows,cols);

    y_predicted.fill(0.9);
    y_expected.fill(0);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            if (i == j) {
                y_predicted(i, j) = 0.1;
                y_expected(i, j) = 1;
            }
            if (i == rows - 1 - j) {
                y_predicted(i, j) = 0.1;
                y_expected(i, j) = 1;
            }
        }

    const utec::neural_network::BCELoss<T> bce_loss(y_predicted, y_expected);

    std::cout << std::fixed << std::setprecision(3);
    // Forward
    const T loss = bce_loss.loss();
    std::cout << loss << "\n";
    // Backward
    const Tensor<T,2> dP = bce_loss.loss_gradient();
    std::cout << dP;
}

TEST_CASE("Question #2.3") {
    execute_test("question_2_test_3.in", test_3);
}