//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
#include "nn_activation.h"
#include <array>

template<typename T, size_t DIMS>
using Tensor = utec::algebra::Tensor<T, DIMS>;

using T = double;

static void test_4() {
    auto sigmoid = utec::neural_network::Sigmoid<T>();
    // Tensores
    constexpr int rows = 15;
    constexpr int cols = 10;
    Tensor<T, 2> M(rows, cols);
    M.fill(-0.2);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            if (i == 0) M(i, j) = 0.5;
            if (i == rows - 1) M(i, j) = 0.7;
            if (j == 0) M(i, j) = 0.5;
            if (j == cols - 1) M(i, j) = 0.7;
        }
    std::cout << std::fixed << std::setprecision(1);
    std::cout << M << std::endl;
    // Forward
    const auto S = sigmoid.forward(M);
    std::cout << S << std::endl;
    // Backward
    Tensor<T, 2> GR(rows,cols); GR.fill(2.0);
    const auto dM = sigmoid.backward(GR);
    std::cout << dM << std::endl;
}

TEST_CASE("Question #1.4") {
    execute_test("question_1_test_4.in", test_4);
}