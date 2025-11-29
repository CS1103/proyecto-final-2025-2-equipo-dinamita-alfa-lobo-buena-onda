//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "nn_loss.h"

template<typename T, size_t DIMS>
using Tensor = utec::algebra::Tensor<T, DIMS>;

static void test_1() {
    using T = double;
    // Tensores
    Tensor<T,2> y_predicted(1,2); y_predicted = {1, 2};
    Tensor<T,2> y_expected(1,2); y_expected = {0, 4};

    const utec::neural_network::MSELoss<T> mse_loss(y_predicted, y_expected);
    // Forward
    const T loss = mse_loss.loss();
    std::cout << loss << "\n";                 // espera 2.5
    // Backward
    const Tensor<T,2> dP = mse_loss.loss_gradient();
    std::cout << dP;
}

TEST_CASE("Question #2.1") {
    execute_test("question_2_test_1.in", test_1);
}