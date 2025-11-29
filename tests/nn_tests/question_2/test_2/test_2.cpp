//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "nn_loss.h"

template<typename T, size_t DIMS>
using Tensor = utec::algebra::Tensor<T, DIMS>;

static void test_2() {
    using T = double;
    // Tensores
    Tensor<T,2> y_predicted(1,2); y_predicted = {0.9, 0.1};
    Tensor<T,2> y_expected(1,2); y_expected = {0, 1};

    const utec::neural_network::BCELoss<T> bce_loss(y_predicted, y_expected);
    // Forward
    const T loss = bce_loss.loss();
    std::cout << loss << "\n";
    // Backward
    const Tensor<T,2> dP = bce_loss.loss_gradient();
    std::cout << dP;
}

TEST_CASE("Question #2.2") {
    execute_test("question_2_test_2.in", test_2);
}