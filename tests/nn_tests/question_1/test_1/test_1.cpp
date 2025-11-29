//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
#include "nn_activation.h"

template<typename T, size_t DIMS>
using Tensor = utec::algebra::Tensor<T, DIMS>;

static void test_1() {
    using T = float;
    auto relu = utec::neural_network::ReLU<T>();
    // Tensores
    Tensor<T, 2> M(2,2); M = {-1, 2, 0, -3};
    Tensor<T, 2> GR(2,2); GR.fill(1.0f);
    // Forward
    auto R = relu.forward(M);
    std::cout << R(0,1) << "\n"; // espera 2
    // Backward
    const auto dM = relu.backward(GR);
    std::cout << dM;
}

TEST_CASE("Question #1.1") {
    execute_test("question_1_test_1.in", test_1);
}