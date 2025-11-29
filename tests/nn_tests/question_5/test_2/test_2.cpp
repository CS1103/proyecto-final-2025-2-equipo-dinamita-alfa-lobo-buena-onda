//
// Created by rudri on 9/12/2020.
//
#include <ranges>
#include <iomanip>
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
#include "nn_optimizer.h"
using namespace std;

static void test_2() {
    using namespace utec::neural_network;
    using T = float;

    Tensor<T,2> W(2,2); W.fill(1.0f);
    Tensor<T,2> dW(2,2); dW.fill(0.5f);
    utec::neural_network::Adam opt(0.1f);

    opt.update(W, dW);
    std::cout
        << std::fixed << std::setprecision(6)
        << W(0,0) << "\n";
}

TEST_CASE("Question #5.2") {
    execute_test("question_5_test_2.in", test_2);
}