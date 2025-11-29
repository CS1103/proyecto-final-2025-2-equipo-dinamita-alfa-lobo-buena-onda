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

static void test_3() {
    using namespace utec::neural_network;
    using T = float;

    Tensor<T,2> W(20,25); W.fill(1.0f);
    Tensor<T,2> dW(20,25); dW.fill(0.2f);
    utec::neural_network::Adam opt(0.01f, 0.009f, 9.00f);

    opt.update(W, dW);
    std::cout << W(0,0) << "\n";
}

TEST_CASE("Question #5.3") {
    execute_test("question_5_test_3.in", test_3);
}