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

static void test_4() {
    using namespace utec::neural_network;
    using T = double;

    Tensor<T,2> W(20,25); W.fill(1.0);
    Tensor<T,2> dW(20,25); dW.fill(0.25);
    utec::neural_network::SGD opt(0.03123);

    opt.update(W, dW);
    std::cout << W(0,0) << "\n";
}

TEST_CASE("Question #5.4") {
    execute_test("question_5_test_4.in", test_4);
}