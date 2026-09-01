#include "neg-head.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: neg-head-check <neg-head.bin>\n";
        return 2;
    }

    darwin_neg_head head;
    std::string error;
    if (!head.load(argv[1], 4096, error)) {
        std::cerr << error << '\n';
        return 1;
    }

    std::vector<float> hidden(4096);
    for (size_t index = 0; index < hidden.size(); ++index) {
        hidden[index] = std::sin(static_cast<float>(index) * 0.01f);
    }
    const auto observation = head.evaluate(hidden.data());
    std::cout << std::setprecision(10) << observation.predicted_entropy << '\n';
    return 0;
}
