#ifndef PREDICTOR_H
#define PREDICTOR_H

#include <string>
#include <vector>
#include <chrono>

#include <torch/torch.h>
#include <torch/script.h>

namespace classification {

struct Result {
    int idx;
    float prob;
};

class Predictor {

public:
    Predictor(const std::string& modelFile);

    ~Predictor();

    void runInference(torch::Tensor input);

    Result getOutput();

private:
    torch::jit::script::Module model;
    torch::Tensor output;
};

}

#endif