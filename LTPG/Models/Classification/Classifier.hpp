#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <vector>
#include <chrono>

#include <torch/torch.h>
#include <torch/script.h>

#include "LTPG/Utils/DataTypes.hpp"

namespace libtorchPG
{
    class Classifier
    {

    public:
        Classifier(const std::string &modelFile);

        ~Classifier();

        void runInference(torch::Tensor input);

        ClsResult getOutput();

    private:
        torch::jit::script::Module model;
        torch::Tensor output;
    };

}

#endif