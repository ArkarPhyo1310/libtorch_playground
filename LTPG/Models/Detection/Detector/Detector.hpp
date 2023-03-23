#ifndef DETECTOR_H
#define DETECTOR_H

#include <string>
#include <vector>
#include <chrono>

#include <torch/torch.h>
#include <torch/script.h>

#include "LTPG/Models/Detection/PostProcessor/Abstract/Abstract.hpp"
#include "LTPG/Models/Detection/PostProcessor/YoloV5/PostProcess.hpp"
#include "LTPG/Utils/Logger.hpp"
#include "LTPG/Utils/Convert.hpp"

namespace libtorchPG
{
    class Detector
    {

    public:
        Detector(const std::string &modelFile, const ModelName modelName, const cv::Size &inputShape,
                 const cv::Size &origShape);

        ~Detector();

        std::vector<DetResult> runInference(torch::Tensor input);

    private:
        Logger &logger = Logger::getInstance(LogType::InfoLog);

        AbstractPostProcessor *postProcessor = nullptr;
        torch::jit::script::Module model;
        torch::Tensor output;
    };
}

#endif