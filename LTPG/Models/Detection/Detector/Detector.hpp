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
    /**
     * @brief Object Detection Model Class
     * 
     */
    class Detector
    {

    public:
        /**
         * @brief Construct a new Detector object
         * 
         * @param modelFile     : Model Weight Path
         * @param modelName     : Model Name Type
         * @param inputShape    : Model Input Shape
         * @param origShape     : Image Original Shape
         */
        Detector(const std::string &modelFile, const ModelName modelName, const cv::Size &inputShape,
                 const cv::Size &origShape);

        /**
         * @brief Destroy the Detector object
         * 
         */
        ~Detector();

        /**
         * @brief Run Detection Inference on Tensor Input of Pre-processed Image
         * 
         * @param input     : Preprocess Tensor Input
         * @return std::vector<DetResult> 
         */
        std::vector<DetResult> runInference(torch::Tensor input);

    private:
        Logger &logger = Logger::getInstance(LogType::InfoLog);

        AbstractPostProcessor *postProcessor = nullptr;
        torch::jit::script::Module model;
        torch::Tensor output;
    };
}

#endif