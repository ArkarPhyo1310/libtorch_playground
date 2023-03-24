#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <vector>
#include <chrono>

#include <torch/torch.h>
#include <torch/script.h>

#include "LTPG/Utils/DataTypes.hpp"
#include "LTPG/Utils/Logger.hpp"

namespace libtorchPG
{
    /**
     * @brief Classification Model Class
     * 
     */
    class Classifier
    {

    public:
        /**
         * @brief Construct a new Classifier object
         * 
         * @param modelFile : Model Weight Path
         */
        Classifier(const std::string &modelFile);

        /**
         * @brief Destroy the Classifier object
         * 
         */
        ~Classifier();

        /**
         * @brief Run Inference on the Tensor Input of Pre-processed Image
         * 
         * @param input 
         */
        void runInference(torch::Tensor input);

        /**
         * @brief Get the Output object
         * 
         * @return ClsResult 
         */
        ClsResult getOutput();

    private:
        Logger &logger = Logger::getInstance(LogType::InfoLog);
        torch::jit::script::Module model;
        torch::Tensor output;
    };

}

#endif