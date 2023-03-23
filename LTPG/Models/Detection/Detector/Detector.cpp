#include <opencv2/opencv.hpp>

#include "LTPG/Models/Detection/Detector/Detector.hpp"

using namespace libtorchPG;

Detector::Detector(
    const std::string &modelFile,
    const ModelName modelName,
    const cv::Size &inputShape,
    const cv::Size &origShape)
{
    try
    {
        this->model = torch::jit::load(modelFile);
        this->model.eval();
    }
    catch (const c10::Error &e)
    {
        std::cerr << e.what() << '\n';
        abort();
    }

    logger.logInfo("Finish loading the model...");
    torch::jit::setGraphExecutorOptimize(false);
    logger.logInfo("Warming up the model...");
    this->model.forward({torch::randn({1, 3, 640, 640})});

    switch (modelName)
    {
    case ModelName::YOLOV5:
        postProcessor = new YOLOV5PostProcessor(origShape, inputShape, 0.5, 0.5);
        break;
    default:
        break;
    }
}

Detector::~Detector()
{
    delete this->postProcessor;
}

std::vector<DetResult> Detector::runInference(torch::Tensor inputTensor)
{
    logger.logInfo("Starting the inference process...");

    std::vector<torch::jit::IValue> input;
    input.push_back(inputTensor);

    auto startTime = std::chrono::high_resolution_clock::now();
    auto output = this->model.forward(input).toTuple()->elements()[0].toTensor();
    auto endTime = std::chrono::high_resolution_clock::now();

    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    logger.logInfo("Inference Time(ms): " + std::to_string(duration));

    std::vector<DetResult> res = this->postProcessor->postProcess(output);

    return res;
}