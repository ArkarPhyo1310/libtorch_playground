#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <vector>

#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <argparse/argparse.hpp>

#include "ImageProcessing/ImageProcessor.hpp"
#include "Models/Classification/Classifier.hpp"
#include "Models/Detection/Detector/Detector.hpp"
#include "Utils/Convert.hpp"
#include "Utils/File.hpp"
#include "Utils/Logger.hpp"

using namespace libtorchPG;
namespace fs = std::filesystem;

argparse::ArgumentParser getOpts(int argc, char *argv[])
{
    argparse::ArgumentParser program("LibTorch Playground");

    fs::path exePath = getExePath(argv[0]);
    fs::path imagePath = getDefault(exePath, "test.jpg");
    fs::path modelPath = getDefault(exePath, "yolov5s.torchscript");
    fs::path labelPath = getDefault(exePath, "coco.txt");
    std::string defaultType = "Detection";

    program.add_argument("-i", "--image")
        .default_value(imagePath.string())
        .required()
        .help("Specify the image path.");

    program.add_argument("-t", "--type")
        .default_value(defaultType)
        .required()
        .help("Specify the Vision Type ['Classification', 'Detection'].");

    program.add_argument("-m", "--model")
        .default_value(modelPath.string())
        .required()
        .help("Specify the model path.");

    program.add_argument("-l", "--label")
        .default_value(labelPath.string())
        .required()
        .help("Specify the label file.");

    program.add_argument("-s", "--save-folder")
        .default_value(std::string("-"))
        .required()
        .help("Specify the save path.");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }
    return program;
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser argparser = getOpts(argc, argv);

    std::string path = argparser.get<std::string>("--image");
    std::string model = argparser.get<std::string>("--model");
    std::string label = argparser.get<std::string>("--label");
    std::string savePath = argparser.get<std::string>("--save-folder");
    std::string type = argparser.get<std::string>("--type");

    bool save = argparser.is_used("--save-folder");

    Logger &logger = Logger::getInstance(LogType::InfoLog);

    std::vector<std::string> labelList = loadLabels(label);
    cv::Mat res_image;

    if (type == "Classification")
    {
        // Classification Part
        ImageProcessor imageProcessor(path, 224);
        Classifier clsPredictor(model);
        torch::Tensor imgTensor = imageProcessor.process(CropType::Stretch, true);
        clsPredictor.runInference(imgTensor);
        ClsResult output = clsPredictor.getOutput();
        imageProcessor.drawText(labelList[output.idx], output.prob);
        res_image = imageProcessor.getImage();
        logger.logInfo("Label: " + labelList[output.idx]);
        logger.logInfo("Probability: " + std::to_string(output.prob));
    }
    else if (type == "Detection")
    {
        // Detection Part
        ImageProcessor imageDetProcessor(path, 640);
        torch::Tensor imgDetTensor = imageDetProcessor.process(CropType::LetterBox, false);
        Detector detPredictor(
            model,
            ModelName::YOLOV5,
            cv::Size(640, 640),
            imageDetProcessor.getSize());
        std::vector<DetResult> res = detPredictor.runInference(imgDetTensor);
        res_image = imageDetProcessor.drawBbox(res, labelList);
    }
    else
    {
        logger.logError("Unknown Model Type['Classification', 'Detection']: " + type);
        abort();
    }

    if (save)
    {
        fs::create_directories(savePath);
        fs::path saveImgPath = fs::path(savePath) / "output.png";
        cv::imwrite(saveImgPath.string(), res_image);
        logger.logInfo("Saving Image @ " + saveImgPath.string());
    }

    cv::imshow("LibTorch Playground DEMO", res_image);
    cv::waitKey(0);

    return 0;
}