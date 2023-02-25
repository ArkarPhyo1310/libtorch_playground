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
#include "Utils/File.hpp"

using namespace libtorchPG;
namespace fs = std::filesystem;

argparse::ArgumentParser getOpts(int argc, char *argv[])
{
    argparse::ArgumentParser program("LibTorch Playground");

    fs::path exePath = getExePath(argv[0]);
    fs::path imagePath = getDefault(exePath, "cat.jpg");
    fs::path modelPath = getDefault(exePath, "traced_resnet_model.pt");
    fs::path labelPath = getDefault(exePath, "labels.txt");

    program.add_argument("-i", "--image")
        .default_value(imagePath.string())
        .required()
        .help("Specify the image path.");

    program.add_argument("-m", "--model")
        .default_value(modelPath.string())
        .required()
        .help("Specify the image path.");

    program.add_argument("-l", "--label")
        .default_value(labelPath.string())
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

    bool save = argparser.is_used("--save-folder");

    std::vector<std::string> labelList = loadLabels(label);
    ImageProcessor imageProcessor(path, 224);
    Classifier predictor(model);

    torch::Tensor imgTensor = imageProcessor.process(CropType::Stretch);
    predictor.runInference(imgTensor);
    Result output = predictor.getOutput();
    imageProcessor.drawText(labelList[output.idx], output.prob);
    cv::Mat image = imageProcessor.getImage();

    std::cout << "Label: " << labelList[output.idx] << std::endl;
    std::cout << "Probability: " << output.prob << std::endl;

    if (save)
    {
        fs::create_directories(savePath);
        fs::path saveImgPath = fs::path(savePath) / "output.png";
        cv::imwrite(saveImgPath.string(), image);
        std::cout << "Saving Image @ " << saveImgPath.string();
    }

    cv::imshow("Demo", image);
    if (cv::waitKey(0) > 0)
    {
        cv::destroyAllWindows();
    }

    return 0;
}