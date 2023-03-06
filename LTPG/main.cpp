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

using namespace libtorchPG;
namespace fs = std::filesystem;

argparse::ArgumentParser getOpts(int argc, char *argv[])
{
    argparse::ArgumentParser program("LibTorch Playground");

    fs::path exePath = getExePath(argv[0]);
    fs::path imagePath = getDefault(exePath, "D:\\Personal_Projects\\yolov5\\data\\images\\zidane.jpg");
    fs::path modelPath = getDefault(exePath, "resnet.torchscript");
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
    
    // Classification Part
    // std::vector<std::string> labelList = loadLabels(label);
    // ImageProcessor imageProcessor(path, 224);
    // Classifier clsPredictor(model);
    // torch::Tensor imgTensor = imageProcessor.process(CropType::Stretch, true);
    // clsPredictor.runInference(imgTensor);
    // ClsResult output = clsPredictor.getOutput();
    // imageProcessor.drawText(labelList[output.idx], output.prob);
    // cv::Mat res_image = imageProcessor.getImage();
    // std::cout << "Label: " << labelList[output.idx] << std::endl;
    // std::cout << "Probability: " << output.prob << std::endl;

    // Detection Part
    ImageProcessor imageDetProcessor(path, 640);
    torch::Tensor imgDetTensor = imageDetProcessor.process(CropType::LetterBox, false);
    Detector detPredictor(
        "D:\\Personal_Projects\\yolov5\\yolov5s.torchscript",
        ModelName::YOLOV5,
        cv::Size(640, 640),
        imageDetProcessor.getSize());

    std::vector<DetResult> res = detPredictor.runInference(imgDetTensor);
    std::vector<std::string> detLabelList = {};
    for (auto &i : res)
    {
        cv::Rect rect(cv::Point(i.x1, i.y1), cv::Point(i.x2, i.y2));
        std::cout << "Result: " << std::endl;
        std::cout << "\tx1: " << i.x1 << std::endl;
        std::cout << "\ty1: " << i.y1 << std::endl;
        std::cout << "\tx2: " << i.x2 << std::endl;
        std::cout << "\ty2: " << i.y2 << std::endl;
        std::cout << "\tscore: " << i.score << std::endl;
        std::cout << "\tidx: " << i.idx << std::endl;

        imageDetProcessor.drawBbox(rect, i.idx, i.score, detLabelList);
    }
    cv::Mat res_image = imageDetProcessor.getImage();

    if (save)
    {
        fs::create_directories(savePath);
        fs::path saveImgPath = fs::path(savePath) / "output.png";
        cv::imwrite(saveImgPath.string(), res_image);
        std::cout << "Saving Image @ " << saveImgPath.string();
    }

    cv::imshow("Demo", res_image);
    if (cv::waitKey(0) > 0)
    {
        cv::destroyAllWindows();
    }

    return 0;
}