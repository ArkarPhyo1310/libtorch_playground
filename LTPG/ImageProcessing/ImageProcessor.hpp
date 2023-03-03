#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <cstdint>
#include <string>
#include <vector>
#include <array>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace libtorchPG {

enum class CropType {
    CenterCut,
    Stretch,
    LetterBox,
};

class ImageProcessor {

private: 

    cv::Mat origImg;
    const uint16_t imgSize;
    const std::vector<double> mean = {0.485, 0.456, 0.406};
    const std::vector<double> std = {0.229, 0.224, 0.225};
    
    cv::Mat resizeCenterCut();

    cv::Mat resizeStretch();

    cv::Mat resizeLetterBox();

public:

    ImageProcessor(const std::string &path, const uint32_t imgSize);

    ~ImageProcessor() {};

    cv::Mat getImage() {
        return this->origImg;
    }

    torch::Tensor process(CropType type, bool normalize);

    void drawText(const std::string& label, const double prob);
};

}

#endif