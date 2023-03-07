#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <cstdint>
#include <string>
#include <vector>
#include <array>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "LTPG/Utils/Convert.hpp"
#include "LTPG/Utils/DataTypes.hpp"

namespace libtorchPG
{
    class ImageProcessor
    {

    private:
        cv::Mat origImg;
        std::vector<cv::Scalar> colorList;

        const uint16_t imgSize;
        const std::vector<double> mean = {0.485, 0.456, 0.406};
        const std::vector<double> std = {0.229, 0.224, 0.225};

        cv::Mat resizeCenterCut();

        cv::Mat resizeStretch();

        cv::Mat resizeLetterBox();

        void generateColor();

    public:
        ImageProcessor(const std::string &path, const uint32_t imgSize);

        ~ImageProcessor(){};

        cv::Mat getImage()
        {
            return this->origImg;
        }

        cv::Size getSize()
        {
            return this->origImg.size();
        }

        torch::Tensor process(CropType type, bool normalize);

        cv::Mat drawText(const std::string &label, const double prob);

        cv::Mat drawBbox(std::vector<DetResult> &result, std::vector<std::string> &labelList);

        void drawFPS();
    };

}

#endif