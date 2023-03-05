#include <algorithm>
#include <iostream>

#include "LTPG/ImageProcessing/ImageProcessor.hpp"

using namespace libtorchPG;

ImageProcessor::ImageProcessor(const std::string &path, const uint32_t imgSize)
    : imgSize(imgSize)
{
    this->origImg = cv::imread(path);
}

cv::Mat ImageProcessor::resizeLetterBox()
{
    cv::Mat dstImg;

    std::vector<float> oldShape{static_cast<float>(this->origImg.rows), static_cast<float>(this->origImg.cols)};
    std::vector<float> newShape{static_cast<float>(this->imgSize), static_cast<float>(this->imgSize)};

    float scale = std::min(newShape[0] / oldShape[0], newShape[1] / oldShape[1]);

    int unpadW = static_cast<int>(oldShape[1] * scale);
    int unpadH = static_cast<int>(oldShape[0] * scale);

    int dw = newShape[1] - unpadW;
    int dh = newShape[0] - unpadH;

    // dw = (dw % 32) / 2;
    // dh = (dh % 32) / 2;

    dw /= 2;
    dh /= 2;

    cv::resize(this->origImg, dstImg, cv::Size(unpadW, unpadH), 0, 0, cv::INTER_LINEAR);

    int top = static_cast<int>(dh);
    int bottom = static_cast<int>(dh);
    int left = static_cast<int>(dw);
    int right = static_cast<int>(dw);

    cv::copyMakeBorder(dstImg, dstImg, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return dstImg;
}

cv::Mat ImageProcessor::resizeStretch()
{
    cv::Mat dstImg = this->origImg.clone();
    cv::resize(this->origImg, dstImg, cv::Size(this->imgSize, this->imgSize), cv::INTER_LINEAR);
    return dstImg;
}

cv::Mat ImageProcessor::resizeCenterCut()
{
    cv::Mat dstImg = cv::Mat::zeros(this->imgSize, this->imgSize, CV_8UC3);

    float aspect_ratio_src = static_cast<float>(this->origImg.cols) / this->origImg.rows;
    float aspect_ratio_dst = static_cast<float>(dstImg.cols) / dstImg.rows;
    cv::Rect target_rect(0, 0, this->origImg.cols, this->origImg.rows);
    if (aspect_ratio_src > aspect_ratio_dst)
    {
        target_rect.width = static_cast<int32_t>(this->origImg.rows * aspect_ratio_dst);
        target_rect.x = (this->origImg.cols - target_rect.width) / 2;
    }
    else
    {
        target_rect.height = static_cast<int32_t>(this->origImg.cols / aspect_ratio_dst);
        target_rect.y = (this->origImg.rows - target_rect.height) / 2;
    }
    cv::Mat target = this->origImg(target_rect);
    cv::resize(target, dstImg, dstImg.size(), 0, 0, cv::INTER_LINEAR);

    return dstImg;
}

torch::Tensor ImageProcessor::process(CropType cType, bool normalize)
{
    cv::Mat dstImg;

    switch (cType)
    {
    case CropType::CenterCut:
        dstImg = this->resizeCenterCut();
        break;
    case CropType::Stretch:
        dstImg = this->resizeStretch();
        break;
    case CropType::LetterBox:
        dstImg = this->resizeLetterBox();
        break;
    };

    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2RGB);
    dstImg.convertTo(dstImg, CV_32FC3, 1 / 255.0);
    torch::Tensor tensorImg = torch::from_blob(
        dstImg.data,
        {1, dstImg.rows, dstImg.cols, 3},
        c10::kFloat);
    tensorImg = tensorImg.permute({0, 3, 1, 2});
    if (normalize)
        tensorImg = torch::data::transforms::Normalize<>(mean, std)(tensorImg);
    return tensorImg.contiguous();
}

void ImageProcessor::drawText(const std::string &label, const double prob)
{
    int baseline = 0;
    char text[256];
    sprintf(text, "%s: %.1f%%", label.c_str(), prob * 100);

    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    cv::rectangle(
        this->origImg,
        cv::Rect(
            cv::Point(0, 0),
            cv::Size(textSize.width, textSize.height + baseline)),
        cv::Scalar(0, 0, 0),
        -1);
    cv::putText(
        this->origImg,
        text,
        cv::Point(0, textSize.height),
        cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(255, 255, 255), 1);
}