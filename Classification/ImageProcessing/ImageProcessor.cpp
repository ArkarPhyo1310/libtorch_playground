#include <iostream>

#include "Classification/ImageProcessing/ImageProcessor.hpp"
using namespace classification;

ImageProcessor::ImageProcessor(const std::string &path, const uint32_t imgSize)
    : imgSize(imgSize)
{
    this->origImg = cv::imread(path);
}

cv::Mat ImageProcessor::resizeCenter()
{
    const uint32_t rows = this->origImg.rows;
    const uint32_t cols = this->origImg.cols;

    const uint32_t cropSize = std::min(rows, cols);
    const uint32_t offsetW = (cols - cropSize) / 2;
    const uint32_t offsetH = (rows - cropSize) / 2;
    const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);

    cv::Mat dstImg = this->origImg(roi);
    
    cv::resize(this->origImg, dstImg, cv::Size(this->imgSize, this->imgSize));

    return dstImg;
}

cv::Mat ImageProcessor::resizeStretch()
{
    cv::Mat dstImg = this->origImg.clone();
    cv::resize(this->origImg, dstImg, cv::Size(this->imgSize, this->imgSize), cv::INTER_LINEAR);
    return dstImg;
}

torch::Tensor ImageProcessor::process(CropType cType)
{
    cv::Mat dstImg = (cType == CropType::Center) ? this->resizeCenter() : this->resizeStretch();
    cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2RGB);
    dstImg.convertTo(dstImg, CV_32FC3, 1 / 255.0);
    torch::Tensor tensorImg = torch::from_blob(
        dstImg.data,
        {1, dstImg.rows, dstImg.cols, 3},
        c10::kFloat
    );
    tensorImg = tensorImg.permute({0, 3, 1, 2});
    tensorImg = torch::data::transforms::Normalize<>(mean, std)(tensorImg);
    return tensorImg;

}

void ImageProcessor::drawText(const std::string& label, const double prob) 
{
    int baseline = 0;
    char text[256];
    sprintf(text, "%s: %.1f%%", label.c_str(), prob * 100);
    
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    cv::rectangle(
        this->origImg,
        cv::Rect(
            cv::Point(0, 0),
            cv::Size(textSize.width, textSize.height + baseline)
        ),
        cv::Scalar(0, 0, 0),
        -1
    );
    cv::putText(
        this->origImg,
        text,
        cv::Point(0, textSize.height),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, 
        cv::Scalar(255, 255, 255), 1
    );
}