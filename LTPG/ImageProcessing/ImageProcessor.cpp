#include <algorithm>
#include <iostream>

#include "LTPG/ImageProcessing/ImageProcessor.hpp"

using namespace libtorchPG;

ImageProcessor::ImageProcessor(const std::string &path, const uint32_t imgSize)
    : imgSize(imgSize)
{
    this->origImg = cv::imread(path);
    this->generateColor();
}

cv::Mat ImageProcessor::resizeLetterBox()
{
    cv::Mat dstImg;

    std::vector<float> oldShape{static_cast<float>(this->origImg.rows), static_cast<float>(this->origImg.cols)};
    std::vector<float> newShape{static_cast<float>(this->imgSize), static_cast<float>(this->imgSize)};

    float scale = std::min(newShape[0] / oldShape[0], newShape[1] / oldShape[1]);

    int unpadW = toInt(oldShape[1] * scale);
    int unpadH = toInt(oldShape[0] * scale);

    int dw = newShape[1] - unpadW;
    int dh = newShape[0] - unpadH;

    dw /= 2;
    dh /= 2;

    cv::resize(this->origImg, dstImg, cv::Size(unpadW, unpadH), 0, 0, cv::INTER_LINEAR);

    int top = toInt(dh);
    int bottom = toInt(dh);
    int left = toInt(dw);
    int right = toInt(dw);

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

void ImageProcessor::generateColor()
{
    cv::RNG range(12345);
    for (int i = 0; i < 100; i++)
    {
        cv::Scalar bboxColor = cv::Scalar(range.uniform(0, 255), range.uniform(0, 255), range.uniform(0, 255));
        this->colorList.push_back(bboxColor);
    }
}

cv::Mat ImageProcessor::drawText(const std::string &label, const double prob)
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
    return this->origImg;
}

cv::Mat ImageProcessor::drawBbox(std::vector<DetResult> &result, std::vector<std::string> &labelList)
{
    for (auto &i : result)
    {
        logger.logInfo("Result: ");
        logger.logInfo("\tx1: " + std::to_string(i.x1));
        logger.logInfo("\ty1: " + std::to_string(i.y1));
        logger.logInfo("\tx2: " + std::to_string(i.x2));
        logger.logInfo("\ty2: " + std::to_string(i.y2));
        logger.logInfo("\tscore: " + std::to_string(i.score));
        logger.logInfo("\tidx: " + std::to_string(i.idx));
        cv::Rect rect(cv::Point(i.x1, i.y1), cv::Point(i.x2, i.y2));

        int baseline = 0;
        char text[256];

        std::string label = (labelList.size() > 0) ? labelList[i.idx] : std::to_string(i.idx);
        sprintf(text, "%s: %.1f%%", label.c_str(), i.score * 100);

        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        cv::Scalar bboxColor = this->colorList[i.idx];
        cv::Scalar textColor = (bboxColor[0] + bboxColor[1] + bboxColor[2]) / 3 > 128 ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        cv::rectangle(this->origImg,
                      rect, bboxColor, 2);
        cv::rectangle(this->origImg,
                      cv::Rect(
                          rect.tl(),
                          cv::Size(textSize.width + baseline, textSize.height + baseline)),
                      bboxColor,
                      -1);
        cv::putText(this->origImg,
                    text,
                    cv::Point(rect.tl().x + 2, rect.tl().y + textSize.height),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5, textColor);
    }
    return this->origImg;
}