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
#include "LTPG/Utils/Logger.hpp"

namespace libtorchPG
{   
    /**
     * @brief Image Pre Processing Class
     * 
     */
    class ImageProcessor
    {

    private:
        cv::Mat origImg;
        std::vector<cv::Scalar> colorList;
        Logger &logger = Logger::getInstance(LogType::InfoLog);

        const uint16_t imgSize;

        // ImageNet Normalization
        const std::vector<double> mean = {0.485, 0.456, 0.406};
        const std::vector<double> std = {0.229, 0.224, 0.225};

        /**
         * @brief Resize the image with top, left, bottom, right Padding
         * 
         * @return cv::Mat 
         */
        cv::Mat resizeCenterCut();

        /**
         * @brief Stretch the image (cv2.resize)
         * 
         * @return cv::Mat 
         */
        cv::Mat resizeStretch();

        /**
         * @brief LetterBox Image Size Method (Mostly used in YOLO)
         * 
         * @return cv::Mat 
         */
        cv::Mat resizeLetterBox();

        /**
         * @brief Bounding Box Color Generator
         * 
         */
        void generateColor();

    public:
        /**
         * @brief Construct a new Image Processor object
         * 
         * @param path 
         * @param imgSize 
         */
        ImageProcessor(const std::string &path, const uint32_t imgSize);

        /**
         * @brief Destroy the Image Processor object
         * 
         */
        ~ImageProcessor(){};

        /**
         * @brief Get the Image object
         * 
         * @return cv::Mat 
         */
        cv::Mat getImage()
        {
            return this->origImg;
        }

        /**
         * @brief Get the Size object
         * 
         * @return cv::Size 
         */
        cv::Size getSize()
        {
            return this->origImg.size();
        }

        /**
         * @brief Image Pre-processing Function 
         * 
         * @param type      : Image Resize Type
         * @param normalize : ImageNet Normalization
         * @return torch::Tensor 
         */
        torch::Tensor process(CropType type, bool normalize);

        /**
         * @brief Draw Label Class and Probability in image for Classifcation 
         * 
         * @param label 
         * @param prob 
         * @return cv::Mat 
         */
        cv::Mat drawText(const std::string &label, const double prob);

        /**
         * @brief Draw Bounding Box with Label class and probability in image for Detection
         * 
         * @param result 
         * @param labelList 
         * @return cv::Mat 
         */
        cv::Mat drawBbox(std::vector<DetResult> &result, std::vector<std::string> &labelList);
        
        // TODO
        void drawFPS();
    };

}

#endif