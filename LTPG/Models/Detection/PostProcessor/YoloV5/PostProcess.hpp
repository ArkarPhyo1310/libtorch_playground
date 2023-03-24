#ifndef YOLOV5_POSTPROCESS_H
#define YOLOV5_POSTPROCESS_H

#include <algorithm>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "LTPG/Models/Detection/PostProcessor/Abstract/Abstract.hpp"

namespace libtorchPG
{   
    /**
     * @brief YOLOV5 Post Processing Model Class
     * 
     */
    class YOLOV5PostProcessor : public AbstractPostProcessor
    {
    private:
        float scale;
        std::vector<int> pad;
        const cv::Size &oldShape;   

        /**
         * @brief Perform NMS on Model output
         * 
         * @param preds : Detection Model Output
         * @return std::vector<DetResult> 
         */
        std::vector<DetResult> NMS(torch::Tensor preds);

        /**
         * @brief Perform Re-Scaling after passing into NMS
         * 
         * @param dets : NMS results
         */
        void ScaleCoords(std::vector<DetResult> &dets);

    public:
        /**
         * @brief Construct a new YOLOV5PostProcessor object
         * 
         * @param oldShape      : Original Image Shape
         * @param newShape      : Preprocessed Image Shape
         * @param scoreThresh   : Confidence Threshold
         * @param iouThresh     : IOU Threshold 
         */
        YOLOV5PostProcessor(const cv::Size &oldShape, const cv::Size &newShape, float scoreThresh = 0.5, float iouThresh = 0.5)
            : AbstractPostProcessor(scoreThresh, iouThresh),
              oldShape(oldShape)
        {
            this->scale = std::min(toFloat(newShape.height) / toFloat(oldShape.height),
                                   toFloat(newShape.width) / toFloat(oldShape.width));

            int w = toInt((newShape.width - oldShape.width * this->scale) / 2);
            int h = toInt((newShape.height - oldShape.height * this->scale) / 2);

            this->pad = {w, h};
        };

        /**
         * @brief Destroy the YOLOV5PostProcessor object
         * 
         */
        ~YOLOV5PostProcessor(){};

        /**
         * @brief Perform NMS and Re-scaling
         * 
         * @param preds 
         * @return std::vector<DetResult> 
         */
        std::vector<DetResult> postProcess(torch::Tensor preds);
    };
} // namespace libtorchPG

#endif
