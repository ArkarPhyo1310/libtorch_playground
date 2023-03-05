#ifndef YOLOV5_POSTPROCESS_H
#define YOLOV5_POSTPROCESS_H

#include <algorithm>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "LTPG/Models/Detection/PostProcessor/Abstract/Abstract.hpp"
#include "LTPG/Models/Detection/Utils/Utils.hpp"

namespace libtorchPG
{
    class YOLOV5PostProcessor : public AbstractPostProcessor
    {
    private:
        float scale;
        std::vector<int> pad;
        const cv::Size &oldShape;

        std::vector<DetResult> NMS(torch::Tensor preds);

        void ScaleCoords(std::vector<DetResult> &dets);

    public:
        YOLOV5PostProcessor(const cv::Size &oldShape, const cv::Size &newShape, float scoreThresh = 0.5, float iouThresh = 0.5)
            : AbstractPostProcessor(scoreThresh, iouThresh),
              oldShape(oldShape)
        {
            this->scale = std::min(toFloat(newShape.height) / toFloat(oldShape.height),
                                   toFloat(newShape.width) / toFloat(oldShape.width));

            int w = static_cast<int>((newShape.width - oldShape.width * this->scale) / 2);
            int h = static_cast<int>((newShape.height - oldShape.height * this->scale) / 2);

            this->pad = {w, h};
            std::cout << scale << " " << w << " " << h << std::endl;
        };

        ~YOLOV5PostProcessor(){};

        std::vector<DetResult> postProcess(torch::Tensor preds);
    };
} // namespace libtorchPG

#endif
