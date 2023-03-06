#ifndef ABSTRACT_POST_PROCESSOR_H
#define ABSTRACT_POST_PROCESSOR_H

#include <vector>

#include <torch/torch.h>
#include "LTPG/Utils/Convert.hpp"

namespace libtorchPG
{
    struct DetResult
    {
        float x1;
        float y1;
        float x2;
        float y2;
        int idx;
        float score;
    };

    enum ModelName
    {
        YOLOV5,
        YOLOv7,
        YOLOv8,
        YOLOX,
    };

    class AbstractPostProcessor
    {
    protected:
        float scoreThresh;
        float iouThresh;

    public:
        AbstractPostProcessor(float scoreThresh = 0.5, float iouThresh = 0.5)
            : scoreThresh(scoreThresh), iouThresh(iouThresh){};

        ~AbstractPostProcessor(){};

        virtual std::vector<DetResult> postProcess(torch::Tensor preds) = 0;
    };

} // namespace libtorchPG

#endif
