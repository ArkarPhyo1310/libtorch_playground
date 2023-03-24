#ifndef ABSTRACT_POST_PROCESSOR_H
#define ABSTRACT_POST_PROCESSOR_H

#include <vector>

#include <torch/torch.h>
#include "LTPG/Utils/Convert.hpp"
#include "LTPG/Utils/DataTypes.hpp"

namespace libtorchPG
{
    class AbstractPostProcessor
    {
    protected:
        float scoreThresh;
        float iouThresh;

    public:
        /**
         * @brief Construct a new Abstract Post Processor object
         * 
         * @param scoreThresh 
         * @param iouThresh 
         */
        AbstractPostProcessor(float scoreThresh = 0.5, float iouThresh = 0.5)
            : scoreThresh(scoreThresh), iouThresh(iouThresh){};

        /**
         * @brief Destroy the Abstract Post Processor object
         * 
         */
        ~AbstractPostProcessor(){};

        /**
         * @brief Perform Post-Processing (nms, re-scaling) for Detection Output Result
         * 
         * @param preds 
         * @return std::vector<DetResult> 
         */
        virtual std::vector<DetResult> postProcess(torch::Tensor preds) = 0;
    };

} // namespace libtorchPG

#endif
