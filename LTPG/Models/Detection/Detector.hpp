#ifndef DETECTOR_H
#define DETECTOR_H

#include <string>
#include <vector>
#include <chrono>

#include <torch/torch.h>
#include <torch/script.h>

namespace libtorchPG 
{
    struct DetResult
    {
        float x;
        float y;
        float width;
        float height;
        int idx;
        float score;
    };

    class Detector
    {

    public:
        Detector(const std::string &modelFile);

        ~Detector();

        void runInference(torch::Tensor input);

        DetResult getOutput();
    
    private:
        torch::jit::script::Module model;
        torch::Tensor output;

        std::vector<DetResult> NMS();
        std::vector<DetResult> non_max_suppression(torch::Tensor preds, float score_thresh=0.5, float iou_thresh=0.5);
    };
}

#endif