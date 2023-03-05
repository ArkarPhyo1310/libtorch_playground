#include "LTPG/Models/Detection/PostProcessor/YoloV5/PostProcess.hpp"

using namespace libtorchPG;

std::vector<DetResult> YOLOV5PostProcessor::NMS(torch::Tensor preds)
{
    std::vector<DetResult> output;
    for (size_t i = 0; i < preds.sizes()[0]; ++i)
    {
        torch::Tensor pred = preds.select(0, i);

        // Filter by scores
        torch::Tensor scores = pred.select(1, 4) * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
        pred = torch::index_select(pred, 0, torch::nonzero(scores > scoreThresh).select(1, 0));
        if (pred.sizes()[0] == 0)
            continue;
        // (center_x, center_y, w, h) to (left, top, right, bottom)
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        // Computing scores and classes
        std::tuple<torch::Tensor, torch::Tensor>
            max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        pred.select(1, 5) = std::get<1>(max_tuple);
        torch::Tensor dets = pred.slice(1, 0, 6);
        torch::Tensor keep = torch::empty({dets.sizes()[0]});
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
        torch::Tensor indexes = std::get<1>(indexes_tuple);
        int count = 0;
        while (indexes.sizes()[0] > 0)
        {
            keep[count] = (indexes[0].item().toInt());
            count += 1;

            // Computing overlaps
            torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
            for (size_t i = 0; i < indexes.sizes()[0] - 1; ++i)
            {
                lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
            }
            torch::Tensor overlaps = widths * heights;
            // FIlter by IOUs
            torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
            indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iouThresh).select(1, 0) + 1);
        }
        keep = keep.toType(torch::kInt64);
        auto outputs = torch::index_select(dets, 0, keep.slice(0, 0, count));

        for (int i = 0; i < outputs.size(0); i++)
        {
            DetResult res;
            res.x1 = outputs.data()[i][0].item().toFloat();
            res.y1 = outputs.data()[i][1].item().toFloat();
            res.x2 = outputs.data()[i][2].item().toFloat();
            res.y2 = outputs.data()[i][3].item().toFloat();
            res.score = outputs.data()[i][4].item().toFloat();
            res.idx = outputs.data()[i][5].item().toInt();
            output.push_back(res);
        }
    }
    return output;
}

void YOLOV5PostProcessor::ScaleCoords(std::vector<DetResult> &dets)
{
    auto clip = [](float n, float lower, float upper)
    {
        return std::max(lower, std::min(n, upper));
    };

    std::vector<DetResult> detections;
    for (auto &i : dets)
    {
        float x1 = (i.x1 - this->pad[0]) / this->scale; // x padding
        float y1 = (i.y1 - this->pad[1]) / this->scale; // y padding
        float x2 = (i.x2 - this->pad[0]) / this->scale; // x padding
        float y2 = (i.y2 - this->pad[1]) / this->scale; // y padding

        i.x1 = clip(x1, 0, this->oldShape.width);
        i.y1 = clip(y1, 0, this->oldShape.height);
        i.x2 = clip(x2, 0, this->oldShape.width);
        i.y2 = clip(y2, 0, this->oldShape.height);
    }
}

std::vector<DetResult> YOLOV5PostProcessor::postProcess(torch::Tensor preds)
{
    std::vector<DetResult> res = this->NMS(preds);
    this->ScaleCoords(res);

    return res;
}
