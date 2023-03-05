#ifndef UTILS_H
#define UTILS_H

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

    static float toFloat(int value)
    {
        return static_cast<float>(value);
    }
} // namespace libtorchPG

#endif