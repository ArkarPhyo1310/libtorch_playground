#ifndef DATATYPES_H
#define DATATYPES_H

namespace libtorchPG
{
    struct ClsResult
    {
        int idx;
        float prob;
    };

    struct DetResult
    {
        float x1;
        float y1;
        float x2;
        float y2;
        int idx;
        float score;
    };

    enum CropType
    {
        CenterCut,
        Stretch,
        LetterBox,
    };

    enum ModelName
    {
        YOLOV5,
        YOLOv7,
        YOLOv8,
        YOLOX,
    };

    enum LogType
    {
        InfoLog,
        ErrorLog,
        WarnLog,
        CriticalLog
    };
} // namespace libtorchPG

#endif