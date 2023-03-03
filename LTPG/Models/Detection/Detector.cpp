#include <opencv2/opencv.hpp>

#include "LTPG/Models/Detection/Detector.hpp"

using namespace libtorchPG;

Detector::Detector(const std::string &modelFile)
{
    try
    {
        this->model = torch::jit::load(modelFile);
        this->model.eval();
    }
    catch(const c10::Error &e)
    {
        std::cerr << e.what() << '\n';
        abort();
    }
    
    std::cout << "Finish loading the model...\n";
    torch::jit::setGraphExecutorOptimize(false);
    std::cout << "Warming up the model...\n";
    this->model.forward({torch::randn({1, 3, 640, 640})});
}

Detector::~Detector()
{
}

void Detector::runInference(torch::Tensor inputTensor)
{
    std::vector<torch::jit::IValue> input;
    input.push_back(inputTensor);

    auto startTime = std::chrono::high_resolution_clock::now();
    this->output = this->model.forward(input).toTuple()->elements()[0].toTensor();
    auto endTime = std::chrono::high_resolution_clock::now();

    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Output shape: " << this->output.sizes() << std::endl;
    printf("Inference Time: %d ms\n", duration);

    // std::vector<DetResult> results = this->non_max_suppression(this->output);
    // std::cout << "Result shape: " << results << std::endl;
}

std::vector<DetResult> Detector::non_max_suppression(torch::Tensor output_tensor, float score_thresh, float iou_thresh)
{
    // Extract the detection results from the output tensor
    torch::Tensor confidences = output_tensor.select(2, 4) * output_tensor.select(2, 5).sigmoid();
    torch::Tensor boxes = output_tensor.narrow(2, 0, 4).sigmoid() * 2 - 0.5;
    boxes.select(2, 0) += torch::arange(0, boxes.size(1)).unsqueeze(1).expand({boxes.size(1), 4}).to(boxes.device()) * 32;
    boxes.select(2, 1) += torch::arange(0, boxes.size(1)).unsqueeze(1).expand({boxes.size(1), 4}).to(boxes.device()) * 32;

    // Convert the tensors to OpenCV format
    cv::Mat confidences_mat(confidences.size(0), confidences.size(1), CV_32FC1, confidences.data_ptr<float>());
    cv::Mat boxes_mat(boxes.size(0), boxes.size(1), CV_32FC4, boxes.data_ptr<float>());

    // Apply NMS using OpenCV
    // std::vector<int> keep_indices;
    // cv::dnn::NMSBoxes(boxes_mat.reshape(1, boxes_mat.total() / 4), confidences_mat, score_thresh, iou_thresh, keep_indices);
    std::vector<cv::Rect> boxes_rect;
    std::vector<float> confidences_vec;
    for (int i = 0; i < boxes_mat.rows; i++) {
        cv::Rect rect;
        rect.x = boxes_mat.at<cv::Vec4f>(i, 0)[0];
        rect.y = boxes_mat.at<cv::Vec4f>(i, 0)[1];
        rect.width = boxes_mat.at<cv::Vec4f>(i, 0)[2] - boxes_mat.at<cv::Vec4f>(i, 0)[0];
        rect.height = boxes_mat.at<cv::Vec4f>(i, 0)[3] - boxes_mat.at<cv::Vec4f>(i, 0)[1];
        boxes_rect.push_back(rect);
        confidences_vec.push_back(confidences_mat.at<float>(i));
    }
    std::vector<int> keep_indices;
    cv::dnn::NMSBoxes(boxes_rect, confidences_vec, static_cast<float>(score_thresh), static_cast<float>(iou_thresh), keep_indices);


    // Get the final detection results
    std::vector<DetResult> detections;
    for (auto idx : keep_indices) {
        DetResult detection;
        detection.idx = output_tensor[idx][5].argmax().item<int>();
        detection.score = confidences[idx].item<float>();
        detection.x = boxes[idx][0].item<float>();
        detection.y = boxes[idx][1].item<float>();
        detection.width = boxes[idx][2].item<float>() - detection.x;
        detection.height = boxes[idx][3].item<float>() - detection.y;
        detections.push_back(detection);
    }
    return detections;
}