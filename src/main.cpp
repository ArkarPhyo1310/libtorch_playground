#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <cstdlib>

int main() {
    torch::Tensor tensor = torch::rand( { 2, 3 } );
    std::cout << tensor << std::endl;
    
    cv::Mat frame;
    cv::VideoCapture cap;
    
    int deviceID = 0;
    int apiID    = cv::CAP_ANY;

    cap.open(deviceID, apiID);

    if (!cap.isOpened()) {
        std::cerr << "Error! Unable to open camera\n";
        return -1;
    }

    std::cout << "Start grabbing" << std::endl
              << "Press any key to terminate..." << std::endl;
    
    for (;;) {
        cap.read(frame);

        if (frame.empty()) {
            std::cerr << "Error! blank frame grabbed\n";
            break;
        }

        cv::imshow("Webcam", frame);
        if (cv::waitKey(1) >= 0) break;
    }
    return 0;
}