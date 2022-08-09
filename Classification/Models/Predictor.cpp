#include "Classification/Models/Predictor.hpp"

using namespace classification;

Predictor::Predictor(const std::string &modelFile)
{
    try {
        this->model = torch::jit::load(modelFile);
        this->model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model!\n";
        abort();
    }

    std::cout << "Finish loading the model...\n";
    torch::jit::setGraphExecutorOptimize( false );
    std::cout << "Warming up model...\n"; 
    this->model.forward({torch::randn({1, 3, 224, 224})});
}

Predictor::~Predictor()
{

}

void Predictor::runInference(torch::Tensor inputTensor) 
{
    std::vector<torch::jit::IValue> input;
    input.push_back(inputTensor);

    auto startTime = std::chrono::high_resolution_clock::now();
    this->output = torch::softmax(this->model.forward(input).toTensor(), 1);
    auto endTime = std::chrono::high_resolution_clock::now();

    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    printf("Inference Time: %d ms\n", duration);
}

Result Predictor::getOutput()
{
    Result data;
    std::tuple<torch::Tensor, torch::Tensor> result = torch::max(this->output, 1);

    torch::Tensor prob = std::get<0>(result);
    torch::Tensor index = std::get<1>(result);
    
    data.prob = prob[0].item<float>();
    data.idx = index[0].item<int>();

    return data;
}