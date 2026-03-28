/*** 
 * @Author: hunheyufu yuyueyang2468@163.com
 * @Date: 2026-03-26 22:26:59
 * @LastEditors: hunheyufu yuyueyang2468@163.com
 * @LastEditTime: 2026-03-28 10:29:26
 * @FilePath: /Calculate/include/infer.hpp
 * @Description: 
 * @
 * @Copyright (c) 2026 by ACTION, All Rights Reserved. 
 */
#pragma once

// TensorRT includes
#include <NvInfer.h>
#include <cuda_runtime.h>

// Standard library includes
#include <array>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace OpenOCR
{
class Logger : public nvinfer1::ILogger 
{
private:
    nvinfer1::ILogger::Severity reportableSeverity;
public:
    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) noexcept
        : reportableSeverity(severity)
    {
    }
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
};

struct TrtDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            delete obj;
        }
    }
};

void CheckCuda(cudaError_t code, const std::string& stage);
int64_t Volume(const nvinfer1::Dims& dims);

class TrtInfer
{
public:
    TrtInfer(const std::string& enginePath, const std::string& dictPath, int warmup = 3);
    ~TrtInfer();

    int infer(const cv::Mat& image, float conf = 0.8f);
    std::string getResult() const { return result; }
    float getScore() const { return score; }
private:
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context;
    cudaStream_t stream = nullptr;

    std::string inputTensorName;
    std::string outputTensorName;
    nvinfer1::Dims inputDims;
    nvinfer1::Dims outputDims;
    int banch_size = 1;
    int engine_h = 0;
    int engine_w = 0;
    int time_steps = 0;
    int num_classes = 0;
    int warmup;
    std::vector<std::string> dict;

    std::string result = "";
    float score = 0.f;
private:
    std::vector<char> loadBinaryFile(const std::string& filePath);
    std::vector<std::string> loadDict(const std::string& dictPath);
    int warmUP();
    int preProcess(const cv::Mat& image, std::vector<float>& chw);
    int execute(const std::vector<float>& chw, std::vector<float>& output);
    int postProcess(const std::vector<float>& output);
};
}