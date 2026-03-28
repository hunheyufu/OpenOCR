/*** 
 * @Author: hunheyufu yuyueyang2468@163.com
 * @Date: 2026-03-26 22:26:11
 * @LastEditors: hunheyufu yuyueyang2468@163.com
 * @LastEditTime: 2026-03-28 10:46:19
 * @FilePath: /Calculate/src/infer.cpp
 * @Description: 
 * @
 * @Copyright (c) 2026 by ACTION, All Rights Reserved. 
 */
#include "infer.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <chrono>

namespace OpenOCR
{
void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept
{
    // Only log messages that are at or above the reportable severity level
    if (severity > reportableSeverity)
    {
        return;
    }

    // Log the message with appropriate severity prefix
    switch (severity)
    {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "[INTERNAL ERROR] " << msg << std::endl;
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "[ERROR] " << msg << std::endl;
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "[WARNING] " << msg << std::endl;
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "[INFO] " << msg << std::endl;
            break;
        case nvinfer1::ILogger::Severity::kVERBOSE:
            std::cerr << "[VERBOSE] " << msg << std::endl;
            break;
        default:
            std::cerr << "[UNKNOWN] " << msg << std::endl;
            break;
    }
}

void CheckCuda(cudaError_t code, const std::string& stage)
{
    if (code != cudaSuccess)
    {
        std::cerr << "CUDA Error during " << stage << ": " << cudaGetErrorString(code) << std::endl;
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(code)));
    }
}

int64_t Volume(const nvinfer1::Dims& dims)
{
    int64_t volume = 1;
    for (int i = 0; i < dims.nbDims; i++)
    {
        // std::cout << dims.d[i] << "\t";
        volume *= dims.d[i];
    }
    // std::cout << "\n" << volume << std::endl;
    return volume;
}

TrtInfer::TrtInfer(const std::string& enginePath, const std::string& dictPath, int warmup) : warmup(warmup)
{

    dict = loadDict(dictPath);

    auto engineData = loadBinaryFile(enginePath);

    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime)        
    {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }
    engine.reset(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!engine)
    {
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }
    context.reset(engine->createExecutionContext());
    if (!context)
    {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    for (int i = 0; i < engine->getNbIOTensors(); i++)
    {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            inputTensorName = name;
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            outputTensorName = name;
        }
    }
    if (inputTensorName.empty() || outputTensorName.empty())
    {
        throw std::runtime_error("Failed to find input or output tensor names in the engine");
    }

    inputDims = engine->getTensorShape(inputTensorName.c_str());
    if (inputDims.nbDims != 4)
    {
        throw std::runtime_error("Expected input tensor to have 4 dimensions, but got " + std::to_string(inputDims.nbDims));
    }

    if (inputDims.d[2] > 0)
    {
        engine_h = inputDims.d[2];
    }
    else
    {
        engine_h = 48;
    }
    if (inputDims.d[3] > 0)
    {
        engine_w = inputDims.d[3];
    }
    else
    {
        engine_w = 320;
    }
    
    outputDims  = engine->getTensorShape(outputTensorName.c_str());
    time_steps  = outputDims.d[1] > 0 ? outputDims.d[1] : 40; // ((((320 - 1)/4) - 1)/2) + 1 = 40
    num_classes = outputDims.d[2] > 0 ? outputDims.d[2] : dict.size(); // blank(1) + characters(17) + 1 = 19

    CheckCuda(cudaStreamCreate(&stream), "creating CUDA stream");

    if (this->warmup)
    {
        for (int i = 0; i < this->warmup; i++)
        {
            warmUP();
        }
    }

    std::cout << "TensorRT engine loaded successfully with warmup " << this->warmup << "times. \nInput tensor: " << inputTensorName
              << ", Output tensor: " << outputTensorName
              << "\nEngine input shape: [" << banch_size << "x" << inputDims.d[1] << "x" << engine_h << "x" << engine_w << "]"
              << "\nEngine output shape: [" << banch_size << "x" << time_steps << "x" << num_classes << "]"
              << std::endl;
}

TrtInfer::~TrtInfer()
{
    if (stream)
    {
        cudaStreamDestroy(stream);
    }
}

/**
 * @brief 执行推理前的预热，主要是为了让TensorRT提前加载和优化引擎，减少第一次推理的延迟
 * 
 * @return int 
 */
int TrtInfer::warmUP()
{
    int out_times = time_steps;
    int out_classes = num_classes;

    const int64_t in_count = static_cast<int64_t>(3) * engine_h * engine_w;
    const int64_t out_count = static_cast<int64_t>(1) * time_steps * num_classes;
    std::vector<float> host_input(static_cast<size_t>(in_count), 0.0F);
	std::vector<float> host_output(static_cast<size_t>(out_count), 0.0F);

    void *device_input = nullptr;
    void *device_output = nullptr;
    CheckCuda(cudaMalloc(&device_input, in_count * sizeof(float)), "allocating device input buffer");
    CheckCuda(cudaMalloc(&device_output, out_count * sizeof(float)), "allocating device output buffer");

    // Set input shape before execution
    nvinfer1::Dims runtime_input_dims = inputDims;
    runtime_input_dims.d[0] = banch_size;
    runtime_input_dims.d[2] = engine_h;
    runtime_input_dims.d[3] = engine_w;
    if (!context->setInputShape(inputTensorName.c_str(), runtime_input_dims))
    {
        cudaFree(device_input);
        cudaFree(device_output);
        throw std::runtime_error("Failed to set input shape for warmup");
    }

    if (!context->setTensorAddress(inputTensorName.c_str(), device_input) ||
        !context->setTensorAddress(outputTensorName.c_str(), device_output))
    {
        cudaFree(device_input);
        cudaFree(device_output);
        throw std::runtime_error("Failed to set tensor addresses for warmup");
    }

    CheckCuda(cudaMemcpyAsync(device_input, host_input.data(), in_count * sizeof(float), cudaMemcpyHostToDevice, stream), 
        "copying input data to device for warmup");
    if (!context->enqueueV3(stream))
    {
        cudaFree(device_input);
        cudaFree(device_output);
        throw std::runtime_error("Failed to execute inference for warmup");
    }
    CheckCuda(cudaMemcpyAsync(host_output.data(), device_output, out_count * sizeof(float), cudaMemcpyDeviceToHost, stream), 
        "copying output data to host for warmup");
    CheckCuda(cudaStreamSynchronize(stream), "synchronizing CUDA stream for warmup");

    cudaFree(device_input);
    cudaFree(device_output);

    return 0;
}

/**
 * @brief 加载二进制文件到内存
 * 
 * @param filePath engine文件路径
 * @return std::vector<char> 文件内容的字节数组
 */
std::vector<char> TrtInfer::loadBinaryFile(const std::string& filePath)
{
    std::ifstream fin(filePath, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }
    fin.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fin.tellg());
    fin.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    fin.read(buffer.data(), static_cast<std::streamsize>(size));
    return buffer;
}

/**
 * @brief 加载字典文件到内存，字典文件每行一个字符，索引从1开始，0保留给CTC blank
 * 
 * @param dictPath 字典文件路径
 * @return std::vector<std::string> 字典内容的字符串向量
 */
std::vector<std::string> TrtInfer::loadDict(const std::string& dictPath) {
    std::vector<std::string> dict;
    if (dictPath.empty()) {
        return dict;
    }
    std::ifstream fin(dictPath);
    if (!fin) {
        throw std::runtime_error("Failed to open dictionary: " + dictPath);
    }

    // Align with CTCLabelDecode.add_special_char: index 0 is blank.
    dict.emplace_back("blank");

    std::string line;
    while (std::getline(fin, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        dict.push_back(line);
    }
    return dict;
}

int TrtInfer::infer(const cv::Mat& image, float conf)
{
    std::vector<float> chw;
    preProcess(image, chw);
    nvinfer1::Dims runtime_input_dims = inputDims;
    runtime_input_dims.d[0] = banch_size;
    runtime_input_dims.d[2] = engine_h;
    runtime_input_dims.d[3] = engine_w;
    std::vector<float> output(Volume(runtime_input_dims));
    execute(chw, output);
    postProcess(output);
    if (score < conf)
    {
        result = "";
        return 1;
    }
    return 0;
}

int TrtInfer::preProcess(const cv::Mat& image, std::vector<float>& chw)
{
    if (image.empty())
    {
        throw std::runtime_error("Input image is empty");
    }

    // bgr->rgb
    cv::Mat rgb_img;
    cv::cvtColor(image, rgb_img, cv::COLOR_BGR2RGB);

    // keep aspect ratio and pad to target width
    const float ratio = static_cast<float>(rgb_img.cols) / static_cast<float>(rgb_img.rows);
    int resized_w = static_cast<int>(std::ceil(static_cast<float>(engine_h) * ratio));
    resized_w = std::max(1, std::min(engine_w, resized_w));

    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(resized_w, engine_h), 0, 0, cv::INTER_CUBIC);

    cv::Mat f32;
    resized_img.convertTo(f32, CV_32FC3, 1.f / 255.f);
    
    chw.clear();
    chw.assign(static_cast<size_t>(3) * engine_h * engine_w, 0.0F);
    const int hw = engine_h * engine_w;
    for (int i = 0; i < engine_h; i++)
    {
        const auto *row = f32.ptr<cv::Vec3f>(i);
        for (int j = 0; j < resized_w; j++)
        {
            const cv::Vec3f& pixel = row[j];
            chw[0 * hw + i * engine_w + j] = (pixel[0] - 0.5f) / 0.5f; // R
            chw[1 * hw + i * engine_w + j] = (pixel[1] - 0.5f) / 0.5f; // G
            chw[2 * hw + i * engine_w + j] = (pixel[2] - 0.5f) / 0.5f; // B
        }
    }
    return 0;
}

int TrtInfer::execute(const std::vector<float>& chw, std::vector<float>& output)
{
    nvinfer1::Dims runtime_input_dims = inputDims;
    runtime_input_dims.d[0] = banch_size;
    runtime_input_dims.d[2] = engine_h;
    runtime_input_dims.d[3] = engine_w;
    if (!context->setInputShape(inputTensorName.c_str(), runtime_input_dims))
    {
        throw std::runtime_error("Failed to set input shape for inference");
    }
    
    // [n, c, h, w] n = 1, c = 3, h = 48, w = 320
    const int64_t in_count = static_cast<int64_t>(3) * engine_h * engine_w;
    // [n, t, c] n = 1, t = 40, c = 19
    const int64_t out_count = static_cast<int64_t>(1) * time_steps * num_classes;
    void *device_input = nullptr;
    void *device_output = nullptr;
    CheckCuda(cudaMalloc(&device_input, in_count * sizeof(float)), "allocating device input buffer");
    CheckCuda(cudaMalloc(&device_output, out_count * sizeof(float)), "allocating device output buffer");

    if (!context->setTensorAddress(inputTensorName.c_str(), device_input) ||
        !context->setTensorAddress(outputTensorName.c_str(), device_output))
    {
        cudaFree(device_input);
        cudaFree(device_output);
        throw std::runtime_error("Failed to set tensor addresses for inference");
    }

    auto infer_start = std::chrono::high_resolution_clock::now();
    CheckCuda(cudaMemcpyAsync(device_input, chw.data(), in_count * sizeof(float), cudaMemcpyHostToDevice, stream), 
        "copying input data to device for inference");
    if (!context->enqueueV3(stream))
    {
        cudaFree(device_input);
        cudaFree(device_output);
        throw std::runtime_error("Inference failed.");
    }
    CheckCuda(cudaMemcpyAsync(output.data(), device_output, out_count * sizeof(float), cudaMemcpyDeviceToHost, stream), 
        "copying output data to host for inference");
    CheckCuda(cudaStreamSynchronize(stream), "synchronizing CUDA stream for inference");
    auto infer_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> infer_duration = infer_end - infer_start;
    std::cout << "Inference completed in " << infer_duration.count() << " ms." << std::endl;

    cudaFree(device_input);
    cudaFree(device_output);
    return 0;
}

int TrtInfer::postProcess(const std::vector<float>& output)
{
    result.clear();
    score = 0.0f;
    const int blank_id = 0;
    int prev = -1;
    std::vector<float> kept_probs;
    for (int t = 0; t < time_steps; t++)
    {
        const float* p = &output[t * num_classes];
        int cur = static_cast<int>(std::max_element(p, p + num_classes) - p);
        float prob = p[cur];
        if (cur == blank_id || cur == prev)
        {
            prev = cur;
            continue;
        }
        if (cur >= 0 && cur < static_cast<int>(dict.size()))
        {
            // Process the valid class index
            result += dict[cur];
        }
        kept_probs.push_back(prob);
        prev = cur;
    }
    if (!kept_probs.empty())
    {
        score = std::accumulate(kept_probs.begin(), kept_probs.end(), 0.0f) / static_cast<float>(kept_probs.size());
    }
    return 0;
}

}