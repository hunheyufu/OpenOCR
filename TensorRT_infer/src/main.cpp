/*** 
 * @Author: hunheyufu yuyueyang2468@163.com
 * @Date: 2026-03-26 22:26:30
 * @LastEditors: hunheyufu yuyueyang2468@163.com
 * @LastEditTime: 2026-03-28 10:55:59
 * @FilePath: /Calculate/src/main.cpp
 * @Description: 
 * @
 * @Copyright (c) 2026 by ACTION, All Rights Reserved. 
 */
#include "infer.hpp"
#include "opencv2/opencv.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

// 构建完整的资源路径
const std::string img_dir = std::string(PROJECT_ROOT_DIR) + "/data/test/";
const std::string engine_path = std::string(PROJECT_ROOT_DIR) + "/engine/rec_model.engine";
const std::string dict_path = std::string(PROJECT_ROOT_DIR) + "/dict/dict.txt";
const std::string output_dir = std::string(PROJECT_ROOT_DIR) + "/output/";

struct Result
{
    cv::Mat image;
    std::string path;
    std::string name;
    std::string text;
    float score;
};

int main()
{
    // 验证关键文件是否存在
    if (!std::filesystem::exists(engine_path)) {
        std::cerr << "Error: Engine file not found at " << engine_path << std::endl;
        return 1;
    }
    if (!std::filesystem::exists(dict_path)) {
        std::cerr << "Error: Dict file not found at " << dict_path << std::endl;
        return 1;
    }
    
    std::cout << "Project root: " << PROJECT_ROOT_DIR << std::endl;
    std::cout << "Engine path: " << engine_path << std::endl;
    std::cout << "Dict path: " << dict_path << std::endl;

    // 创建输出目录
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    // 读取目录下全部图片
    std::vector<Result> results;
    for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
        if (entry.is_regular_file()) {
            std::string img_path = entry.path().string();
            cv::Mat img = cv::imread(img_path);
            if (img.empty()) {
                std::cerr << "Warning: Failed to read image " << img_path << std::endl;
                continue;
            }
            results.emplace_back();
            results.back().path = img_path;
            results.back().name = entry.path().filename().string();
            results.back().image = img;
        }
    }

    std::shared_ptr<OpenOCR::TrtInfer> infer;
    infer = std::make_shared<OpenOCR::TrtInfer>(engine_path, dict_path);

    // 检测图片中的文字
    for (auto& res : results)
    {
        infer->infer(res.image);
        res.text = infer->getResult();
        res.score = infer->getScore();
    }

    std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
        return a.name < b.name;
    });

    // 输出结果到文本文件
    std::ofstream fout(output_dir + "results.txt");
    if (!fout) {
        std::cerr << "Error: Failed to open output file for writing." << std::endl;
        return 1;
    }
    for (const auto& res : results) {
        fout << res.name << "\t" << res.text << "\t" << res.score << std::endl;
    }
    fout.close();

    return 0;
}