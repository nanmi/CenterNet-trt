#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#include "decode.h"





#define CHECK_CUDA(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.65
#define CONF_THRESH 0.5

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
const char* INPUT_BLOB_NAME = "input.1";
const char* OUTPUT_BLOB_NAME0 = "950";
const char* OUTPUT_BLOB_NAME1 = "956";
const char* OUTPUT_BLOB_NAME2 = "953";
static Logger gLogger;


int INPUT_W = 512;
int INPUT_H = 512;

const float mean[3]= {0.408f, 0.447f, 0.470f};
const float stde[3] = {0.289f, 0.274f, 0.278f};

float* blobFromImage(cv::Mat& img){
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (int c = 0; c < channels; c++) 
    {
        for (int  h = 0; h < img_h; h++) 
        {
            for (int w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    // (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f);
                    ((((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f) - mean[c]) / stde[c];
            }
        }
    }
    return blob;
}


// static resize and padding
static cv::Mat preprocess_img(cv::Mat& img, int net_input_w, int net_input_h)
{
    int w, h, x, y;
    float r_w = net_input_w / (img.cols*1.0);
    float r_h = net_input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = net_input_w;
        h = r_w * img.rows;
        x = 0;
        y = (net_input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = net_input_h;
        x = (net_input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(net_input_h, net_input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}


void post_process(std::vector<Detection> & result, const cv::Mat& img, int input_w, int input_h, const bool& forwardFace)
{
    using namespace cv;
    int mark;

    float scale = min(float(input_w)/img.cols, float(input_h)/img.rows);
    float dx = (input_w - scale * img.cols) / 2;
    float dy = (input_h - scale * img.rows) / 2;
    for(auto&item:result)
    {
        float x1 = (item.bbox.x1 - dx) / scale ;
        float y1 = (item.bbox.y1 - dy) / scale ;
        float x2 = (item.bbox.x2 - dx) / scale ;
        float y2 = (item.bbox.y2 - dy) / scale ;
        x1 = (x1 > 0 ) ? x1 : 0 ;
        y1 = (y1 > 0 ) ? y1 : 0 ;
        x2 = (x2 < img.cols  ) ? x2 : img.cols - 1 ;
        y2 = (y2 < img.rows ) ? y2  : img.rows - 1 ;
        item.bbox.x1  = x1 ;
        item.bbox.y1  = y1 ;
        item.bbox.x2  = x2 ;
        item.bbox.y2  = y2 ;
        if(forwardFace){
            float x,y;
            for(mark=0;mark<5; ++mark ){
                 x = (item.marks[mark].x - dx) / scale ;
                 y = (item.marks[mark].y - dy) / scale ;
                 x = (x > 0 ) ? x : 0 ;
                 y = (y > 0 ) ? y : 0 ;
                 x = (x < img.cols  ) ? x : img.cols - 1 ;
                 y = (y < img.rows ) ? y  : img.rows - 1 ;
                item.marks[mark].x = x ;
                item.marks[mark].y = y ;
            }
        }
    }
}

void draw_image(const std::vector<Detection> & result, cv::Mat& img, const bool& forwardFace)
{

    const cv::Scalar color{255, 0, 0};
    int box_think = (img.rows + img.cols) * 0.001f ;
    float label_scale = img.rows * 0.0009;

    for (const auto &item : result) {
        std::string label = std::to_string(item.classId);
        std::string clss_score = std::to_string(item.prob);
        std::string score = clss_score.substr(0, clss_score.find(".") + 2 + 1);

        std::cout << item.classId << " " << item.prob << std::endl;

        cv::rectangle(img, cv::Point(item.bbox.x1,item.bbox.y1), cv::Point(item.bbox.x2 ,item.bbox.y2), color, 1, 8, 0);

        cv::putText(img, label+" "+score, cv::Point(item.bbox.x1, item.bbox.y1), cv::FONT_HERSHEY_COMPLEX, label_scale , color, box_think/2, 8, 0);

        if(!forwardFace){
            cv::putText(img, label, cv::Point(item.bbox.x2,item.bbox.y2 - 5),
                    cv::FONT_HERSHEY_COMPLEX, label_scale , color, box_think/2, 8, 0);
        }
        if(forwardFace)
        {
            for(int mark=0;mark<5; ++mark )
            cv::circle(img, cv::Point(item.marks[mark].x, item.marks[mark].y), 1, cv::Scalar(255, 255, 0), 1);
        }

    }
}



int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 4 && std::string(argv[2]) == "-i") {
        const std::string engine_file_path {argv[1]};
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./ctdet ../model_trt.engine -i ../*.jpg  // deserialize engine file and run inference" << std::endl;
        return -1;
    }

    const std::string input_image_path {argv[3]};

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Create stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));


    const int batch_size = 1;
    const int num_classes = 80;
    const int net_w = 512;
    const int net_h = 512;
    const int net_oh = net_h / 4;
    const int net_ow = net_w / 4;
    const int kernel_size = 3;
    const float vis_thresh = 0.3f;

    const size_t input_size = net_h * net_w * 3 * batch_size;
    const size_t hm_size = net_oh * net_ow * num_classes * batch_size;
    const size_t wh_size = net_oh * net_ow * 2 * batch_size;
    const size_t reg_size = net_oh * net_ow * 2 * batch_size;
    const size_t output_size = hm_size * 6;

    void* buffers[4];
    float* output  = new float[output_size];
    float* output_gpu = nullptr;

    const int input_idx = context->getEngine().getBindingIndex(INPUT_BLOB_NAME);
    const int hm_idx = context->getEngine().getBindingIndex(OUTPUT_BLOB_NAME0);
    const int wh_idx = context->getEngine().getBindingIndex(OUTPUT_BLOB_NAME1);
    const int reg_idx = context->getEngine().getBindingIndex(OUTPUT_BLOB_NAME2);
    std::cout <<"Engine binding buffers'index, input:" << input_idx << " hm:" << hm_idx << " wh:" << wh_idx << " reg:" << reg_idx << std::endl;

    CHECK_CUDA(cudaMalloc((void**)&buffers[input_idx], sizeof(float) * input_size));
    CHECK_CUDA(cudaMalloc((void**)&buffers[hm_idx], sizeof(float) * hm_size));
    CHECK_CUDA(cudaMalloc((void**)&buffers[wh_idx], sizeof(float) * wh_size));
    CHECK_CUDA(cudaMalloc((void**)&buffers[reg_idx], sizeof(float) * reg_size));

    CHECK_CUDA(cudaMalloc((void**)&output_gpu , sizeof(float) * output_size));


    cv::Mat img = cv::imread(input_image_path);
    std::cout << "Input image width: " << img.cols << " height: " << img.rows << std::endl;

    cv::Mat pr_img = preprocess_img(img, net_w, net_h);
    
    float* blob = blobFromImage(pr_img);

    // run inference
    auto start = std::chrono::system_clock::now();

    cudaMemcpyAsync(buffers[input_idx], blob, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);

    // context->enqueue(batch_size, buffers, stream, nullptr);
    context->enqueueV2(buffers, stream, nullptr);

    CTdetforward_gpu(static_cast<const float *>(buffers[hm_idx]),static_cast<const float *>(buffers[reg_idx]),
                        static_cast<const float *>(buffers[wh_idx]),static_cast<float *>(output_gpu),
                            net_ow, net_oh, num_classes, kernel_size, vis_thresh);

    CHECK_CUDA(cudaMemcpyAsync(output, output_gpu, sizeof(float) * output_size, cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    int num_det = static_cast<int>(output[0]);
    std::cout << "Detection objs num: " << num_det << std::endl;

    std::vector<Detection> result;
    result.resize(num_det);
    memcpy(result.data(), &output[1], num_det * sizeof(Detection));

    post_process(result, img, net_w, net_h, false);

    draw_image(result, img, false);

    cv::imwrite("result.jpg", img);

    delete output;

    // destroy the engine
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
