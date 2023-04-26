#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <utility>
#include <map>
#include <fstream>
#include <memory>
#include <NvInfer.h>
#include <cassert>
#include <sys/time.h>
#include <dirent.h>

#include "common/logger.h"
#include "dcn_v2.hpp" //! DCN plugin

//#include "gpu_sort.hpp"
//#include "det_kernels.hpp"
#include "custom.hpp"
#include <ctime>
#include <string>
// ----------------- thread header ------------------
#include <thread>
#include <mutex>
#include <chrono>
// --------------------------------------------------
// ---------------- udp header -----------------------
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
// -------------------------------------------------
// -------------------- random seed ----------------
#include "stdlib.h"
#include <time.h>
// --------------------------------------------------
using namespace std;
using namespace cv;
using namespace nvinfer1;
#include "process.hpp"
#include "decode.hpp"
// -------------- udp define --------------------------------------
#define SERVER_PORT 9999
#define SERVER_IP "127.0.0.1"
// ---------------------------------------------------------------
// ------------------ gloabl variable for thread -----------------
#define printf_h printf

#define BLUE(a) "\033[34m " a " \033[0m"
#define RED(a) "\033[31m " a " \033[0m"

mutex g_Lock;
Mat g_Image;
// --------------------------------------------------------------
#define CHECK_CUDA(e)                                                 \
    {                                                                 \
        if (e != cudaSuccess)                                         \
        {                                                             \
            printf("cuda failure: %s:%d: '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                            \
            exit(0);                                                  \
        }                                                             \
    }

struct NvInferDeleter
{
    template <typename T>
    void operator()(T *obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

#include "load_config.h"
GLB_APP_CTX *app_ctx_l = NULL;

int getBindingInputIndex(IExecutionContext *context)
{
    return !context->getEngine().bindingIsInput(0); // 0 (false) if bindingIsInput(0), 1 (true) otherwise
}

// ------------------- 从文件夹中返回文件名列表 --------------------------------
int read_files_in_dir(const char *p_dir_name, vector<string> &file_names)
{
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr)
        return -1;

    struct dirent *p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr)
    {
        if (strcmp(p_file->d_name, ".") != 0 && strcmp(p_file->d_name, ".."))
        {
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }
    return closedir(p_dir);
}
// -----------------------------------------------------------------------------
// ----------------------- udp client ------------------------------------------
void udp_msg_sender(int fd, struct sockaddr *dst, const char *buf, size_t BUFF_LEN)
{
    socklen_t len;
    // struct sockaddr_in src;
    len = sizeof(*dst);
    printf("client:%s\n", buf); //打印自己发送的信息
    sendto(fd, buf, BUFF_LEN, 0, dst, len);
}
// -----------------------------------------------------------------------------
vector<Scalar> colors(80);

const vector<pair<int, int>> edges{
    make_pair(0, 1), make_pair(0, 2), make_pair(1, 3), make_pair(2, 4),
    make_pair(3, 5), make_pair(4, 6), make_pair(5, 6), make_pair(5, 7),
    make_pair(7, 9), make_pair(6, 8), make_pair(8, 10), make_pair(5, 11),
    make_pair(6, 12), make_pair(11, 12), make_pair(11, 13), make_pair(13, 15),
    make_pair(12, 14), make_pair(14, 16)};

const vector<Scalar> e_colors{Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 255), Scalar(255, 0, 0), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 255), Scalar(255, 0, 0), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(0, 0, 255)};

const vector<Scalar> hp_colors{Scalar(255, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 0, 255)};
// 计时变量声明
struct timeval t1, t2, t3, t4, totalStart, totalEnd;
// -------------- define for opencv-gst ---------------------------
#define CAPTURE_DEVICE 0
#define CAPTURE_FPS 30
#define CAPTURE_WIDTH 1280
#define CAPTURE_HEIGHT 720
// ----------------------------------------------------------------
// --------------- 命令行参数判断 ------------------------------------
#define USE_RTSP 0
#define USE_CAMERA 1
#define USE_FILE 2

int open_sources(GLB_APP_CTX *app_ctx_l)
{

    //VideoCapture *capture = 0;
    string gst_str;
    int output_width = 1280;
    int output_height = 720;
    string uri;
    int i = 0;
    for (i = 0; i < app_ctx_l->engine_config.source_count; ++i)
    {
        output_width = app_ctx_l->engine_config.source_config[i].source_output_width;
        output_height = app_ctx_l->engine_config.source_config[i].source_output_height;

        if (app_ctx_l->engine_config.source_config[i].source_type == USE_FILE)
        {
            // --------------- test opencv with gst----------------------------

            //gst_str = "nvarguscamerasrc sensor-id=" + to_string(CAPTURE_DEVICE) + " ! video/x-raw(memory:NVMM), width=" + to_string(CAPTURE_WIDTH) + ", height=" + to_string(CAPTURE_HEIGHT) + ", format=(string)NV12, framerate=(fraction)" + to_string(CAPTURE_FPS) + "/1 ! nvvidconv ! video/x-raw, width=(int)" + to_string(output_width) + ", height=(int)" + to_string(output_height) + ", format=(string)BGRx ! videoconvert ! appsink";
            gst_str = app_ctx_l->engine_config.source_config[i].source_addr;
            cout << gst_str << endl;
            app_ctx_l->source_runtime_data[i].source_cap = new VideoCapture(gst_str);
            app_ctx_l->source_runtime_data[i].source_start_time = time(NULL);
            if (!((VideoCapture *)(app_ctx_l->source_runtime_data[i].source_cap))->isOpened())
            {
                printf("the rtsp camera cannot opening\n");
                return -1;
            }
            // ------------------------------------------------------------------------
        }
        else if (app_ctx_l->engine_config.source_config[i].source_type == USE_CAMERA)
        {
            // --------------- test opencv with gst----------------------------

            gst_str = "nvarguscamerasrc sensor-id=" + to_string(CAPTURE_DEVICE) + 
            " ! video/x-raw(memory:NVMM), width=" + to_string(CAPTURE_WIDTH) + 
            ", height=" + to_string(CAPTURE_HEIGHT) + 
            ",format=(string)NV12, framerate=(fraction)" + to_string(CAPTURE_FPS) + 
            "/1 ! nvvidconv ! video/x-raw, width=(int)" + to_string(output_width) + 
            ", height=(int)" + to_string(output_height) + 
            ", format=(string)BGRx ! videoconvert ! appsink";
            cout << gst_str << endl;
            app_ctx_l->source_runtime_data[i].source_cap = new VideoCapture(gst_str, CAP_GSTREAMER);
            app_ctx_l->source_runtime_data[i].source_start_time = time(NULL);
            if (!((VideoCapture *)(app_ctx_l->source_runtime_data[i].source_cap))->isOpened())
            {
                printf("the rtsp camera cannot opening\n");
                return -1;
            }
            // ------------------------------------------------------------------------
        }
        else if (app_ctx_l->engine_config.source_config[i].source_type == USE_RTSP)
        {
            // --------------- test opencv with gst----------------------------

            uri = app_ctx_l->engine_config.source_config[i].source_addr;
            int latency = 10;
            gst_str = "rtspsrc location=" + uri + 
            " latency=" + to_string(latency) + " ! " + 
            "rtph264depay ! h264parse ! omxh264dec ! " + "nvvidconv ! " + 
            "video/x-raw, width=(int)" + to_string(output_width) + 
            ",  height=(int)" + to_string(output_height) + 
            ", format=(string)BGRx ! " + "videoconvert ! appsink";
            cout << gst_str << endl;
            if(app_ctx_l->engine_config.engine_source_hard_decode)
            {
                app_ctx_l->source_runtime_data[i].source_cap = new VideoCapture(gst_str, CAP_GSTREAMER);
            }else
            {
                app_ctx_l->source_runtime_data[i].source_cap = new VideoCapture(uri);
                
            }

            app_ctx_l->source_runtime_data[i].source_start_time = time(NULL);
            if (!((VideoCapture *)(app_ctx_l->source_runtime_data[i].source_cap))->isOpened())
            {
                printf("the rtsp camera cannot opening\n");
                return -1;
            }else
            {
                ((VideoCapture *)(app_ctx_l->source_runtime_data[i].source_cap))->set(CAP_PROP_FRAME_WIDTH, output_width);//宽度 
                ((VideoCapture *)(app_ctx_l->source_runtime_data[i].source_cap))->set(CAP_PROP_FRAME_HEIGHT, output_height);//高度
            }

            // ------------------------------------------------------------------------
        }
        else
        {
            std::cerr << "arguments not right!" << std::endl;
            return -1;
        }
    }
    return 0;
    // --------------------------------------------------------------------------------
}
// ---------------------- 【on_MouseHandle()函数】------------------------------------
Rect g_rectangle;
bool g_bDrawingBox = false;
RNG g_rng(12345);
void DrawRectangle(cv::Mat &img, cv::Rect box)
{
    rectangle(img, box.tl(), box.br(), Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255)), 3);
}
void on_MouseHandle(int event, int x, int y, int flags, void *param)
{
    Mat &image = *(cv::Mat *)param;
    Point center;
    int thickness = -1;
    int lineType = 8;
    if (event == EVENT_MOUSEMOVE)
    {
        if (g_bDrawingBox)
        {
            g_rectangle.width = x - g_rectangle.x;
            g_rectangle.height = y - g_rectangle.y;
        }
    }
    // 左键按下信息
    else if (event == EVENT_LBUTTONDOWN)
    {
        g_bDrawingBox = true;
        cout << "x : " << x << "   y : " << y << endl;
        Point center;
        int thickness = -1;
        int lineType = 8;
        g_rectangle = Rect(x, y, 0, 0); // 记录起始点坐标

        // 画点
        center.x = x;
        center.y = y;
        circle(image, center, 10, Scalar(0, 0, 255), thickness, lineType);
        // imshow("【结果】", image);
    }
    else if (event == EVENT_LBUTTONUP)
    {
        // 左键抬起消息
        g_bDrawingBox = false; // 置标识符为false
        // 对宽和高小于0的处理
        if (g_rectangle.width < 0)
        {
            g_rectangle.x += g_rectangle.width;
            g_rectangle.width *= -1;
        }
        if (g_rectangle.height < 0)
        {
            g_rectangle.y += g_rectangle.height;
            g_rectangle.height *= -1;
        }

        // 调用函数进行绘制
        DrawRectangle(image, g_rectangle);
        // 画点
        center.x = x;
        center.y = y;
        circle(image, center, 10, Scalar(0, 0, 255), thickness, lineType);
    }
}

// --------------------------------------------------------------------------------
// ----------------------- 取流线程 -------------------------------------------
void fetchImage(int index)
{
    SOURCE_RUNTIME_DATA *source_runtime_data = app_ctx_l->source_runtime_data + index;
    ENGINE_CONFIG *engine_config = &app_ctx_l->engine_config;
    SOURCE_CONFIG *source_config = app_ctx_l->engine_config.source_config + index;

    cv::Mat frame;
    cv::Mat frame_bak;
    cv::Mat imgs;
    cv::Rect rects;

    //for (i = 0; i < app_ctx_l->engine_config.source_count; ++i)
    // {
    //     rects.x = source_config->source_region_left_top_x;
    //     rects.y = source_config->source_region_left_top_y;
    //     rects.width = source_config->source_region_right_bottom_x - source_config->source_region_left_top_x;
    //     rects.height = source_config->source_region_right_bottom_y - source_config->source_region_left_top_y;
    // }
    int output_height = engine_config->engine_infer_frame_size;
    int output_width = engine_config->engine_infer_frame_size;
    //((VideoCapture *)(app_ctx_l->source_runtime_data[i].source_cap))->set(CV_, 3);
    while (true)
    {
        //cout << "index:" << index << endl << " fetch image ..." << source_runtime_data->source_frame_id << endl;
        
        //cout << "..." << index << endl;

        // for (i = 0; i < app_ctx_l->engine_config.source_count; ++i)
        {
            *((VideoCapture *)(source_runtime_data->source_cap)) >> frame;
            //resize(app_ctx_l->engine_config.source_config[i].source_output_width
            //((VideoCapture *)(app_ctx_l->source_runtime_data[i].source_cap))->grab();
            //((VideoCapture *)(app_ctx_l->source_runtime_data[i].source_cap))->retrieve(frame);
            //frame = imread("../build/a.jpg");
            //if(i==0)
            //imwrite( "test.jpg",frame[0]);
            if (source_runtime_data->source_input_frame_height == 0 && frame.rows != 0)
            {
                source_runtime_data->source_input_frame_height = frame.rows;
                source_runtime_data->source_input_frame_width = frame.cols;

                if(source_config->source_region_right_bottom_x < 1)
                {
                    rects.x = source_config->source_region_left_top_x * frame.cols;
                    rects.y = source_config->source_region_left_top_y * frame.rows;
                    rects.width = (source_config->source_region_right_bottom_x - source_config->source_region_left_top_x) * frame.cols;
                    rects.height = (source_config->source_region_right_bottom_y - source_config->source_region_left_top_y) * frame.rows;
                }else
                {
                    rects.x = source_config->source_region_left_top_x;
                    rects.y = source_config->source_region_left_top_y;
                    rects.width = (source_config->source_region_right_bottom_x - source_config->source_region_left_top_x);
                    rects.height = (source_config->source_region_right_bottom_y - source_config->source_region_left_top_y);
                  
                }
                rects.x = min(rects.x, frame.cols - 3);
                rects.y = min(rects.y, frame.rows - 3);

                rects.width = min(max(1, frame.cols - rects.x - 1), rects.width - 2);
                rects.height = min(max(1, frame.rows - rects.y - 1), rects.height - 2);
                rects.width = min(rects.width, rects.height);
                rects.height = rects.width;

                source_runtime_data->source_region_rect_x = rects.x;
                source_runtime_data->source_region_rect_y = rects.y;
                source_runtime_data->source_region_rect_w = rects.width;
                source_runtime_data->source_region_rect_h = rects.height;
                source_runtime_data->source_region_resize_scale = (float)rects.height / engine_config->engine_infer_frame_size;
            }
            if (frame.cols != 0)
            {
                g_Lock.lock();
                    frame.copyTo(frame_bak);
                    frame(rects).copyTo(imgs);
                    resize(imgs, imgs, Size(output_width, output_height));
                    source_runtime_data->source_data_ptr = imgs.data;
                    source_runtime_data->source_org_data_ptr = frame_bak.data;
                    source_runtime_data->source_frame_id++;
                g_Lock.unlock();
            }
            else
            {
                if (source_config->source_type == USE_FILE)
                {
                    cout << "no image left in the video!" << endl;
                    break;
                }
            }
            
        }
        
        if (source_config->source_type == USE_FILE)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    exit(0);
}

int call_SVM(float *data)
{
    //init SVM
    static int data_len = app_ctx_l->engine_config.engine_pose_joint_count * 2 + edges.size();
    static Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>(app_ctx_l->engine_config.pose_SVM_config_path);
    static cv::Mat SampleMat = cv::Mat(1, data_len, CV_32FC1, data);
    SampleMat.data = (unsigned char*)data;
    if( !app_ctx_l->SVM_ptr )
    {
        app_ctx_l->SVM_ptr = (void *)svm;
    }
    
    
    // static float *norm_data = (float *)calloc(data_len, sizeof(float));
    // for(int i = 0 ;i < data_len; ++i)
    // {
    //     norm_data[i] = data[i] / app_ctx_l->engine_config.engine_infer_frame_size;
    // }
    int Response = -1;
    if(svm)
    {
        Response = static_cast<int>(svm->predict(SampleMat));
        
    }
    
    return Response;
}
int init_ctx(int argc, char **argv)
{
    int ret = -1;
    do
    {
        if (!app_ctx_l)
        {
            app_ctx_l = (GLB_APP_CTX *)calloc(1, sizeof(GLB_APP_CTX));
            if (!app_ctx_l)
            {
                break;
            }
        }
        if (load_config(argv[1], &(app_ctx_l->engine_config)) != 0)
        {
            printf("load config errror.\n");
            break;
        }

        app_ctx_l->source_runtime_data = (SOURCE_RUNTIME_DATA *)calloc(app_ctx_l->engine_config.source_count, sizeof(SOURCE_RUNTIME_DATA));

        const int K = app_ctx_l->engine_config.engine_max_obj_count;
        const int num_joints = app_ctx_l->engine_config.engine_pose_joint_count;

        const int det_len = 2 + 4 + num_joints * 2; // each object score + class +points
        const int batch_len = 1 + det_len * K;      // each object prediction

        for (int i = 0; i < app_ctx_l->engine_config.source_count; ++i)
        {
            app_ctx_l->source_runtime_data[i].source_id = i;
            app_ctx_l->source_runtime_data[i].source_res_history = (int *)calloc(app_ctx_l->engine_config.alarm_window_size + 1, sizeof(int));
            app_ctx_l->source_runtime_data[i].source_res_point_list = (float *)calloc(app_ctx_l->engine_config.engine_pose_joint_count * 2 + edges.size(), sizeof(float));
            app_ctx_l->source_runtime_data[i].source_res_center_net = (float *)calloc(batch_len, sizeof(float));
            app_ctx_l->source_runtime_data[i].source_alarm_code = (int *)calloc(6, sizeof(int));
        }
        // Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>(app_ctx_l->engine_config.pose_SVM_config_path);
        // app_ctx_l->SVM_ptr = (void *)svm;

        open_sources(app_ctx_l);

        // create push image thread
        static std::thread *threads = (std::thread *)calloc(app_ctx_l->engine_config.source_count + 1, sizeof(std::thread));
        for (int i = 0; i < app_ctx_l->engine_config.source_count; ++i)
        {
            threads[i] = thread(fetchImage, i);
        }

        for (int i = 0; i < app_ctx_l->engine_config.source_count; ++i)
        {
            threads[i].detach();
        }

        ret = 0;
    } while (0);
    return ret;
}

// ---------------------------------------------------------------------------
// ----------------------- drop duplicated box and keypoints ------------------
std::vector<float> dropDuplicated(int num, float *h_det, int num_bbox, int det_len)
{
    std::vector<float> v;
    v.push_back(num_bbox);
    if (num > 0)
    {
        for (int j = 0; j < det_len; j++)
        {
            v.push_back(h_det[1 + j]);
        }
        for (int i = num; i < num_bbox; i++)
        {
            for (int j = 0; j < det_len; j++)
            {
                v.push_back(h_det[num * det_len + 1 + j]);
            }
        }
    }
    else
    {
        for (int i = num; i < num_bbox; i++)
        {
            for (int j = 0; j < det_len; j++)
            {
                v.push_back(h_det[num * det_len + 1 + j]);
            }
        }
    }
    return v;
}

int get_pose_points(Mat &img, float *rm, int num_bbox, int batch_id)
{
    SOURCE_RUNTIME_DATA *source_tuntime_data = app_ctx_l->runtime_ctl.runtime_src_data_ptr;
    //SOURCE_CONFIG *source_config = app_ctx_l->runtime_ctl.runtime_src_cfg_ptr;
    float *h_det = source_tuntime_data->source_res_center_net;
    float *res = source_tuntime_data->source_res_point_list;
    //const int K = app_ctx_l->engine_config.engine_max_obj_count;
    const int num_joints = app_ctx_l->engine_config.engine_pose_joint_count;

    const int det_len = 2 + 4 + num_joints * 2; // each object 1 score + 1 class + 4 rect + 2 points * n
    //const int batch_len = 1 + det_len * K;      // each object prediction

    //for(int idx = 0; idx < (num_bbox * det_len + 1); idx++)
    //            cout << "[" << idx << "]" << h_det[idx] << "," << std::endl;
    // ------------------ udp传输h_det ----------------------
    //cout << "batch id:" << batch_id << "num bbox" << num_bbox  << "num bbox" << h_det[0] << endl;
    // -----------------------------------------------------

    const int font_face = FONT_HERSHEY_SIMPLEX;
    const double font_scale = 0.5;
    const int thickness = 1.5;
    int x_default = source_tuntime_data->source_region_rect_x;
    int y_default = source_tuntime_data->source_region_rect_y;
    int x_default1 = source_tuntime_data->source_region_rect_x + source_tuntime_data->source_region_rect_w;
    int y_default1 = source_tuntime_data->source_region_rect_y + source_tuntime_data->source_region_rect_h;
    float scale = source_tuntime_data->source_region_resize_scale;
    int x, y, w0, h0;
    //app_ctx_l->engine_config.source_config[batch_id].
    for (int i = 0; i < num_bbox; ++i)
    {
        // ---------------- detection visulize --------------------
        int x0 = h_det[1 + i * det_len + 2];
        int y0 = h_det[1 + i * det_len + 3];
        int x1 = h_det[1 + i * det_len + 4];
        int y1 = h_det[1 + i * det_len + 5];
        w0 = x1 - x0;
        h0 = y1 - y0;
        int cls_id = h_det[1 + i * det_len + 0];
        float score = h_det[1 + i * det_len + 1];
        //cout << " score:"
        //     << score << " bbox:[" << x0 << ","
        //     << y0 << "," << x1 << "," << y1 << "]\n";

        string text = "person";
        text += ":";
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << score;
        text += ss.str();

        //int baseline = 0;
        
        //cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

        if(app_ctx_l->engine_config.alarm_show_bbox)
        {
            cv::rectangle(img, cv::Point(x0 * scale + x_default, y0 * scale + y_default), cv::Point(x1 * scale + x_default, y1 * scale + y_default), colors[cls_id], 5);
        }
        
        if(app_ctx_l->engine_config.alarm_show_region)
        {
            cv::rectangle(img, cv::Point( x_default, y_default), cv::Point(x_default1, y_default1), Scalar(0 ,255 , 0), 1);
        }
        
        cv::putText(img, text, cv::Point(x0 * scale + x_default, y0 * scale - 3 + y_default), font_face, font_scale, cv::Scalar(0, 0, 255), thickness, cv::LINE_AA);
        // -------------------------------------------------------------------
        // ------------------ keypoints visualize -----------------------------
        for (int j = 0; j < num_joints; ++j)
        {
            x = h_det[1 + i * det_len + 6 + j * 2];
            y = h_det[1 + i * det_len + 6 + j * 2 + 1];
            res[j * 2] = (float)(x - x0) / w0;
            res[j * 2 + 1] = (float)(y - y0) / h0;
            cv::circle(img, cv::Point(x * scale + x_default, y * scale + y_default), 6, hp_colors[i], -1);
        }
        for (unsigned int j = 0; j < edges.size(); ++j)
        {
            int p0 = edges[j].first;
            int p1 = edges[j].second;

            x0 = h_det[1 + i * det_len + 6 + p0 * 2];
            y0 = h_det[1 + i * det_len + 6 + p0 * 2 + 1];
            x1 = h_det[1 + i * det_len + 6 + p1 * 2];
            y1 = h_det[1 + i * det_len + 6 + p1 * 2 + 1];

            res[num_joints * 2 + j] = pow(pow((float)(x1 - x0) / w0, 2) + pow((float)(y1 - y0) / h0, 2), 0.5);
            //app_ctx_l->source_runtime_data[batch_id].source_res_point_list[num_joints + j * 2 + 1] = y1 - y0;
            if (x0 > 0 && y0 > 0 && x1 > 0 && y1 > 0)
            {
                cv::line(img, cv::Point(x0 * scale + x_default, y0 * scale + y_default), cv::Point(x1 * scale + x_default, y1 * scale + y_default), e_colors[j], 4, cv::LINE_AA);
            }
        }
        // ----------------------------------------------------------
    }
    return 0;
}

// ------------------ draw result and udp ------------------------------------
int saveResults(Mat &img, float *h_det, int camera_id,
                int alarm_code, int on_alarm)
{
    SOURCE_RUNTIME_DATA *source_runtime_data = app_ctx_l->runtime_ctl.runtime_src_data_ptr;
    //camera_id = source_runtime_data->source_id;

    string img_name = to_string(time(NULL)) + "_" + to_string(camera_id) + "_" + to_string(alarm_code) + "_" + to_string(on_alarm) + ".jpg"; //相机ID，报警类别，报警/取消报警
    if (img_name.size() > MAX_STR_LEN)
    {
        abort();
    }
    if (source_runtime_data->source_save_img_name)
    {
        free(source_runtime_data->source_save_img_name);
    }
    source_runtime_data->source_save_img_name = (char *)calloc(1, img_name.size());
    memcpy(source_runtime_data->source_save_img_name, img_name.c_str(), img_name.size());
    // ---------------------------------------------------------------
    // -------------------- 输出文件保存 ------------------------------
    string ful_name = app_ctx_l->engine_config.alarm_image_dir;
    ful_name  += IMG_HEAD;

    size_t index = img_name.find_last_of("/\\");

    if (index != img_name.npos)
    {
        ful_name += img_name.substr(index + 1);
    }
    else
    {
        ful_name += img_name;
    }
    cout << "Saving the image to " << ful_name << endl;

    cv::imwrite(ful_name, img);
    // ---------------------------------------------------------------
    // ------------------ 计时函数 ------------------------------------
    //gettimeofday(&totalEnd, NULL);
    // 间隔时间
    //int totalTelta = (totalEnd.tv_sec - totalStart.tv_sec) * 1000 + (totalEnd.tv_usec - totalStart.tv_usec) / 1000;
    //cout << "total time of each iteration : " << totalTelta << " ms" << endl; // ms为单位
    //cout << endl;
    // ----------------------------------------------------------------
    return 0;
}
// ------
//disbaled
int post_email(char *img_path, unsigned char *img_data, int width, int height, int channels, char *addr, int camera_id, int on_alarm, int alarm_type)
{
    SOURCE_RUNTIME_DATA *source_runtime_data = app_ctx_l->source_runtime_data + camera_id;
//    ENGINE_CONFIG *engine_cfg = &app_ctx_l->engine_config;
//    SOURCE_CONFIG *source_config = app_ctx_l->engine_config.source_config + camera_id;

    if(! app_ctx_l->engine_config.alarm_send_email)
    {
        source_runtime_data->source_alarm_code[alarm_type] = ALARM_SEND;
        return 0; //disable email
    }
    printf_h("\r\n\r\npost_email for %s%s.\r\n\n\n", IMG_HEAD, img_path);
    char cmd[256] = {0};
    snprintf(cmd, 256, "./mail.py ./det_out/%s%s %d %d %d", IMG_HEAD, img_path, on_alarm, camera_id, alarm_type);
    system(cmd);
    source_runtime_data->source_alarm_code[alarm_type] = ALARM_SEND;
    return 0;
}

int start_alarm(void *src)
{
    int *data = (int *)src;
    int camera_id = data[0];
    int alarm_type = data[1];
    int on_alarm = data[2];

    SOURCE_RUNTIME_DATA *source_runtime_data = app_ctx_l->source_runtime_data + camera_id;
    ENGINE_CONFIG *engine_config = &app_ctx_l->engine_config;
    SOURCE_CONFIG *source_config = app_ctx_l->engine_config.source_config + camera_id;

    char *alarm_addr = NULL;

    if(on_alarm == ALARM_ON)
    {
        if (source_config->source_alarm_addr)
        {
            alarm_addr = source_config->source_alarm_addr;
        }
        else
        {
            alarm_addr = source_config->source_alarm_addr;
        }

        post_email(source_runtime_data->source_save_img_name,
                (unsigned char *)source_runtime_data->source_data_ptr,
                engine_config->engine_infer_frame_size,
                engine_config->engine_infer_frame_size,
                3, alarm_addr, camera_id, ALARM_ON, alarm_type);
        
    }
    return 0;
}

int stop_alarm(void *src)
{
    int *data = (int *)src;
    int camera_id = data[0];
    int alarm_type = data[1];
    int on_alarm = data[2];

    SOURCE_RUNTIME_DATA *source_runtime_data = app_ctx_l->source_runtime_data + camera_id;
    ENGINE_CONFIG *engine_config = &app_ctx_l->engine_config;
    SOURCE_CONFIG *source_config = app_ctx_l->engine_config.source_config + camera_id;

    char *alarm_addr = NULL;
    if(on_alarm == ALARM_OFF)
    {
        if (source_config->source_alarm_addr)
        {
            alarm_addr = source_config->source_alarm_addr;
        }
        else
        {
            alarm_addr = source_config->source_alarm_addr;
        }

        post_email(source_runtime_data->source_save_img_name,
                (unsigned char *)source_runtime_data->source_data_ptr,
                engine_config->engine_infer_frame_size,
                engine_config->engine_infer_frame_size,
                3, alarm_addr, camera_id, ALARM_OFF, alarm_type);
        source_runtime_data->source_alarm_code[alarm_type] = ALARM_NULL;
    }
    return 0;
}

int action_detect(Mat &img, int res, int i)
{
    SOURCE_RUNTIME_DATA *source_runtime_data = app_ctx_l->runtime_ctl.runtime_src_data_ptr;

    int slide_window = app_ctx_l->engine_config.alarm_window_size;
    int on_alarm_thrd = app_ctx_l->engine_config.alarm_confirm_thrd * slide_window;
    int sum = 0;
    int alarm_type_count = 5;
    if (res < 0)
    {
        return 0;
    }

    int index = source_runtime_data->source_res_counter++;
    if(source_runtime_data->source_res_counter > 1000000 * slide_window)
    {
        source_runtime_data->source_res_counter = 1;
    }
    source_runtime_data->source_res_history[index % slide_window] = res;
    sum = 0;

    //printf_h("handle_result[%d]:%d.\n", i, res);
    for (int k = 1; k < alarm_type_count; ++k)
    {
        // if( k != res && k != app_ctx_l->source_runtime_data[i].source_alarm_type)
        // {
        //    continue;
        // }
        sum = 0;

        for (int j = 0; j < slide_window; ++j)
        {
            if (source_runtime_data->source_res_history[j] == k)
            {
                sum += 1;
            }
        }
        //printf_h("handle_result[%d-%d]:%d/%d.\n", i, k, sum,on_alarm_thrd);
        if (sum > on_alarm_thrd)
        {
            if(k == 3)
            {
                int gg =1;
            }

            if (source_runtime_data->source_alarm_code[k] == ALARM_NULL)
            {
     
                saveResults(img, NULL, i,
                            k, ALARM_ON);
                int tmp[3] = {i, k, ALARM_ON};
                //thread postThread(start_alarm, tmp);
                //postThread.detach();
                source_runtime_data->source_alarm_code[k] = ALARM_ON;
                start_alarm(tmp);
            }
        }
        else
        {
            if (source_runtime_data->source_alarm_code[k] == ALARM_SEND)
            {
                
                saveResults(img, NULL, i,
                            k, ALARM_OFF);
                int tmp[3] = {i, k, ALARM_OFF};
                // thread postThread(stop_alarm, tmp);
                //postThread.detach();
                source_runtime_data->source_alarm_code[k] = ALARM_OFF;
                stop_alarm(tmp);
            }
        }
    }
    return 0;
}

int absent_detect(Mat &img, int index, int bbox_count)
{
    static long target_missing_start = 0;
    static int counter = 0;
    SOURCE_RUNTIME_DATA *source_runtime_data = app_ctx_l->runtime_ctl.runtime_src_data_ptr;
    //index = source_runtime_data->source_id;

    if (bbox_count == 0)
    {
        counter -= 2;
        if (counter < 0)
        {
            counter = 0;
        }
        if (target_missing_start != 0)
        {

            int time_last = time(NULL) - target_missing_start;
            if (time_last > app_ctx_l->engine_config.alarm_absent_confirm_thrd)
            {
                if (source_runtime_data->source_alarm_code[5] == ALARM_NULL)
                {
                    counter = 0;
                    int tmp[3] = {index, 5, ALARM_ON};
                    saveResults(img, NULL, index,
                            5, ALARM_ON);
                    start_alarm(tmp);
                }
            }
        }
        else
        {
            target_missing_start = time(NULL);
        }
    }
    else
    {
        counter++;
        if (counter > 10)
        {
            target_missing_start = 0;
            if (source_runtime_data->source_alarm_code[5] == ALARM_SEND)
            {

                int tmp[3] = {index, 5, ALARM_OFF};
                saveResults(img, NULL, index,
                            5, ALARM_OFF);
                stop_alarm(tmp);
            }
        }
        if (counter > 100000)
        {
            counter = 11;
        }
    }
    return 0;
}
//----------------------------------------------------------------------

int pose_infer(Mat &img, int index, int bbox_count)
{
    int alarm_on_flag = 0;
    SOURCE_RUNTIME_DATA *source_runtime_data = app_ctx_l->runtime_ctl.runtime_src_data_ptr;
    //index = source_runtime_data->source_id;

    //return if alarm on
    for (int l = 0; l < 6; ++l)
    {
        if (source_runtime_data->source_alarm_code[l] % 2 == 1)
        {
            alarm_on_flag = 1;
            break;
        }
    }
    if (alarm_on_flag == 0)
    {
        //get pose points data and draw it to org img
        get_pose_points(img, source_runtime_data->source_res_center_net, bbox_count, index);

        //call svm to get pose code
        int pose_code = 0;
        if (bbox_count > 1 || bbox_count <= 0)
        {
            pose_code = 0;
        }
        else
        {
            
            pose_code = call_SVM(source_runtime_data->source_res_point_list);
            if (pose_code == 1 || pose_code == 4 ||  pose_code == 0)
            {   
                std::cout << "camera id : " << index << ", classifier preidection response: " << pose_code << std::endl;
            }else
            {
                if(pose_code == 2)
                {
                    std::cout << RED("camera id : ") << index << ", classifier preidection response: " << pose_code << std::endl;
                }else
                {
                    std::cout << BLUE("camera id : ") << index << ", classifier preidection response: " << pose_code << std::endl;
                }
                
            }
            source_runtime_data->source_objs_pose_code = pose_code;
        }

        //for debug
        //if(bbox_count)
        //pose_code = 3;

        //take 1 and 4 as normal
        if (pose_code == 1 || pose_code == 4)
        {   
            pose_code = 0;
        }


        //only handle res with only one person detected
        if(bbox_count == 1)
        {
            action_detect(img, pose_code, index);
        }

        //detect absent every calling
        absent_detect(img, index, bbox_count);
    }
    return 0;
}

int parse_batch_res(float *h_det, int batch_index, Mat &org_mat)
{
    RUNTIME_CONTOLLER *runtime_ctl= &app_ctx_l->runtime_ctl;
    runtime_ctl->runtime_handling_source_index = batch_index;
    runtime_ctl->runtime_engine_cfg_ptr = &app_ctx_l->engine_config;
    runtime_ctl->runtime_src_data_ptr = &app_ctx_l->source_runtime_data[batch_index];
    runtime_ctl->runtime_src_cfg_ptr = &app_ctx_l->engine_config.source_config[batch_index];

    SOURCE_RUNTIME_DATA *source_runtime_data = runtime_ctl->runtime_src_data_ptr;
    ENGINE_CONFIG *engine_cfg = runtime_ctl->runtime_engine_cfg_ptr;
//    SOURCE_CONFIG *src_cfg = runtime_ctl->runtime_src_cfg_ptr;

    const int num_joints = engine_cfg->engine_pose_joint_count;
    const int det_len = 2 + 4 + num_joints * 2;    // each object score + class +points

    // ---------------------- 可视化输出结果 ------------------------------------
    //cout << "h_det:" << h_det[batch_id * batch_len] << endl;
    int num_bbox = static_cast<int>(h_det[0]);

    //cout << num_bbox << " human detected in the image!" << endl;
    if (num_bbox > 1)
    {
        srand((unsigned)time(NULL));
        //float max_score = h_det[1];
        int box_count = num_bbox;

        //cout << "the existing boxes valid: " << num_bbox << endl;

        for (int num = 1; num < num_bbox; num++)
        {
            int random_data = (rand() % (39 - 6 + 1)) + 6;
            assert(6 <= random_data && random_data <= 39);
            float base_point = h_det[random_data + 1];
            // 如果随机关键点的值相同那么则认为box重复
            if (h_det[num * det_len + 1 + random_data] - base_point < 1e-5 && h_det[num * det_len + 1 + random_data] - base_point > 1e-5)
            {
                //cout << "existing duplicated boxes------------------------------" << endl;
                // 如果存在重复box将det清空
                //det.clear();
                // 将检测的box有效信息-1
                box_count -= 1;
                // 选择两者之间置信度高的信息
                //if (max_score < h_det[num * det_len + 1])
                //{
                // 第二个的置信度大于第一个
                //    max_score = h_det[num * det_len + 1];
                //    det = dropDuplicated(0, h_det, num_bbox, det_len);
                //}
                //else
                //{
                // 第一个的置信度大于第二个
                //   det = dropDuplicated(num, h_det, num_bbox, det_len);
                //}
            }
        }

        cout << "remove duplicated boxes information ... existing valid: " << box_count << endl;
        source_runtime_data->source_objs_detected = box_count;
        source_runtime_data->source_res_center_net = h_det;
        pose_infer(org_mat, batch_index, box_count);
        //drawResults(img, (float*)&(det[0]), img_name, det_len, box_count, batch_id, batch_len, num_joints);
    }
    else
    {
        source_runtime_data->source_objs_detected = num_bbox;
        source_runtime_data->source_res_center_net = h_det;
        
        pose_infer(org_mat, batch_index, num_bbox);
        //drawResults(img, h_det, img_name, det_len, num_bbox, batch_id, batch_len, num_joints);
    }
    return 0;
}

int main(int argc, char *argv[])
{

    init_ctx(argc, argv);

    CHECK_CUDA(cudaSetDevice(0));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // --------------- read the serialized engine --------------------
    string trt_file(app_ctx_l->engine_config.engine_path);
    vector<char> trtModelStream_;
    size_t size(0);
    cout << "Loading engine file:" << trt_file << endl;
    ifstream file(trt_file, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream_.resize(size);
        file.read(trtModelStream_.data(), size);
        file.close();
    }
    else 
    {
        cerr << "Failed to open the engine file:" << trt_file << endl;
        return -1;
    }
    cout << " size: " << size << endl;
    // ---------------------------------------------------------------
    // -------------------- create engine ----------------------------
    auto runtime = unique_ptr<IRuntime, NvInferDeleter>(createInferRuntime(gLogger1));
    assert(runtime);
    auto engine = unique_ptr<ICudaEngine, NvInferDeleter>(runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr));
    if (!engine)
    {
        cerr << " Failed to create the engine from .trt file!" << endl;
        return -1;
    }
    else
    {
        cout << " Create the engine from " << trt_file << " successfully!" << endl;
    }
    auto input_dims = engine->getBindingDimensions(0);
    std::cout << "Input Dimensions = [" << input_dims.d[0] << "," 
                                        << input_dims.d[1] << "," 
                                        << input_dims.d[2] << "," 
                                        << input_dims.d[3] << "," 
                                        << "]" << std::endl;
    /// an execution context holds additional memory to store intermediate activation values. an engine can have multiple contexts sharing the same weights for multi-tasks/streams
    auto context = unique_ptr<IExecutionContext, NvInferDeleter>(engine->createExecutionContext());
    if (!context)
    {
        cerr << " Failed to createExecutionContext!" << endl;
        exit(-1);
    }
    // set an optimization profile 
    context->setOptimizationProfile(0);
    // set the input dimensions
    int batch_size = app_ctx_l->engine_config.engine_infer_batch_size;
    int infer_height = app_ctx_l->engine_config.engine_infer_frame_size;
    int infert_width = app_ctx_l->engine_config.engine_infer_frame_size;
    context->setBindingDimensions(0, Dims4(batch_size, 3, infer_height, infert_width));
    // -----------------------------------------------------------------
    const int nb_bindings = context->getEngine().getNbBindings();
    map<string, pair<int, size_t>> engine_name_size;
    // map<string, pair<int, size_t>>::iterator mit;
    vector<string> binding_names(nb_bindings);
    // ------------------ count input/output memory --------------
    string input_name = "input";
    
    for (int i = 0; i < nb_bindings; ++i)
    {
        auto dim = context->getEngine().getBindingDimensions(i);
        string name = context->getEngine().getBindingName(i);
        size_t size = dim.d[0] * dim.d[1] * dim.d[2] * dim.d[3];
        size_t pos = name.find("input");
        if (pos != name.npos)
            input_name = name;
        cout << "i=" << i << " tensor's name:"
             << name
             << " dim:("
             << dim.d[0] << ","
             << dim.d[1] << ","
             << dim.d[2] << ","
             << dim.d[3] << ")"
             << " size:" << size
             << endl;
        binding_names[i] = name;
        engine_name_size.emplace(name, make_pair(i, size));
    }

    // -----------------------------------------------------------
    /// memory allocation
    uint8_t *d_in;
    float *buffers[32];
    size_t *d_heat_ind, *d_hp_ind;
    float *d_det;
    float *h_det;
    /// calculate the affine transformation matrix
    float *d_inv_trans;
    float h_inv_trans[12];

    // ------------------ 解析命令行参数----------------------------------
    cout << "-i: single input image"
         << "\n"
         << "-l: multi-image in one folder"
         << "\n"
         << "-v: single video as input"
         << "\n"
         << "-gc: data from camera encoded by gstreamer"
         << "\n"
         << endl;
    // -----------------------------------------------------------------

    int step = 0;    // for output image name
    string img_name; // for output image name

    // assert(g_rectangle.size() == 4);
    // ROI 感兴趣区域
    // Rect roi(g_rectangle[0], g_rectangle[1], (g_rectangle[2]-g_rectangle[0]), (g_rectangle[3]-g_rectangle[1]));
    // ----------------------------------------------------------------------------
    
    int infer_size = infer_height * infert_width * 3;
    Mat img(infer_height, infert_width, CV_8UC3, Scalar(0, 0, 0));
    const int batch_num = batch_size;
    unsigned char *buf = (unsigned char *)calloc(infer_height * infert_width * 3, batch_size);
    unsigned char *org_buf = (unsigned char *)calloc(4096 * 4096 * 3, batch_size);
    app_ctx_l->runtime_ctl.runtime_org_frame_buf_ptr = org_buf;
    app_ctx_l->runtime_ctl.runtime_region_frame_buf_ptr = buf;

    img.data = buf;

    const size_t input_size = img.rows * img.cols * 3 * batch_num;
    const int net_h = 512;
    const int net_w = 512;
    const int down_ratio = 4;
    const int net_oh = net_h / down_ratio;
    const int net_ow = net_w / down_ratio;

    const int K = app_ctx_l->engine_config.engine_max_obj_count;
    const int num_joints = app_ctx_l->engine_config.engine_pose_joint_count;

    const int det_len = 2 + 4 + num_joints * 2;    // each object score + class +points
    const int batch_len = 1 + det_len * K;         // each object prediction
    const size_t det_size = batch_len * batch_num; // batch prediction

    cv::RNG rng(time(0));
    for (int i = 0; i < 80; ++i)
    {
        colors[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }

    CHECK_CUDA(cudaMalloc((void **)&d_inv_trans, sizeof(float) * 12));
    float center[2], scale[2];
    center[0] = img.cols / 2.;
    center[1] = img.rows / 2.;
    scale[0] = img.rows > img.cols ? img.rows : img.cols;
    scale[1] = scale[0];

    float shift[2] = {0., 0.};

    // -------------------------------- host to GPU ------------------------------
    CHECK_CUDA(cudaMallocHost((void **)&h_det, sizeof(float) * det_size));
    for (int i = 0; i < nb_bindings; ++i)
    {
        CHECK_CUDA(cudaMalloc((void **)&buffers[i], sizeof(float) * engine_name_size[binding_names[i]].second));
        CHECK_CUDA(cudaMemset(buffers[i], 0, sizeof(float) * engine_name_size[binding_names[i]].second));
    }

    CHECK_CUDA(cudaMalloc((void **)&d_det, sizeof(float) * det_size));
    CHECK_CUDA(cudaMalloc((void **)&d_in, sizeof(uint8_t) * input_size));
    CHECK_CUDA(cudaMalloc((void **)&d_heat_ind, sizeof(size_t) * engine_name_size["hm"].second));
    CHECK_CUDA(cudaMalloc((void **)&d_hp_ind, sizeof(size_t) * engine_name_size["hm_hp"].second));

    get_affine_transform(h_inv_trans, center, scale, shift, 0, net_h, net_w, true);                               //! src -> dst
    get_affine_transform(h_inv_trans + 6, center, scale, shift, 0, net_h / down_ratio, net_w / down_ratio, true); // det's
    CHECK_CUDA(cudaMemcpyAsync(d_inv_trans, h_inv_trans, sizeof(float) * 12, cudaMemcpyHostToDevice, stream));
    int org_frame_size = 0;
    Mat *org_mat = new Mat[app_ctx_l->engine_config.source_count + 1];
    int tmp_flag = 0;
    float engine_infer_pose_thrd = app_ctx_l->engine_config.engine_infer_pose_thrd;
    cout << "engine is working..." << endl;
    while (true)
    {

        // ----------------- camera data ---------------------------------------
        img_name = IMG_HEAD + to_string(step) + ".jpg";
        //std::cout << "load image from video encoded by gstreamer : " << img_name << std::endl;
        step++;
        if (step > 10000000)
        {
            step = 1;
        }
        // 计时函数
        gettimeofday(&t1, NULL);
        // capture->read(img);
        // (*capture) >> orignal_img;
        // 裁剪全局变量Image
        
        // imshow("[gloabl img]", g_Image);
        // waitKey(100);
        //g_Image(g_rectangle).copyTo(img);
        tmp_flag = 0;
        g_Lock.lock();
        for (int i = 0; i < app_ctx_l->engine_config.source_count; ++i)
        {
            org_frame_size = app_ctx_l->source_runtime_data[i].source_input_frame_height * app_ctx_l->source_runtime_data[i].source_input_frame_width * 3;
            if (org_frame_size < 0)
            {
                tmp_flag = 1;
                continue;
            }
            tmp_flag = 0;
            for (int l = 0; l < 6; ++l)
            {
                if (app_ctx_l->source_runtime_data[i].source_alarm_code[l] % 2 == 1)
                {
                    tmp_flag = 1;
                    break;
                }
            }

            if (tmp_flag == 0 && app_ctx_l->source_runtime_data[i].source_data_ptr)
            {
                
                memcpy(buf, app_ctx_l->source_runtime_data[i].source_data_ptr, infer_size);
                memcpy(buf + i * infer_size, app_ctx_l->source_runtime_data[i].source_data_ptr, infer_size);

                memcpy(org_buf, app_ctx_l->source_runtime_data[i].source_org_data_ptr, org_frame_size);
                memcpy(org_buf + i * 4096 * 4096, app_ctx_l->source_runtime_data[i].source_org_data_ptr, org_frame_size);
                
                if (org_mat[i].cols == 0)
                {
                    org_mat[i] = Mat(app_ctx_l->source_runtime_data[i].source_input_frame_height, app_ctx_l->source_runtime_data[i].source_input_frame_width, CV_8UC3,org_buf + i * 4096 * 4096);
                    //org_mat[i].data = org_buf + i * 4096 * 4096;
                }
                //imwrite("test" + to_string(i) + "test.jpg", org_mat[i]);
            }
        }
        g_Lock.unlock();
        
        if (tmp_flag)
        {
            continue;
        }
        //imwrite("test.jpg", org_mat[0]);
        // Mat img = imread("det_out/pose_pose_0.jpg");
        // imshow("result", img);
        // waitKey(0);
        // break;
        // ---------------------------------------------------------------------
        // 计时函数
        //gettimeofday(&totalStart, NULL);

        // capture >> img;
        //if(img.data == NULL) break;
        // string oringinal_name = "origin_img/original_" + to_string(step) + ".jpg";
        // cv::imwrite(oringinal_name, img);
        // -------------------------------------------------------------------
        // batch size

        //CHECK_CUDA(cudaMemsetHost((void**)&h_det, sizeof(float) * det_size));
        CHECK_CUDA(cudaMemset(d_det, 0, sizeof(float) * det_size));

        CHECK_CUDA(cudaMemcpyAsync(d_in, buf, sizeof(uint8_t) * input_size, cudaMemcpyHostToDevice, stream));

        float *d_mean = NULL;
        float *d_std = NULL;

        
        // ------------------ 输入预处理 -----------------------------------
        cuda_centernet_preprocess(d_in, batch_num,
                                  3, img.rows, img.cols,
                                  buffers[engine_name_size[input_name].first], net_h, net_w,
                                  d_inv_trans, d_mean, true,
                                  d_std, true, stream);
        // ----------------------------------------------------------------
        // 计时函数
        //gettimeofday(&t2, NULL);
        // 间隔时间
        //int deltaImage = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
        //cout << "image preprocessing time of each detection : " << deltaImage << " ms" << endl; // ms为单位

        //cout << " starting inference ... " << endl;

        // -------------------- 网络前向传播 ----------------------------
        context->enqueue(batch_num, (void **)buffers, stream, nullptr);
        // -------------------------------------------------------------

        // float* h_buffers[32];

        // static int tt = 1;
        // if(tt)
        // {
        // 	tt = 0;
        // 	for(int i = 0; i < nb_bindings; ++i) {
        // 		CHECK_CUDA(cudaMallocHost((void**)&h_buffers[i], sizeof(float) * engine_name_size[binding_names[i]].second));
        // 	}
        // 	///device -> host
        // 	for(int i = 0; i < nb_bindings; ++i) {
        // 		CHECK_CUDA(cudaMemcpyAsync(h_buffers[i], buffers[i], sizeof(float) * engine_name_size[binding_names[i]].second, cudaMemcpyDeviceToHost, stream));
        // 	}
        //     cudaStreamSynchronize(stream);
        // 	for(int i=0;i<10;++i)
        // 	{
        // 		printf("%f,",h_buffers[5][i]);
        // 		printf("%f\n",h_buffers[5][i + 128 * 128]);
        // 	}
        // }

        // 计时函数
        //gettimeofday(&t3, NULL);
        //int deltaForward = (t3.tv_sec - t1.tv_sec) * 1000 + (t3.tv_usec - t1.tv_usec) / 1000;
        //cout << "image preprocessing + forward time of each detection : " << deltaForward << " ms" << endl; // ms为单位

        // ------------------ decode the detection's result -----------------------------
        //for(mit = bzzdMap.begin();mit!=bzzdMap.end();mit++)
        //{


        multi_pose_decode(d_det,
                          buffers[engine_name_size["hm"].first],
                          buffers[engine_name_size["wh"].first],
                          buffers[engine_name_size["reg"].first],
                          buffers[engine_name_size["hps"].first],
                          buffers[engine_name_size["hm_hp"].first],
                          buffers[engine_name_size["hp_offset"].first],
                          d_heat_ind, d_hp_ind, d_inv_trans + 6,
                          batch_size, num_joints, net_oh, net_ow, K,
                          engine_infer_pose_thrd, true, true, stream);

        // -------------------------------------------------------------------------------
        gettimeofday(&t4, NULL);

        // 间隔时间
        int deltaDecode = (t4.tv_sec - t1.tv_sec) * 1000 + (t4.tv_usec - t1.tv_usec) / 1000;
        cout << "total time of each detection : " << deltaDecode << " ms" << endl; // ms为单位

        CHECK_CUDA(cudaMemcpyAsync(h_det, d_det, sizeof(float) * det_size, cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

       

        for (int i = 0; i < batch_size; ++i)
        {
            parse_batch_res(h_det + batch_len * i, i, org_mat[i]);
        }

        cudaStreamSynchronize(stream);

        //while(1);
        // ---------------------- 释放输入和输出变量的内存 -----------------------

        // ----------------------------------------------------------------------
        // waitKey(0);
    }
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_heat_ind));
    CHECK_CUDA(cudaFree(d_hp_ind));
    CHECK_CUDA(cudaFree(d_det));
    CHECK_CUDA(cudaFree(d_inv_trans));
    CHECK_CUDA(cudaFreeHost(h_det));
    for (int i = 0; i < nb_bindings; ++i)
        CHECK_CUDA(cudaFree(buffers[i]));

    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
