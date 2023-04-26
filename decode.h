

#ifndef CTDET_DECODE_H
#define CTDET_DECODE_H

void CTdetforward_gpu(const float *hm, const float *reg, const float *wh , float *output,
                      const int w,const int h,const int classes, const int kernerl_size, const float visthresh  );

void CTfaceforward_gpu(const float *hm, const float *wh, const float *reg, const float* landmarks, float *output,
                       const int w, const int h, const int classes, const int kernerl_size, const float visthresh );


struct Box{
    float x1;
    float y1;
    float x2;
    float y2;
};
struct landmarks{
    float x;
    float y;
};
struct Detection{
    //x1 y1 x2 y2
    Box bbox;
    //float objectness;
    int classId;
    float prob;
    landmarks marks[5];
};

#endif //CTDET_DECODE_H






