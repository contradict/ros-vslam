#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

namespace frame_image_proc {
class StereoMatcher {
    private:
        cv::StereoBM cpu_stereo_;
        cv::gpu::StereoBM_GPU gpu_stereo_;

        void do_gpu_stereo(const cv::Mat &left, const cv::Mat &right, cv::Mat &disp, int dtype);

    public:
        bool use_gpu;

    StereoMatcher() :
         cpu_stereo_(cv::StereoBM::BASIC_PRESET),
         gpu_stereo_(),
         use_gpu(false) {};

    void operator()(const cv::Mat &left, const cv::Mat &right, cv::Mat &disp, int dtype=CV_16S);

    void determineValidWindow(int width, int height, int &left, int &top, int &right, int &bottom);

    inline int getPreFilterSize() const
    {
        if(use_gpu) {
            return gpu_stereo_.preset==cv::gpu::StereoBM_GPU::PREFILTER_XSOBEL?1:0;
        } else {
          return cpu_stereo_.state->preFilterSize;
        }
    };

    inline void setPreFilterSize(int size)
    {
        if(use_gpu) {
            gpu_stereo_.preset=size>0?cv::gpu::StereoBM_GPU::PREFILTER_XSOBEL:cv::gpu::StereoBM_GPU::BASIC_PRESET;
        } else {
            cpu_stereo_.state->preFilterSize = size;
        }
    };

    inline int getPreFilterCap() const
    {
        if(use_gpu)
            return -1;
        else {
            return cpu_stereo_.state->preFilterCap;
        }
    };

    inline void setPreFilterCap(int cap)
    {
        if( !use_gpu )
            cpu_stereo_.state->preFilterCap = cap;
    };

    inline int getCorrelationWindowSize() const
    {
        if( use_gpu ) {
            return gpu_stereo_.winSize;
        } else {
            return cpu_stereo_.state->SADWindowSize;
        }
    };

    inline void setCorrelationWindowSize(int size)
    {
        if( use_gpu ) {
            gpu_stereo_.winSize = size;
        } else {
            cpu_stereo_.state->SADWindowSize = size;
        }
    };

    inline int getMinDisparity() const
    {
        if(use_gpu) {
            return 0;
        } else {
            return cpu_stereo_.state->minDisparity;
        }
    };

    inline void setMinDisparity(int min_d)
    {
        if( !use_gpu )
            cpu_stereo_.state->minDisparity = min_d;
    };

    inline int getDisparityRange() const
    {
        if( use_gpu ) {
            return gpu_stereo_.ndisp;
        } else {
            return cpu_stereo_.state->numberOfDisparities;
        }
    };

    inline void setDisparityRange(int range)
    {
        if( use_gpu ) {
            gpu_stereo_.ndisp = range;
        } else {
            cpu_stereo_.state->numberOfDisparities = range;
        }
    };

    inline int getTextureThreshold() const
    {
        if(use_gpu) {
            return (int)round(gpu_stereo_.avergeTexThreshold*(gpu_stereo_.winSize*gpu_stereo_.winSize));
        } else {
            return cpu_stereo_.state->textureThreshold;
        }
    };

    inline void setTextureThreshold(int threshold)
    {
        if(use_gpu) {
            gpu_stereo_.avergeTexThreshold = (float)threshold/
                (gpu_stereo_.winSize*gpu_stereo_.winSize);
        } else {
            cpu_stereo_.state->textureThreshold = threshold;
        }
    };

    inline float getUniquenessRatio() const
    {
        if(use_gpu) {
            return -1;
        } else {
            return cpu_stereo_.state->uniquenessRatio;
        }
    };

    inline void setUniquenessRatio(float ratio)
    {
        if( !use_gpu )
            cpu_stereo_.state->uniquenessRatio = ratio;
    };

    inline int getSpeckleSize() const
    {
        if(use_gpu) {
            return -1;
        } else {
            return cpu_stereo_.state->speckleWindowSize;
        }
    };

    inline void setSpeckleSize(int size)
    {
        if( !use_gpu )
            cpu_stereo_.state->speckleWindowSize = size;
    };

    inline int getSpeckleRange() const
    {
        if(use_gpu) {
            return -1;
        } else {
            return cpu_stereo_.state->speckleRange;
        }
    };

    inline void setSpeckleRange(int range)
    {
        if( !use_gpu )
            cpu_stereo_.state->speckleRange = range;
    };

};
}
