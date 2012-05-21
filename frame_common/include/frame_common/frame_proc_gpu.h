#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "frame_common/frame.h"

namespace frame_common
{
  class FrameProcGpu : public FrameProc
  {
  public:
    /// Create a FrameProc object using ORB feature detector/descriptor
    FrameProcGpu(void);

    /// Set the feature detector.
    void setFrameDetector(const cv::Ptr<cv::FeatureDetector>& detector){};
    /// Set the descriptor extractor.
    void setFrameDescriptor(const cv::Ptr<cv::DescriptorExtractor>& extractor){};

    /// \brief Set up stereo frame, assumes frame has camera parameters already set.
    /// \param frame The frame to be processed.
    /// \param img   Left camera image.
    /// \param imgr  Right camera image.
    /// \param nfrac Fractional disparity. If above 0, then imgr is a 16-bit fractional disparity
    ///              image instead, with <nfrac> counts per pixel disparity
    /// \param setPointCloud.  True if point cloud is to be set up from disparities
    virtual void setStereoFrame(Frame &frame, const cv::Mat &img, const cv::Mat &imgr, int nfrac = 0, 
                        bool setPointCloud=false);
  private:
    cv::Ptr<gpu::ORB_GPU>      ORB_;
    cv::Ptr<gpu::StereoBM_GPU> StereoBM_;

    gpu::GpuMat gpu_img_left, gpu_img_right;
    gpu::GpuMat gpu_disparity;
    gpu::GpuMat gpu_keypoints_left, gpu_keypoints_right;
  };
}
