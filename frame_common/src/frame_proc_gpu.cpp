#include <ros/ros.h>
#include <frame_common/frame_proc_gpu.h>

namespace frame_common {
    FrameProcGpu::FrameProcGpu(void)
    {
        StereoBM_ = cv::Ptr<cv::gpu::StereoBM_GPU>(
                new cv::gpu::StereoBM_GPU(
                    cv::gpu::StereoBM_GPU::PREFILTER_XSOBEL,
                    /* ndisparities = */ 128));
        ROS_INFO("GPU call is reasonable: %s",
                StereoBM_->checkIfGpuCallReasonable()?"true":"false");
    }
    // set up stereo frame
    // assumes frame has camera params already set
    // <nfrac> is nonzero if <imgr> is a dense stereo image
    void FrameProcGpu::setStereoFrame(Frame &frame, const cv::Mat &img, const cv::Mat &imgr, int nfrac,
		    bool setPointCloud)
    {
	setStereoFrame( frame, img, imgr, cv::Mat(), nfrac, setPointCloud );
    }


    void FrameProcGpu::setStereoFrame(Frame &frame,
                                      const cv::Mat &img, const cv::Mat &imgr,
                                      const cv::Mat &left_mask,
                                      int nfrac , bool setPointCloud)
    {
        setMonoFrame( frame, img, left_mask );
        frame.imgRight = imgr;

        // set stereo
        frame.disps.clear();

        gpu_img_left.upload(img);
        gpu_img_right.upload(imgr);

        (*StereoBM_)(gpu_img_left, gpu_img_right, gpu_disparity);
        cv::Mat disparity;
        gpu_disparity.download(disparity);

        int nkpts = frame.kpts.size();
        frame.goodPts.resize(nkpts);
        frame.pts.resize(nkpts);
        frame.disps.resize(nkpts);

        #pragma omp parallel for shared( disparity )
        for (int i=0; i<nkpts; i++)
          {
            double disp = disparity.at<double>(frame.kpts[i].pt.x,frame.kpts[i].pt.y);
            frame.disps[i] = disp;
            if (disp > 0.0)           // good disparity
              {
                frame.goodPts[i] = true;
                Eigen::Vector3d pt(frame.kpts[i].pt.x,frame.kpts[i].pt.y,disp);
                frame.pts[i].head(3) = frame.pix2cam(pt);
                frame.pts[i](3) = 1.0;
                //          cout << pts[i].transpose() << endl;
              }
            else
              frame.goodPts[i] = false;
          }

    }
}
