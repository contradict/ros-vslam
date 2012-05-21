#include <ros/ros.h>
#include <frame_common/frame_proc_gpu.h>

namespace frame_common {
    FrameProcGpu::FrameProcGpu(void)
    {
        ORB_ = cv::Ptr<cv::gpu::ORB_GPU>(new cv::gpu::ORB_GPU());
        StereoBM_ = cv::Ptr<cv::gpu::StereoBM_GPU>(
                new cv::gpu::StereoBM_GPU(
                    cv::gpu::StereoBM_GPU::PREFILTER_XSOBEL,
                    ndisparities = 128));
        ROS_INFO("GPU call is reasonable: %s",
                StereoBM_->checkIfGpuCallReasonable()?"true":"false");
    }

    void FrameProcGpu::setStereoFrame(Frame &frame, const cv::Mat &img, const
			cv::Mat &imgr, int nfrac, bool setPointCloud)
    {
        gpu_img_left.upload(img);
        gpu_img_right.upload(imgr);

        frame.img = img;
        frame.imgRight = imgr;

        // set keypoints and descriptors
        frame.kpts.clear();

        (*ORB_)(gpu_image_left, cv::gpu::GpuMat(), gpu_keypoints_left);
		ORB_.downloadKeypoints(gpu_keypoints_left, frame.kpts);

        int nkpts = frame.kpts.size();

        frame.pts.resize(nkpts);
        frame.goodPts.assign(nkpts, false);
        frame.disps.assign(nkpts, 10);

        // set stereo
        frame.disps.clear();

        (*StereoBM)(gpu_img_left, gpu_img_right, gpu_disparity);
		cv::Mat disparity;
		gpu_disparity.download(disparity);

        frame.goodPts.resize(nkpts);
        frame.pts.resize(nkpts);
        frame.disps.resize(nkpts);

        #pragma omp parallel for shared( st )
        for (int i=0; i<nkpts; i++)
          {
            double disp = disparity.at<double>(frame.kpts[i].pt.x,frame.kpts[i].pt.y);
            frame.disps[i] = disp;
            if (disp > 0.0)           // good disparity
              {
                frame.goodPts[i] = true;
                Vector3d pt(frame.kpts[i].pt.x,frame.kpts[i].pt.y,disp);
                frame.pts[i].head(3) = frame.pix2cam(pt);
                frame.pts[i](3) = 1.0;
                //          cout << pts[i].transpose() << endl;
              }
            else
              frame.goodPts[i] = false;
          }

    }
}
