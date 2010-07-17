#ifndef _PROJ_H_
#define _PROJ_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/StdVector>
#include <map>

#include <sba/node.h>

namespace sba
{
  class Proj; // Forward reference.
  
  /// Obnoxiously long type def for the map type that holds the point 
  /// projections in tracks.
  typedef std::map<const int, Proj, std::less<int>, Eigen::aligned_allocator<Proj> > ProjMap;

  /// PROJ holds a projection measurement of a point onto a
  /// frame.  They are a repository for the link between the frame and
  /// the point, with aux info such as jacobians
  class Proj
  {
    public:
      /// General & stereo constructor. To construct a monocular projection, 
      /// either use stereo = false or the other constructor.
      /// NOTE: sets the projection to be valid.
      Proj(int ci, Eigen::Vector3d &q, bool stereo = true);
      
      /// Monocular constructor. To construct a stereo projection, use other
      /// constructor.
      /// NOTE: sets the projection to be valid.
      Proj(int ci, Eigen::Vector2d &q);
      
      /// Default constructor. Initializes to default values, kp = <0 0 0>
      /// and ndi = <0>. Also sets the projection to be invalid.
      Proj();
      
      /// Node index, the camera node for this projection
      int ndi;
      
      /// Keypoint, u,v,d vector
      Eigen::Vector3d kp;
      
      /// projection error
      Eigen::Vector3d err;
      
      /// Whether the projection is Stereo (True) or Monocular (False).
      bool stereo;
      
      /// calculates re-projection error and stores it in <err>
      double calcErr(const Node &nd, const Point &pt);
      
      /// Get the correct squared norm of the error, depending on whether the
      /// point is monocular or stereo.
      double getErrSquaredNorm();
      
      /// Get the correct norm of the error, depending on whether the point is
      /// monocular or stereo.
      double getErrNorm();
      
      /** Monocular:
      
          dpc/dq = dR'/dq [pw-t], in homogeneous form, with q a quaternion param
          dpc/dx = -R' * [1 0 0]', in homogeneous form, with x a translation param
          d(px/pz)/du = [ pz dpx/du - px dpz/du ] / pz^2,
          works for all variables       
          
          Stereo:
          pc = R'[pw-t]            => left cam
          pc = R'[pw-t] + [b 0 0]' => right cam px only

          dpc/dq = dR'/dq [pw-t], in homogeneous form, with q a quaternion param
          dpc/dx = -R' * [1 0 0]', in homogeneous form, with x a translation param
          d(px/pz)/du = [ pz dpx/du - px dpz/du ] / pz^2,
          works for all variables
          only change for right cam is px += b */
      void setJacobians(const Node &nd, const Point &pt);
      
      /// Point-to-point Hessian (JpT*Jp).
      Eigen::Matrix<double,3,3> Hpp;
      
      /// Point-to-camera Hessian (JpT*Jc)
      Eigen::Matrix<double,3,6> Hpc;
      
      /// Camera-to-camera Hessian (JcT*Jc)
      Eigen::Matrix<double,6,6> Hcc;
      
      /// The B matrix with respect to points (JpT*Err)
      Eigen::Matrix<double,3,1> Bp;
      
      /// Another matrix with respect to cameras (JcT*Err)
      Eigen::Matrix<double,6,1> JcTE;
      
      /// Point-to-camera matrix (HpcT*Hpp^-1)
      Eigen::Matrix<double,6,3> Tpc;
      
      /// valid or not (could be out of bounds)
      bool isValid;
      
      /// scaling factor for quaternion derivatives relative to translational ones;
      /// not sure if this is needed, it's close to 1.0
      const static double qScale = 1.0;
      
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW // needed for 16B alignment
      
    protected:
      /// Set monocular jacobians/hessians.
      void setJacobiansMono_(const Node &nd, const Point &pt);
      
      /// Set stereo jacobians/hessians.
      void setJacobiansStereo_(const Node &nd, const Point &pt);
      
      /// Calculate error function for stereo.
      double calcErrMono_(const Node &nd, const Point &pt);
      
      /// Calculate error function for stereo.
      double calcErrStereo_(const Node &nd, const Point &pt);
      
  };
    
  class Track
  {
    public:
      /// Constructor for a Track at point <p>.
      Track(Point p);
      
      /// Default constructor for Track.
      Track();
      
      /// A map of all the projections of the point with camera index as key, 
      /// based off an STL map.
      ProjMap projections;
      
      /// An Eigen 4-vector containing the <x, y, z, w> coordinates of the point
      /// associated with the track.
      Point point;
  };
  
  
} // sba

#endif // _PROJ_H