# image position
float32   x
float32   y
# stereo disparity
float32   d
# from cv::KeyPoint
float32   size
float32   angle
float32   response
int32     octave
int32     class_id
# from feature descriptor extractor
float32[] descriptor
# is this keypoint complete
bool      goodPt
