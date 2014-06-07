#ifndef __FRAMEINFO_H__
#define __FRAMEINFO_H__

#include <opencv2\opencv.hpp>

class FrameInfo
{
public:
    cv::Mat frameFullSize;
    cv::Mat frameFaceSec;
    bool faceDetected;
    std::vector<cv::Point2f> vectFaceRegion;
    cv::Rect faceRegion;
    bool multipleFaces; // Not really required as such. Might be useful for debugging at a later stage.

    FrameInfo();
    cv::Rect ObtainRoiUnion(cv::vector<cv::Rect>& objects);
    // Region where face was detected.
};
#endif
