
#include <opencv2\opencv.hpp>

class FrameInfo
{
public:
    cv::Mat frame;
    bool faceDetected;
    cv::Rect faceRegion;
    bool multipleFaces; // Not really required as such. Might be useful for debugging at a later stage.

    FrameInfo();
    cv::Rect ObtainRoiUnion(cv::vector<cv::Rect>& objects);
    // Region where face was detected.
};
