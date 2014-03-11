
//#include <iostream>
#include <opencv2\opencv.hpp>
#include "blink_detector.h"
#include "blineME.h"


#define READ_IMAGES_FROM_FOLDER
//cv::string face_cascade_name = "cascades\\haarcascade_frontalface_alt2.xml";
cv::string face_cascade_name = "cascades\\lbpcascade_frontalface.xml";

int main()
{
    
    //blineMEBatch();
    
    
    cv::VideoCapture vidCapture;
    cv::CascadeClassifier face_cascade;
    BlinkDetector blinkDetect (face_cascade_name);

    vidCapture.open(0);

    if (vidCapture.isOpened() == false)
    {
        printf("Failed to open video device.");
        return -1;
    }

    //double fExposure = vidCapture.get(CV_CAP_PROP_EXPOSURE);
    //vidCapture.set(CV_CAP_PROP_EXPOSURE, -5);

    //vidCapture.set(CV_CAP_PROP_AUTO_EXPOSURE, 1 ); 

#ifdef READ_IMAGES_FROM_FOLDER
    cv::String rootDir = "D:\\work\\blinkdetection\\cpp\\BlinkDetection\\BlinkDetection\\frame_dump\\";
#endif
    int frameNum=0;
    while (true)
    {
        frameNum++;
        cv::String sFrameNum;
        if (frameNum <10)
            sFrameNum = "0" + std::to_string(frameNum);
        else
            sFrameNum = std::to_string(frameNum);

        cv::Mat frame;
#ifdef READ_IMAGES_FROM_FOLDER
        frame = cv::imread(rootDir + "frame_" + sFrameNum + ".png");
#else
        vidCapture.read(frame);
#endif
        

        frame = blinkDetect.blink_detect(frame);

        double fExposure = vidCapture.get(CV_CAP_PROP_EXPOSURE);
        char text[255]; 
        sprintf(text, "Exposure= %f", fExposure);
        cv::putText(frame, text, cv::Point(30,30),cv::FONT_HERSHEY_COMPLEX_SMALL,1.0, cv::Scalar(0.5,0.5,0.5));

        cv::imshow("Frame", frame);
        int key = cv::waitKey(1);

        if (key > 0)
            break;
            
    }
    
    return 0;
}