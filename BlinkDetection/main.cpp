
//#include <iostream>
#include <opencv2\opencv.hpp>
#include "blink_detector.h"
#include "blineME.h"
#include "blinkStats.h"
#include "FrameDump.h"

//#define READ_IMAGES_FROM_FOLDER
//cv::string face_cascade_name = "cascades\\haarcascade_frontalface_alt2.xml";
cv::string face_cascade_name = "cascades\\lbpcascade_frontalface.xml";

cv::Mat processFrameToDisplay(BlinkDetectorReturnType outputBlinkDetector);

int main()
{
    
    //blineMEBatch();
    BlinkDetectorReturnType outputBlinkDetector;
    
    cv::VideoCapture vidCapture;
    cv::CascadeClassifier face_cascade;
    BlinkDetector blinkDetect (face_cascade_name);
    FRAMEDUMP objectFrameDump;

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

        cv::Mat frame;
#ifdef READ_IMAGES_FROM_FOLDER
        frameNum = frameNum;        // Edit to change the frame number of first frame.
        cv::String sFrameNum;
        if (frameNum <10)
            sFrameNum = "0000" + std::to_string(frameNum);
        else if (frameNum <100)
            sFrameNum = "000" + std::to_string(frameNum);
        else if (frameNum <1000)
            sFrameNum = "00" + std::to_string(frameNum);
        else if (frameNum <1000)
            sFrameNum = "0" + std::to_string(frameNum);
        else 
            sFrameNum = std::to_string(frameNum);
        frame = cv::imread(rootDir + "frame_" + sFrameNum + ".png");
#else
        vidCapture.read(frame);
#endif
        

        outputBlinkDetector = blinkDetect.blink_detect(frame);
        objectFrameDump.AddFrameToInternalMemory(frame);
        /*
        double fExposure = vidCapture.get(CV_CAP_PROP_EXPOSURE);
        char text[255]; 
        sprintf(text, "Exposure= %f", fExposure);
        cv::putText(frame, text, cv::Point(30,30),cv::FONT_HERSHEY_COMPLEX_SMALL,1.0, cv::Scalar(0.5,0.5,0.5));
        */

        cv::Mat dispFrame = processFrameToDisplay(outputBlinkDetector);

        cv::imshow("Frame", dispFrame);
        int key = cv::waitKey(1);

        if (key == 'c')
            objectFrameDump.DumpFrameArrayToMemory("D:\\frame_dump");
        else if (key > 0)
            break;
            
    }
    
    return 0;
}


cv::Mat generateFrameWithEffect(cv::Mat frame, int effectID, cv::Rect &faceBox);

BlinkDetector::OutputState prevState = BlinkDetector::OutputState::OUTSTATE_BLINKDETECT_FRAME_IDLE;
int sustainCount = 5;
int effect = 0;
int framesToIgnore = 30;

cv::Mat processFrameToDisplay(BlinkDetectorReturnType outputBlinkDetector)
{

    if (outputBlinkDetector.outState == BlinkDetector::OutputState::OUTSTATE_BLINKDETECT_FRAME_IDLE)
    {
        sustainCount = 0;
        effect = 0;
        framesToIgnore = 30;
    } 
    else if (outputBlinkDetector.outState == BlinkDetector::OutputState::OUTSTATE_BLINKDETECT_FRAME_NOBLINK)
    {
        // Sustain the previous effect. Else do nothing
        if (framesToIgnore != 0)
            framesToIgnore--;

        if (sustainCount == 0)
            effect = 1;         // Normal RGB
        else
            sustainCount--;
        
    } 
    else if (outputBlinkDetector.outState == BlinkDetector::OutputState::OUTSTATE_BLINKDETECT_FRAME_BLINK)
    {
        if (framesToIgnore != 0)
        {
            framesToIgnore--;
        }
        else
        {
            sustainCount = 5;
            effect = 2;
        }
    }
    else if (outputBlinkDetector.outState == BlinkDetector::OutputState::OUTSTATE_BLINKDETECT_FRAME_IN_RESET)
    {
        framesToIgnore = 30;
        sustainCount = 30;
        effect = 3;
    }
    /*
    printf("sustainCount= %d, effect= %d, currState= %d, framesToIgnore= %d\n", 
                                sustainCount, effect, currState, framesToIgnore);*/


    return generateFrameWithEffect(outputBlinkDetector.frame, effect, outputBlinkDetector.faceBox);

};



cv::Mat generateFrameWithEffect(cv::Mat frame, int effectID, cv::Rect &faceBox)
{
    cv::Mat outputImg;

    if (effectID == 0)
    {
        outputImg = frame.clone();
        //outputImg = outputImg / 2;
        //cv::cvtColor(frame, outputImg, CV_BGR2GRAY);
    }
    else if (effectID == 1)
    {
        outputImg = frame;

		// Draw rectangle only if processing is happening
		if (faceBox.width > 0)
		{
			cv::Point lefttop(faceBox.x, faceBox.y);
			cv::Point rightbottom((faceBox.x + faceBox.width), (faceBox.y + faceBox.height));
			cv::rectangle(outputImg, lefttop, rightbottom, cv::Scalar(0, 255, 0), 2);
		}
    }
    else if (effectID == 2)
    {
        int BORDER = 75;
        outputImg = frame.clone();
        for (int j = 0; j < BORDER; j++)
        {
            for (int i = 0; i < BORDER; i++)
            {
				outputImg.at<cv::Vec3b>(j, i)[0] = (outputImg.at<cv::Vec3b>(j, i)[0] / 4);
                outputImg.at<cv::Vec3b>(j, i)[1] = (191 + outputImg.at<cv::Vec3b>(j, i)[1] / 4);
				outputImg.at<cv::Vec3b>(j, i)[2] = (outputImg.at<cv::Vec3b>(j, i)[2] / 4);
            }
        }

		// Draw rectangle only if processing is happening
		if (faceBox.width > 0)
		{
			cv::Point lefttop(faceBox.x, faceBox.y);
			cv::Point rightbottom((faceBox.x + faceBox.width), (faceBox.y + faceBox.height));
			cv::rectangle(outputImg, lefttop, rightbottom, cv::Scalar(0, 255, 0), 2);
		}

    }
    else if (effectID == 3)
    {
        int BORDER = 15;
        outputImg = frame.clone();
        //outputImg = outputImg / 2;

        // Draw rectangle only if processing is happening
        if (faceBox.width > 0)
        {
            cv::Point lefttop(faceBox.x, faceBox.y);
            cv::Point rightbottom((faceBox.x + faceBox.width), (faceBox.y + faceBox.height));
            cv::rectangle(outputImg, lefttop, rightbottom, cv::Scalar(255, 0, 0),2);
        }
    }

    return outputImg;

}