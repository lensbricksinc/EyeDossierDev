
//#include <iostream>
#include <opencv2\opencv.hpp>
#include "blink_detector.h"
#include "blineME.h"
#include "blinkStats.h"
#include "FrameDump.h"
#include "time_stats.h"
#include <fstream>
#include <iostream>
#include <string>
#include "ml_core.h"
#include "svm.h"

#define VERSION "3"

#define READ_IMAGES_FROM_FOLDER
#define ML_TRAIN
//#define NO_FRAME_LOOP

#define CAPTURE

#ifndef ML_TRAIN
#undef NO_FRAME_LOOP
#endif
//cv::string face_cascade_name = "cascades\\haarcascade_frontalface_alt2.xml";
cv::string face_cascade_name = "cascades\\haarcascade_frontalface_alt2.xml";
cv::string eye_cascade_name = "cascades\\haarcascade_mcs_lefteye.xml";// = "cascades\\haarcascade_eye_tree_eyeglasses.xml";
//cv::string version = VERSION;
//cv::string name = "chintak";
//cv::string dumpHOGFileName = "C:\\Users\\chintak\\Desktop\\testcases\\dataset\\"+name+"\\"+name+"_eye.dat";
//cv::string blinkMarkFileName = "C:\\Users\\chintak\\Desktop\\testcases\\dataset\\"+name+"\\"+name+".csv";
//cv::string readSVMModelName = "C:\\Users\\chintak\\Downloads\\svm_light_windows32\\test\\svm_model"+version+".xml";
//std::string modelName = "C:\\Users\\chintak\\Downloads\\libsvm-3.18\\windows\\test\\model"+version+".dat";
//cv::string dumpResSVMName1 = "C:\\Users\\chintak\\Desktop\\testcases\\dataset\\"+name+"\\"+name+"01\\";

cv::Mat processFrameToDisplay(BlinkDetectorReturnType outputBlinkDetector);

int main()
{
    
    //blineMEBatch();
    BlinkDetectorReturnType outputBlinkDetector;
    
    cv::VideoCapture vidCapture;
    BlinkDetector blinkDetect (face_cascade_name, eye_cascade_name);
	/*BlinkDetector blinkDetect (face_cascade_name, eye_cascade_name, modelName,
		dumpHOGFileName, blinkMarkFileName, readSVMModelName, dumpResSVMName1);*/

    FRAMEDUMP objectFrameDump;
    TIMETRACKER objectTimeTracker(20);

#ifdef CAPTURE
	
	//cv::string vidname = "C:\\Users\\chintak\\Desktop\\testcases\\dataset\\"+name+"\\"+name+"01.avi";
	vidCapture.open(0);
    if (vidCapture.isOpened() == false)
    {
        printf("Failed to open video device.");
        return -1;
    }
	
	cv::Mat frame;
	while (true)
	{
		if(!vidCapture.read(frame)) break;
		outputBlinkDetector = blinkDetect.blink_detect(frame,-1);

		cv::imshow("Video", frame);
		if(cv::waitKey(10) == 27) break;
	}
	
#else

#ifndef READ_IMAGES_FROM_FOLDER
    vidCapture.open(0);
    if (vidCapture.isOpened() == false)
    {
        printf("Failed to open video device.");
        return -1;
    }
#endif


#ifdef READ_IMAGES_FROM_FOLDER
    cv::String rootDir = "C:\\Users\\chintak\\Desktop\\testcases\\full_frame_sequence\\pranav01\\";

#ifdef ML_TRAIN
	// Get a list of frames with a blink.
	std::vector<int> frameNumsWithBlink = std::vector<int>();
	std::ifstream fileBlinkMarks(filenameManualBLinkMarkings);
	std::string line;
	if (!fileBlinkMarks.is_open())
		printf("Unable to read input file for manual blink markings\n");

	std::getline(fileBlinkMarks,line);		// Discard the header line

	while (std::getline(fileBlinkMarks,line))
	{
		std::stringstream strstr(line);
        std::string word = "";
		std::vector<std::string> all_words = std::vector<std::string>();
        while (getline(strstr,word, ','))
        {
            all_words.push_back(word);
        }

		if (all_words.size() == 0)
			continue;

		if (all_words.size() != 2)
			printf("Error in input format\n");

		int lastFrameWithBlink = std::stoi(all_words[1]);
	
		frameNumsWithBlink.push_back(lastFrameWithBlink);
	}
	fileBlinkMarks.close();

	frameNumsWithBlink.push_back(-10);		// This will never match. Right at the end.
#endif		// ML_TRAIN

#endif		// READ_IMAGES_FROM_FOLDER

	int currIndex = 0;
    int frameNum=0;
#ifndef NO_FRAME_LOOP
    while (true)
    {
        frameNum++;
		std::cout << "frameNum: " << frameNum << std::endl;
		/*
		if (frameNum == 5)
			frameNum = 5;
			*/
        cv::Mat frame;
#ifdef READ_IMAGES_FROM_FOLDER
        frameNum = frameNum;        // Edit to change the frame number of first frame.
        cv::String sFrameNum;
        if (frameNum <10)
            sFrameNum = "000" + std::to_string(frameNum);
        else if (frameNum <100)
            sFrameNum = "00" + std::to_string(frameNum);
        else if (frameNum <1000)
            sFrameNum = "0" + std::to_string(frameNum);
        else if (frameNum <10000)
            sFrameNum = "" + std::to_string(frameNum);
		else if (frameNum <100000)
            sFrameNum = "" + std::to_string(frameNum);
        else 
            sFrameNum = std::to_string(frameNum);
        frame = cv::imread(rootDir + "frame_" + sFrameNum + ".png");
#else
        vidCapture.read(frame);
#endif
		if (frame.empty())
			break;

        objectTimeTracker.addNewEntry();  
#ifdef ML_TRAIN
		if (frameNumsWithBlink[currIndex] < 0)
			break;			// Skip the rest of the frames.

		if (frameNum == frameNumsWithBlink[currIndex])
		{
			outputBlinkDetector = blinkDetect.blink_detect(frame,1);
			currIndex ++;
		}
		else
		{
			outputBlinkDetector = blinkDetect.blink_detect(frame,0);
		}
#else
		outputBlinkDetector = blinkDetect.blink_detect(frame,-1);
#endif
        objectFrameDump.AddFrameToInternalMemory(frame);

        double frameRate = objectTimeTracker.getFPS();

        char text[255]; 
        sprintf(text, "FPS= %f, FrameNum = %d", frameRate, frameNum);
        cv::putText(frame, text, cv::Point(30,30),cv::FONT_HERSHEY_COMPLEX_SMALL,1.0, cv::Scalar(0.5,0.5,0.5));

        cv::Mat dispFrame = processFrameToDisplay(outputBlinkDetector);

        cv::imshow("Frame", dispFrame);
        if (outputBlinkDetector.matFaceBox.rows > 0)
            cv::imshow("Face Section", outputBlinkDetector.matFaceBox);
        else
            cv::destroyWindow("Face Section");

        int key = cv::waitKey(1);

        if (key == 'c')
            objectFrameDump.DumpFrameArrayToMemory("D:\\frame_dump");
        else if (key > 0)
            break;
            
    }
#endif

#ifdef ML_TRAIN
	blinkDetect.BlinkDetectionMLTrain();
#endif

#endif // CAPTURE

    return 0;
}


cv::Mat generateFrameWithEffect(cv::Mat frame, int effectID);

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
        sustainCount = 5;
        effect = 2;
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


    return generateFrameWithEffect(outputBlinkDetector.frame, effect);

};



cv::Mat generateFrameWithEffect(cv::Mat frame, int effectID)
{
    cv::Mat outputImg;
    outputImg = frame.clone();

    if (effectID == 2)
    {
        int BORDER = 75;
        for (int j = 0; j < BORDER; j++)
        {
            for (int i = 0; i < BORDER; i++)
            {
				outputImg.at<cv::Vec3b>(j, i)[0] = (outputImg.at<cv::Vec3b>(j, i)[0] / 4);
                outputImg.at<cv::Vec3b>(j, i)[1] = (191 + outputImg.at<cv::Vec3b>(j, i)[1] / 4);
				outputImg.at<cv::Vec3b>(j, i)[2] = (outputImg.at<cv::Vec3b>(j, i)[2] / 4);
            }
        }
    }

    return outputImg;
}