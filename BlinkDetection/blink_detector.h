#ifndef __FACE_DETECTOR__
#define __FACE_DETECTOR__

#include <opencv2\opencv.hpp>
#include "FrameInfo.h"
#include "statsMotion.h"
#include "blinkStats.h"

#if 1
#define FD_PERSISTENCE 25
#define FD_VALIDATION_COUNT 5
#else
#define FD_PERSISTENCE 5
#define FD_VALIDATION_COUNT 3
#endif

#define BLOCK_SIZE 8



struct FaceTrackingInfo
{
    bool isValid;
    int reqValidationCount;
    int remPersistenceCount;
    cv::Rect FaceDim;
	bool IsUpdated;

    FaceTrackingInfo() {
        isValid = false;
        reqValidationCount = FD_VALIDATION_COUNT;
        remPersistenceCount = FD_PERSISTENCE;
		FaceDim = cv::Rect(-1,-1,-1,-1);
    }

    void reset()
    {
        isValid = false;
        reqValidationCount= FD_VALIDATION_COUNT;
        remPersistenceCount= FD_PERSISTENCE;
		FaceDim = cv::Rect(-1,-1,-1,-1);
    }

};

class BlinkDetector{
public:
    cv::string face_cascade_name; // = "cascades\\haarcascade_frontalface_alt2.xml";
    //cv::string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

    BlinkDetector();
    ~BlinkDetector();
    
    BlinkDetector(cv::string face_cascade_file1,  cv::string face_cascade_file2);
    struct BlinkDetectorReturnType blink_detect(cv::Mat frame);
    

private:
	bool hasFaceBoxChanged(cv::Rect roi);
	int resetUpdateStateFaceArray();
	int updateFaceSizeForBlockProc(cv::vector<cv::Rect>& faces);
    void resetBlinkStates();
    cv::Rect fObtainRoiUnion();
    bool fHasMotion(cv::Rect roi);
    cv::Rect postProcessFaces(cv::vector<cv::Rect>& faces);
    int addToInternalFaceArray(cv::Rect currFace);
	void updateExistingFaceArray();
    cv::Rect getBestFaceFrmInternalArray();
	int StateMachine(cv::Rect faceRegion);
    int doMotionEstimation(cv::Mat newFrame, cv::Mat oldFrame, cv::Rect faceRegion, int Index, double &thres1, double minEyeBlocks, int &motion);

    cv::CascadeClassifier face_cascade;
    FrameInfo *prevFrameInfo;
    FrameInfo *currFrameInfo;
    cv::Rect currBaseSizeBox;
    cv::Rect prevFaceRect;
    int countNoFace;
    FaceTrackingInfo *faceArray;
    int FrameNum;

    // Eye Tracking related
    int count;
    int blinkState;
    int count1;
    int prevCount,prevcount1;
	int maxCount;
	int prevMotion;
    int framesInCurrState;
    cv::Mat prevFrame;    // Needs to be initialised
    cv::Mat refStartFrame;
	cv::Rect prevFaceBox;
	bool isReset;
	MotionRegionOnSAD *motionStats;

    BLINK_STATS *blinkStats;

public:
    enum OutputState
    {
        OUTSTATE_BLINKDETECT_FRAME_IDLE= 0,
        OUTSTATE_BLINKDETECT_FRAME_NOBLINK= 1,
        OUTSTATE_BLINKDETECT_FRAME_BLINK= 2,
        OUTSTATE_BLINKDETECT_FRAME_IN_RESET = 3
    };

};


typedef struct BlinkDetectorReturnType {
    cv::Mat frame;
    BlinkDetector::OutputState outState;
    cv::Rect faceBox;
}BlinkDetectorReturnType;


#endif