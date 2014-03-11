#ifndef __FACE_DETECTOR__
#define __FACE_DETECTOR__

#include <opencv2\opencv.hpp>
#include "FrameInfo.h"


#if 1
#define FD_PERSISTENCE 100
#define FD_VALIDATION_COUNT 10
#else
#define FD_PERSISTENCE 5
#define FD_VALIDATION_COUNT 3
#endif

// Even if the face is not visible for 5 frames, we do not remove the face box



struct FaceTrackingInfo
{
    bool isValid;
    int reqValidationCount;
    int remPersistenceCount;
    cv::Rect FaceDim;

    FaceTrackingInfo() {
        isValid = false;
        reqValidationCount= FD_VALIDATION_COUNT;
        remPersistenceCount= FD_PERSISTENCE;
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
    
    BlinkDetector(cv::string face_cascade_file);
    cv::Mat blink_detect( cv::Mat frame);
    

private:
    void resetBlinkStates();
    cv::Rect fObtainRoiUnion();
    bool fHasMotion(cv::Rect roi);
    cv::Rect postProcessFaces(cv::vector<cv::Rect>& faces);
    int addToInternalFaceArray(cv::Rect currFace);
	void updateExistingFaceArray(int index);
    cv::Rect getBestFaceFrmInternalArray();
	void StateMachine(cv::Rect faceRegion);
    int doMotionEstimation(cv::Mat newFrame, cv::Mat oldFrame, cv::Rect faceRegion, int Index, double &thres1);

    cv::CascadeClassifier face_cascade;
    FrameInfo *prevFrameInfo;
    FrameInfo *currFrameInfo;
    cv::Rect currBaseSizeBox;
    int countNoFace;
    FaceTrackingInfo *faceArray;
    int FrameNum;

    // Eye Tracking related
    int count;
    int blinkState;
    int count1;
    int prevCount;
    int framesInCurrState;
    cv::Mat prevFrame;    // Needs to be initialised
    cv::Mat refStartFrame;
};




#endif