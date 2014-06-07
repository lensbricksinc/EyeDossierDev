#ifndef __BLINK_DETECTOR_H__
#define __BLINK_DETECTOR_H__

#include <opencv2\opencv.hpp>
#include "FrameInfo.h"
#include "statsMotion.h"
#include "blinkStats.h"
#include "ml_core.h"
#include "svm.h"

#define VERSION "3"
#define CONSECUTIVE_CLOSED_EYE_FOR_BLINK 4

//#define DUMP_HOG
#define TEST_SVM
//#define TEST_AND_DUMP_SVM_RESULTS
//#define TEST_AND_DUMP_OPEN
//#define TEST_AND_DUMP_CLOSED
//#define USE_HOG_DETECTOR

#define DETECT_THRESHOLD 0.0

#if 1
#define FD_PERSISTENCE 3
#define FD_VALIDATION_COUNT 1
#else
#define FD_PERSISTENCE 5
#define FD_VALIDATION_COUNT 3
#endif

#define BLOCK_SIZE 8
#define MIN_FACE_WIDTH 125

#define FEATURESET_INDEX 0

#if FEATURESET_INDEX == 0
#define BLINK_ML_FEATURES_PER_FRAME 2
#elif FEATURESET_INDEX == 1
#define BLINK_ML_FEATURES_PER_FRAME 9
#elif FEATURESET_INDEX == 2
#define BLINK_ML_FEATURES_PER_FRAME 101
#endif


std::vector<float> computeHOG(cv::Mat image, bool visualize);
void computeAndDumpHOG(cv::Mat image, long frameCount);

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
    cv::string eyes_cascade_name; // = "cascades\\haarcascade_eye_tree_eyeglasses.xml";

    BlinkDetector();
    ~BlinkDetector();
	
    std::vector<cv::Rect> prevEyes;

	BlinkDetector(cv::string face_cascade_file1, 
		cv::string eye_cascade_file);

    /*BlinkDetector(cv::string face_cascade_file1, 
		cv::string eye_cascade_file, 
		cv::string modelName,
		cv::string dumpHOGFileName,
		cv::string blinkCountReadName,
		cv::string readSVMModelName,
		cv::string dumpResSVMName);*/
    struct BlinkDetectorReturnType blink_detect(cv::Mat frame, int blinkStateFromFile = -1);
//	struct BlinkDetectorReturnType blink_detect_new(cv::Mat frame, int blinkStateFromFile = -1);
    int BlinkDetectionMLTrain();

#ifdef DUMP_HOG
	std::ifstream blinkMarkFileRead;
	int blinkFrameCount;
#endif 

#ifdef TEST_SVM
	svm_model * model;
	int closedEyeCount;
#endif

#ifdef TEST_AND_DUMP_SVM_RESULTS
	cv::string dumpResSVMName;
	float accuracy;
	int totalNumSamples;
	//std::ofstream dumpSVMResFile;
#endif


private:    
    FrameInfo *prevFrameInfo;
    FrameInfo *currFrameInfo;
    
    int mFrameNum;

#ifdef DUMP_HOG
	std::ofstream dumpHOGFile;
#endif

    // Face box/ Tracking related
    cv::CascadeClassifier face_cascade;
	cv::CascadeClassifier eye_cascade;

    // Face post processing
    FaceTrackingInfo *faceArray;
    int resetFaceData();
    int resetUpdateStateFaceArray();
    int updateFaceSizeForBlockProc(cv::vector<cv::Rect>& faces);    // Make face box size multiple of BLOCK_SIZE
    cv::Rect postProcessFaces(cv::vector<cv::Rect>& faces);
    int addToInternalFaceArray(cv::Rect currFace);
    void updateExistingFaceArray();
    cv::Rect getBestFaceFrmInternalArray();

    // Face tracker related
    cv::Mat extractWarpedFace(cv::Mat frameRGB, bool &prevFacePhasedOut);
    bool faceBoxLockedKLT;
	bool eyeBoxLockedKLT;
    cv::Mat prevFrameTracked;
    std::vector <cv::Point2f> pointsFaceBoxTransformed;
    std::vector <cv::Point2f> prevFeaturePoints;
    std::vector <cv::Point2f> currFeaturePoints;
    cv::Mat affInvCumulative;
    cv::Rect faceBoxFromFD;     // Last frame received from FD which is being tracked
    int framesInKLTLoop;
 
    // Core blink detection logic/ State machine related
    cv::Mat refStartFrame;
    MotionRegionOnSAD *motionStats;
    BLINK_STATS *blinkStats;
    bool isReset;         // whether state machine is reset
    int framesInCurrState;
    int prevMotion;
    int count;
    int blinkState;
    int count1;
    int prevCount,prevcount1;
    int maxCount;
    void resetBlinkStates();
    int StateMachine(cv::Rect faceRegion);
    int doMotionEstimation(cv::Mat newFrame, cv::Mat oldFrame, cv::Rect faceRegion, int Index, double &thres1, double minEyeBlocks, int &motion, double *extraOutput = nullptr);
	void computeAndDumpHOG(cv::Mat image, int label, int frameCount);

	// ML
	BlinkDetectorML *blinkSVM;
	int BlinkDetectionMLPredict(cv::Rect faceRegion);
	int BlinkDetectionMLSaveFeaturePoints(cv::Rect faceRegion, int inpBlinkState);

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
    //cv::Rect faceBox;
    cv::Mat  matFaceBox;
}BlinkDetectorReturnType;


#endif
