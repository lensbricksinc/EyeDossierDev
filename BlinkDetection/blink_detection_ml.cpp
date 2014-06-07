
#include "blink_detector.h"
#include "ml_core.h"

int BlinkDetector::BlinkDetectionMLPredict(cv::Rect faceRegion)
{
    isReset = false;
    int hasBlink=0;

    int row = faceRegion.width/BLOCK_SIZE;
    int numBlocks = row*row;
    double minEyeBlocks,minEyeBlocks1;
    minEyeBlocks = (double)(faceRegion.width*faceRegion.width*5/(100*100));
    double thres1;
    int motion;
    minEyeBlocks1 = (double)((7*minEyeBlocks)/10);

    double *extraOutput;
    extraOutput = new double[8 + 100];

    count1 = doMotionEstimation(currFrameInfo->frameFaceSec, prevFrameInfo->frameFaceSec, faceRegion, 0, thres1, minEyeBlocks, motion, extraOutput);
    //printf("blinkState= %d, count1= %d . FrameNum= %d motion = %d minEyeBlocks %lf \n", blinkState, count1, mFrameNum, motion, minEyeBlocks);
#if FEATURESET_INDEX == 0
	float featureVecCurr[BLINK_ML_FEATURES_PER_FRAME] = {((float)count1), (float) motion};
#elif FEATURESET_INDEX == 1
	float featureVecCurr[BLINK_ML_FEATURES_PER_FRAME] = {(float)count1};
	for (int i=0; i<8; i++)
		featureVecCurr[1+i] = (float)extraOutput[i];
#elif FEATURESET_INDEX == 2
	float featureVecCurr[BLINK_ML_FEATURES_PER_FRAME] = {(float)count1};
	for (int i=0; i<100; i++)
		featureVecCurr[1+i] = (float)extraOutput[i+8];
#endif
	blinkSVM->MLAddEntry(featureVecCurr);
	hasBlink = blinkSVM->MLPredict();

    if (hasBlink == -1)
        hasBlink = 0;

    delete[] extraOutput;
    extraOutput = nullptr;
	// Add to the feature vector list

	// Get the outcome from machine learning stuff

    return hasBlink;

}



int BlinkDetector::BlinkDetectionMLSaveFeaturePoints(cv::Rect faceRegion, int inpBlinkState)
{
    isReset = false;
    int hasBlink=0;

    int row = faceRegion.width/BLOCK_SIZE;
    int numBlocks = row*row;
    double minEyeBlocks,minEyeBlocks1;
    minEyeBlocks = (double)(faceRegion.width*faceRegion.width*5/(100*100));
    double thres1;
    int motion;
    minEyeBlocks1 = (double)((7*minEyeBlocks)/10);

    double *extraOutput;
    extraOutput = new double[8 + 100];

    count1 = doMotionEstimation(currFrameInfo->frameFaceSec, prevFrameInfo->frameFaceSec, faceRegion, 0, thres1, minEyeBlocks, motion, extraOutput);
    //printf("blinkState= %d, count1= %d . FrameNum= %d motion = %d minEyeBlocks %lf \n", blinkState, count1, mFrameNum, motion, minEyeBlocks);

#if FEATURESET_INDEX == 0
	float featureVecCurr[BLINK_ML_FEATURES_PER_FRAME] = {((float)count1), (float) motion};
#elif FEATURESET_INDEX == 1
	float featureVecCurr[BLINK_ML_FEATURES_PER_FRAME] = {(float)count1};
	for (int i=0; i<8; i++)
		featureVecCurr[1+i] = (float)extraOutput[i];
#elif FEATURESET_INDEX == 2
	float featureVecCurr[BLINK_ML_FEATURES_PER_FRAME] = {(float)count1};
	for (int i=0; i<100; i++)
		featureVecCurr[1+i] = (float)extraOutput[i+8];
#endif

	blinkSVM->MLAddEntry(featureVecCurr);
	blinkSVM->MLDumpFeaturesToFileForTraining(inpBlinkState);

    delete[] extraOutput;
    extraOutput = nullptr;
	// Add to the feature vector list

	// Get the outcome from machine learning stuff

    return hasBlink;

}


int BlinkDetector::BlinkDetectionMLTrain()
{
	blinkSVM->MLTrain();



	return 0;
};

