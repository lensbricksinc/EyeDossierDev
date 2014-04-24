#include "blink_detector.h"
#include "utilityFuncs.h"
#include "motionEstDS.h"


BlinkDetector::BlinkDetector()
{
    BlinkDetector("haarcascade_frontalface_alt2.xml","cascades\\haarcascade_frontalface_alt2.xml");
}


BlinkDetector::~BlinkDetector()
{
    if (blinkStats != NULL)
    {
        delete blinkStats;
        blinkStats = NULL;
    }
        
    if (motionStats != NULL)
    {
        delete motionStats;
        motionStats = nullptr;
    }

    if (faceArray != NULL)
    {
        delete[] faceArray;
        faceArray = nullptr;
    }

    if (currFrameInfo != NULL)
    {
        currFrameInfo->frameFullSize = NULL;
        currFrameInfo->frameFaceSec = NULL;
        delete currFrameInfo;
        currFrameInfo = nullptr;
    }
    
    if (prevFrameInfo != NULL)
    {
        prevFrameInfo->frameFullSize = NULL;    // Is this required for dereferencing??
        prevFrameInfo->frameFaceSec = NULL;
        delete prevFrameInfo;
        prevFrameInfo = nullptr;
    }

    return;
}


BlinkDetector::BlinkDetector(cv::string face_cascade_file1,  cv::string face_cascade_file2)
{
    prevFrameInfo = NULL;
    currFrameInfo = NULL;
    mFrameNum= 0;
    faceBoxFromFD = cv::Rect(-1,-1,-1,-1);
    
    
    faceArray = new FaceTrackingInfo[5];
    isReset = false;
    motionStats = new MotionRegionOnSAD();
    resetBlinkStates();

    blinkStats = NULL;
    //blinkStats = new BLINK_STATS();
    face_cascade_name = "--";
	if (!face_cascade.load(face_cascade_file1))
    {
        if (!face_cascade.load(face_cascade_file2))
        {
            printf("Unable to load face cascade");
            goto EXIT;
        }
        else
        {
            face_cascade_name = face_cascade_file2;
        }
    }
    else
    {
        face_cascade_name = face_cascade_file1;
    };
    faceBoxLockedKLT = false;
    prevFrameTracked = cv::Mat();

EXIT:
    return;
}


BlinkDetectorReturnType BlinkDetector::blink_detect(cv::Mat frame)
{
    BlinkDetectorReturnType ret;

    if (prevFrameInfo != NULL)
    {
        prevFrameInfo->frameFullSize = NULL;
        prevFrameInfo->frameFaceSec = NULL;
        delete prevFrameInfo;
        prevFrameInfo = nullptr;
    }

    if (currFrameInfo != NULL)
        prevFrameInfo = currFrameInfo;

    mFrameNum++;

    currFrameInfo = new FrameInfo();
    currFrameInfo->frameFullSize = frame.clone();
    currFrameInfo->faceDetected = false;
    currFrameInfo->faceRegion = cv::Rect(-1, -1, -1, -1);

    bool resetFlag = false;
    cv::Mat faceWarpedImage = extractWarpedFace(frame, resetFlag);

    if (faceWarpedImage.rows > 0)
    {
        currFrameInfo->faceDetected = true;
        currFrameInfo->frameFaceSec = faceWarpedImage;

        if (resetFlag == true)
        {
            resetBlinkStates();
            ret.outState = OUTSTATE_BLINKDETECT_FRAME_IN_RESET;
        }
        else
        {
            int state = -1;
            state = StateMachine(cv::Rect(0, 0, faceWarpedImage.rows, faceWarpedImage.cols) );
            if (state == 1)
            {
                ret.outState = OUTSTATE_BLINKDETECT_FRAME_BLINK;
                //blinkStats->addEvent(BLINK_STATS::BLINK_STATS_EVENT::EVENT_BLINK_EYEBLINK);
            }
            else
            {
                ret.outState = OUTSTATE_BLINKDETECT_FRAME_NOBLINK;
                //blinkStats->addEvent(BLINK_STATS::BLINK_STATS_EVENT::EVENT_BLINK_ANALYSING);
            }
        }
        ret.matFaceBox = faceWarpedImage;
    }
    else
    {
        ret.outState = OUTSTATE_BLINKDETECT_FRAME_IDLE;
        ret.matFaceBox = cv::Mat();
        //blinkStats->addEvent(BLINK_STATS::BLINK_STATS_EVENT::EVENT_BLINK_NODATA);
    }
    ret.frame = frame;

    return ret;
}


int BlinkDetector::doMotionEstimation(cv::Mat newFrame, cv::Mat oldFrame, cv::Rect faceRegion, int Index, double &thres1,double minEyeBlocks, int &motion)
{
    int row = faceRegion.width/BLOCK_SIZE;
    int numBlocks = row*row;
    

    //double thres1;

    cv::Mat tempImage = newFrame(faceRegion);
    cv::Mat currImage;
    //cv::cvtColor(tempImage, currImage, CV_BGR2YCrCb );
    cv::Mat t1,t2,t3;
    matlabRGB2YCrCb(tempImage, t1,t2,t3, currImage);
    std::vector<cv::Mat> channelsCurrImage(3);
    cv::split(currImage, channelsCurrImage);

    tempImage = oldFrame(faceRegion);
    cv::Mat refImage1;
    //cv::cvtColor(tempImage, refImage1, CV_BGR2YCrCb );
    matlabRGB2YCrCb(tempImage, t1,t2,t3, refImage1);
    std::vector<cv::Mat> channelsRefImage(3);
    cv::split(refImage1, channelsRefImage);

    cv::Scalar meanCurrImage = cv::mean(channelsCurrImage[0]);
    cv::Scalar meanRefImage = cv::mean(channelsRefImage[0]);

    //Implementing cv::Mat A = channelsCurrImage[0] - meanCurrImage[0,0]; 
    cv::Mat srcMat = channelsCurrImage[0];
    cv::Mat A = cv::Mat(srcMat.rows,srcMat.cols,CV_32F);
    for (int a=0; a<srcMat.rows; a++)
    {
        for (int b=0; b<srcMat.cols; b++)
        {
            A.at<float>(a,b) = ((float) srcMat.at<uchar>(a,b)) - (float)meanCurrImage[0,0];
        }
    }

    // Implementing cv::Mat B = channelsRefImage[0] - meanRefImage[0,0];
    cv::Mat srcMat2 = channelsRefImage[0];
    cv::Mat B = cv::Mat(srcMat2.rows, srcMat2.cols,CV_32F);
    for (int a=0; a< srcMat2.rows; a++)
    {
        for (int b=0; b< srcMat2.cols; b++)
        {
            B.at<float>(a,b) = ((float) srcMat2.at<uchar>(a,b)) - (float)meanRefImage[0,0];
        }
    }

    
    double *motionVect[4];
    int imgRows = A.rows;
    int imgCols = A.cols;
    int lenVectors = imgRows*imgCols/((BLOCK_SIZE*BLOCK_SIZE));
	int **mask = new int*[imgRows];
	
	for (int i = 0; i < imgRows; i++)
	{
		mask[i] = new int[imgCols] ();
	}

    for (int i=0; i<4; i++)
        motionVect[i] = new double[lenVectors];
     
    double DScomputations;
    motionEstDS(A, B, BLOCK_SIZE, 2*BLOCK_SIZE, motionVect, DScomputations);
    
    int *motionMask = new int[lenVectors]();

    motionStats->updateStats(motionVect, lenVectors, 1.9, motionMask, Index);
    
    int localCount = 0;
    for (int i = 2; i <= ((row+1) / 2 - 2); i++)
    {
        for (int j = 2; j <= row - 2; j++)
        {
            int ind = i*row + j - 1;
            if (motionMask[ind] == 1)
			{
				mask[i][j] = 255;
                localCount++;
			}
        }
    }
    motion = motionStats->analyzeMotion(mask,row,minEyeBlocks,localCount,blinkState,row,row);
	
    delete[] motionMask;
    motionMask = nullptr;

    for (int i=0; i<4; i++)
    {
        delete[] motionVect[i];
        motionVect[i] = nullptr;
    }
	for (int i = 0; i < imgRows; i++)
	{
		delete[] mask[i]; 
	}
	delete[] mask;
    return localCount;

}

int BlinkDetector::StateMachine(cv::Rect faceRegion)
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
	if (blinkState == 0)
	{
		count1 = doMotionEstimation(currFrameInfo->frameFaceSec, prevFrameInfo->frameFaceSec, faceRegion, blinkState, thres1, minEyeBlocks, motion);
		printf("blinkState= %d, count1= %d . FrameNum= %d motion = %d minEyeBlocks %lf \n", blinkState, count1, mFrameNum, motion, minEyeBlocks);
	}
    switch (blinkState)
    {
            case 0:
				if (count1 > minEyeBlocks1 && count1 < 3*minEyeBlocks)
				{
					if (motion == 1)
					{
						prevMotion = 1;
						blinkState = 4;
						prevCount = count1;
						framesInCurrState = 0;
						refStartFrame = prevFrameInfo->frameFaceSec.clone();
						maxCount = prevCount;
					}
					else
					{
						prevCount = count1;
					}
				}
                else
                {
                    prevCount = count1;
                }
                break;

            case 4:
            {
                framesInCurrState = framesInCurrState+1;
                prevcount1 = count1;
                //%do motion estimation
                count1 = doMotionEstimation(currFrameInfo->frameFaceSec, refStartFrame, faceRegion, blinkState, thres1,minEyeBlocks, motion);
                printf("blinkState= %d, count1= %d . minEyeBlocks= %lf motion %d\n",blinkState, count1, minEyeBlocks,motion);
				if (count1 > maxCount)
					maxCount = count1;
				/*if (prevMotion == 0 && motion == 1)
				{
					count1 = 2000; //cannot be blink
				}*/ //for later
				//if ((count1 < maxCount )&&(((count1 <=  minEyeBlocks && motion == 0) || ((double)count1 <= (double)(minEyeBlocks) && motion == 1)) && framesInCurrState > 1) || (framesInCurrState == 1 && (double)count1 <= (double)(minEyeBlocks/2) && prevCount > minEyeBlocks && motion == 0 && count1 < prevCount))
				if ((count1 < maxCount) && ((count1 <= 2 * minEyeBlocks && motion == 0) && count1 < prevCount && framesInCurrState > 1) || (framesInCurrState == 1 && (double)count1 <= (double)(minEyeBlocks / 2) && prevCount > minEyeBlocks && motion == 0 && count1 < prevCount))
				//if ((count1 < maxCount) && count1 <= 2*minEyeBlocks  )
			    {
                    printf( "***********BLINK DETECTED************\n" );
                    hasBlink = 1;
                    blinkState = 0;
					prevCount = 0;
                }
				else
				{
					prevMotion = motion;
					if (framesInCurrState == 1 && (count1 < minEyeBlocks / 2 || count1 > prevCount || count1 > minEyeBlocks) && motion == 0)
					{
						blinkState = 0;
						prevCount = 0;
					}
					else if (count1 > 4 * minEyeBlocks)
					{
						blinkState = 0;
					}
				}
                if (blinkState == 4 && framesInCurrState > 7)
                {
                    blinkState = 0;
					prevCount = 0;
                }
             }
                break;
    }

    return hasBlink;
}



void BlinkDetector::resetBlinkStates()
{
    if (isReset == false)
    {
        //faceBoxFromFD = cv::Rect(-1, -1, -1, -1);
        motionStats->resetMotionStats();
        count= 0;
        blinkState= 0;
        count1= 0;
        prevCount= 0;
        framesInCurrState= 0;
        refStartFrame = NULL;
        isReset = true;
        printf("\n++++++++++++++\n++++++++++++++\nSTATE MACHINE RESET\n++++++++++++++\n++++++++++++++\n");
    }
}


