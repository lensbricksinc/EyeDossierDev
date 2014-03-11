#include "blink_detector.h"
#include "utilityFuncs.h"
#include "motionEstDS.h"


BlinkDetector::BlinkDetector()
{
    BlinkDetector("cascades\\haarcascade_frontalface_alt2.xml");
}


BlinkDetector::~BlinkDetector()
{
    if (blinkStats != NULL)
    {
        delete blinkStats;
        blinkStats = NULL;
    }
        
    if (prevFrameInfo != NULL)
    {
        prevFrameInfo->frame = NULL;    // Is this required for dereferencing??
        delete prevFrameInfo;
        prevFrameInfo = nullptr;
    }

    if (currFrameInfo != NULL)
    {
        currFrameInfo->frame = NULL;
        delete currFrameInfo;
        currFrameInfo = nullptr;
    }

    if (faceArray != NULL)
    {
        delete[] faceArray;
        faceArray = nullptr;
    }

    if (motionStats != NULL)
    {
        delete[] motionStats;
        motionStats = nullptr;
    }
    return;
}


BlinkDetector::BlinkDetector(cv::string face_cascade_file)
{
    prevFrameInfo = NULL;
    currFrameInfo = NULL;
    currBaseSizeBox = cv::Rect(-1,-1,0,0);
    countNoFace = 0;
    FrameNum= 0;
    prevFaceBox = cv::Rect(-1,-1,-1,-1);

    face_cascade_name = face_cascade_file;
    
    faceArray = new FaceTrackingInfo[5];
    isReset = false;
    motionStats = new MotionRegionOnSAD();
    resetBlinkStates();

    blinkStats = new BLINK_STATS();
	if (!face_cascade.load(face_cascade_name))
    {
        printf("Unable to load face cascade");
        return;
    };
}


BlinkDetectorReturnType BlinkDetector::blink_detect(cv::Mat frame)
{
    BlinkDetectorReturnType ret;

    if (prevFrameInfo != NULL)
    {
        prevFrameInfo->frame = NULL;
        delete prevFrameInfo;
        prevFrameInfo = nullptr;
    }

    if (currFrameInfo != NULL)
        prevFrameInfo = currFrameInfo;

    FrameNum++;
    
#if 0
    cv::String sFrameNum;
    sFrameNum = std::to_string(FrameNum);

    if (sFrameNum.size() == 1)
        sFrameNum = "000" + sFrameNum;
    else if (sFrameNum.size() == 2)
        sFrameNum = "00" + sFrameNum;
    else if (sFrameNum.size() == 3)
        sFrameNum = "0" + sFrameNum;

    cv::string fileName = "frame_dump\\frame_" + sFrameNum + ".png";
    cv::imwrite(fileName, frame);
#endif
    

    currFrameInfo = new FrameInfo();
    currFrameInfo->frame = frame.clone();
    currFrameInfo->faceDetected = false;
    currFrameInfo->faceRegion = cv::Rect(-1, -1, -1, -1);

    std::vector<cv::Rect> faces;
    cv::Rect currFace;
    cv::Mat frame_gray;

    //cvCreateMat(t_height,t_width,CV_8U);

    cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);


    /*
    //printf("Num Colour channels of RGB = %d. Bit depth = %d. Data type = %d\n", frame.channels(), frame.depth(), frame.type());

    cv::Mat frame_yuv;
    cv::cvtColor(frame, frame_yuv, CV_BGR2YUV);

    //printf("Num Colour channels of YUV = %d. Bit depth = %d. Data type = %d \n", frame_yuv.channels(), frame_yuv.depth(), frame_yuv.type());
    
    cv::Mat frame_ycbcr;
    std::vector<cv::Mat> yCbCrChannels(3);
    cv::cvtColor(frame, frame_ycbcr, CV_BGR2YCrCb);
    cv::split(frame_ycbcr, yCbCrChannels);

    //cv::imshow("Y channel", yCbCrChannels[0]);
    //cv::imshow("Cr channel", yCbCrChannels[1]);
    //cv::imshow("Cb channel", yCbCrChannels[2]);

    //cv::waitKey(1);

    printf("Num Colour channels of YCrCb = %d. Bit depth = %d. Data type = %d \n",
            frame_ycbcr.channels(), frame_ycbcr.depth(), frame_ycbcr.type());
    */
    

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

    updateFaceSizeForBlockProc(faces);
    currFace = postProcessFaces(faces);

    if (currFace.width != -1)
    {
        currFrameInfo->faceDetected = true;
        currFrameInfo->faceRegion = currFace;
    }

    cv::Rect roi = fObtainRoiUnion();

    bool faceBoxChanged = hasFaceBoxChanged(roi);

    ret.outState = OUTSTATE_BLINKDETECT_FRAME_IDLE;
    ret.faceBox = cv::Rect(-1, -1, -1, -1);
    if (roi.width > 0)
    {
        if (faceBoxChanged == false)
        {
            int state = -1;
            state = StateMachine(roi);
            if (state == 1)
            {
                ret.outState = OUTSTATE_BLINKDETECT_FRAME_BLINK;
                blinkStats->addEvent(BLINK_STATS::BLINK_STATS_EVENT::EVENT_BLINK_EYEBLINK);
            }
            else
            {
                ret.outState = OUTSTATE_BLINKDETECT_FRAME_NOBLINK;
                blinkStats->addEvent(BLINK_STATS::BLINK_STATS_EVENT::EVENT_BLINK_ANALYSING);
            }
        }
        else
        {
            resetBlinkStates();
            ret.outState = OUTSTATE_BLINKDETECT_FRAME_IN_RESET;
            
        }

        ret.faceBox = roi;
        /*
        // Draw rectangle only if processing is happening
        cv::Point lefttop(roi.x, roi.y);
        cv::Point rightbottom( (roi.x + roi.width), (roi.y + roi.height));
        cv::rectangle(frame, lefttop, rightbottom, cv::Scalar( 255, 0, 0 ));
        */
    }
    else
    {
        blinkStats->addEvent(BLINK_STATS::BLINK_STATS_EVENT::EVENT_BLINK_NODATA);
    }
    ret.frame = frame;

    return ret;
}


bool BlinkDetector::fHasMotion(cv::Rect roi)
{
    cv::Mat img1 = currFrameInfo->frame(roi);
    cv::Mat img2 = prevFrameInfo->frame(roi);
    cv::Mat diffImg;

    cv::absdiff(img1, img2, diffImg);

    cv::imshow("Diff Frame", diffImg);
    
    return false;
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

    /*
    faceRegion.x = 289;
    faceRegion.y = 175;
    faceRegion.width = 208;
    faceRegion.height = 208;
    */
    int row = faceRegion.width/BLOCK_SIZE;
    int numBlocks = row*row;
    double minEyeBlocks,minEyeBlocks1;
    minEyeBlocks = (double)(faceRegion.width*faceRegion.width*5/(100*100));
    double thres1;
	int motion;
	minEyeBlocks1 = (double)((7*minEyeBlocks)/10);
	if (blinkState == 0)
	{
		
		count1 = doMotionEstimation(currFrameInfo->frame, prevFrameInfo->frame, faceRegion, blinkState, thres1, minEyeBlocks, motion);
		printf("blinkState= %d, count1= %d . FrameNum= %d motion = %d minEyeBlocks %lf \n", blinkState, count1, FrameNum, motion, minEyeBlocks);
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
						refStartFrame = prevFrameInfo->frame.clone();
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
                count1 = doMotionEstimation(currFrameInfo->frame, refStartFrame, faceRegion, blinkState, thres1,minEyeBlocks, motion);
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


cv::Rect BlinkDetector::fObtainRoiUnion()
{
    cv::Rect rectangle(-1,-1,0,0);
    if (prevFrameInfo != NULL 
        && currFrameInfo != NULL
        && prevFrameInfo->faceDetected == true 
        && currFrameInfo->faceDetected == true)
    {
        rectangle = currFrameInfo->faceRegion;
    }

/*
   
    if (prevFrameInfo != NULL 
        && currFrameInfo != NULL
        && prevFrameInfo->faceDetected == true 
        && currFrameInfo->faceDetected == true)
    {
        int x1 = (prevFrameInfo->faceRegion.x > currFrameInfo->faceRegion.x) ? prevFrameInfo->faceRegion.x:currFrameInfo->faceRegion.x;
        int y1 = (prevFrameInfo->faceRegion.y > currFrameInfo->faceRegion.y) ? prevFrameInfo->faceRegion.y:currFrameInfo->faceRegion.y;

        int val1 = prevFrameInfo->faceRegion.x + prevFrameInfo->faceRegion.width;
        int val2 = currFrameInfo->faceRegion.x + currFrameInfo->faceRegion.width;
        int x2 = (val1 > val2) ? val1:val2;
        int width = x2-x1;
        width = ((width)>>1)<<1;    // force width to be even

        val1 = prevFrameInfo->faceRegion.y + prevFrameInfo->faceRegion.height;
        val2 = currFrameInfo->faceRegion.y + currFrameInfo->faceRegion.height;
        int y2 = (val1 > val2) ? val1:val2;
        int height = y2-y1;
        height = ((height+1)>>1)<<1;    // force height to be even

        rectangle = cv::Rect(x1,y1, width, height);
    }

    //if ( (rectangle.width & 0x0000001F) != 0)    // Make rectangle.width a multiple of 32 for motionEst
    if ( (rectangle.width % BLOCK_SIZE) != 0)    // Make rectangle.width a multiple of 32 for motionEst
    {
        //int extra = (rectangle.width & 0x0000001F);
        int extra = rectangle.width % BLOCK_SIZE;
        int extra1 = extra/2;

        rectangle.x = rectangle.x + extra1;
        rectangle.y = rectangle.y + extra1;
        rectangle.width = rectangle.width -extra;
        rectangle.height = rectangle.height -extra;
    }
    */

    /*
    // Post process the roi to ensure a consistent face width across frames
    if (currBaseSizeBox.width != rectangle.width)
    {
        if (((currBaseSizeBox.width - rectangle.width)> 40)
          ||((currBaseSizeBox.width - rectangle.width)< -10))
        {
            // Update the currBaseSizeBox with the size of the rectangle
            currBaseSizeBox = rectangle;
        }
        else
        {
            // Adjust the rectangle to make it's size consistent across frames
            int diff = (currBaseSizeBox.width - rectangle.width);
            int diff_2 = diff/2;
            // Add the diff on both sides of the rectangle
            rectangle.x = rectangle.x - diff_2;
            rectangle.y = rectangle.y - diff_2;   // Since face box is always a square
            rectangle.width = rectangle.width + diff;
            rectangle.height = rectangle.height + diff;
        }
    }
    */

    return rectangle;
    
};


void BlinkDetector::resetBlinkStates()
{
    if (isReset == false)
    {
        prevFaceBox = cv::Rect(-1, -1, -1, -1);
        motionStats->resetMotionStats();
        count= 0;
        blinkState= 0;
        count1= 0;
        prevCount= 0;
        framesInCurrState= 0;
        //prevFrame;    // Needs to be initialised
        refStartFrame = NULL;
        isReset = true;
        printf("\n++++++++++++++\n++++++++++++++\nSTATE MACHINE RESET\n++++++++++++++\n++++++++++++++\n");
    }
}


bool BlinkDetector::hasFaceBoxChanged(cv::Rect roi)
{
    bool val = false;
    if (roi.width == prevFaceBox.width && roi.height == prevFaceBox.height
        && roi.x == prevFaceBox.x && roi.y == prevFaceBox.y)
        val =  false;
    else
        val = true;
    prevFaceBox = roi;

    return val;
};

