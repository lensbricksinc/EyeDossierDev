#include "blink_detector.h"
#include "utilityFuncs.h"
#include "motionEstDS.h"

#define BLOCK_SIZE 8

BlinkDetector::BlinkDetector()
{
    BlinkDetector("cascades\\haarcascade_frontalface_alt2.xml");
}


BlinkDetector::~BlinkDetector()
{
    if (prevFrameInfo != NULL)
        prevFrameInfo->frame = NULL;    // Is this required for dereferencing??
        delete prevFrameInfo;

    if (currFrameInfo != NULL)
        currFrameInfo->frame = NULL;
        delete currFrameInfo;

    if (faceArray != NULL)
        delete[] faceArray;

    return;
}


BlinkDetector::BlinkDetector(cv::string face_cascade_file)
{
    prevFrameInfo = NULL;
    currFrameInfo = NULL;
    currBaseSizeBox = cv::Rect(-1,-1,0,0);
    countNoFace = 0;
    FrameNum= 0;

    face_cascade_name = face_cascade_file;
    if (!face_cascade.load(face_cascade_name))
    {
        printf("Unable to load face cascade");
        return;
    };
    faceArray = new FaceTrackingInfo[5];
}


cv::Mat BlinkDetector::blink_detect( cv::Mat frame)
{
    if (prevFrameInfo != NULL)
    {
        prevFrameInfo->frame = NULL;
        delete prevFrameInfo;
    }

    if (currFrameInfo != NULL)
        prevFrameInfo = currFrameInfo;

    FrameNum++;
    cv::String sFrameNum;
    sFrameNum = std::to_string(FrameNum);

    if (sFrameNum.size() == 1)
        sFrameNum = "000" + sFrameNum;
    else if (sFrameNum.size() == 2)
        sFrameNum = "00" + sFrameNum;
    else if (sFrameNum.size() == 3)
        sFrameNum = "0" + sFrameNum;

    //cv::string fileName = "frame_dump\\frame_" + sFrameNum + ".png";
    //cv::imwrite(fileName, frame);
    

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

    currFace = postProcessFaces(faces);

    if (currFace.width != -1)
    {
        currFrameInfo->faceDetected = true;
        currFrameInfo->faceRegion = currFace;

        // Draw rectangle only for largest face
        cv::Point lefttop(currFace.x, currFace.y);
        cv::Point rightbottom( (currFace.x + currFace.width), (currFace.y + currFace.height));
        cv::rectangle(frame, lefttop, rightbottom, cv::Scalar( 255, 0, 255 ));
    }

    cv::Rect roi = fObtainRoiUnion();

    if (roi.width > 0)
    {
        StateMachine(roi);
    }
    else
    {
        resetBlinkStates();
    }

    return frame;
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


int BlinkDetector::doMotionEstimation(cv::Mat newFrame, cv::Mat oldFrame, cv::Rect faceRegion, int Index, double &thres1)
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

    cv::Mat thres_img;
    double high_thres = cv::threshold( channelsRefImage[0], thres_img, 0, 255, CV_THRESH_BINARY+CV_THRESH_OTSU );

    /// Reduce noise with a kernel 3x3
    cv::Mat detected_edges;
    cv::blur( channelsRefImage[0], detected_edges, cv::Size(3,3) );

    int lowThreshold = (int)(high_thres/2);
    int ratio= 3;
    int kernel_size = 3;
    cv::Mat dst;
    /// Canny detector
    cv::Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

    /// Using Canny's output as a mask, we display our result
    dst = cv::Scalar::all(0);

    channelsCurrImage[0].copyTo( dst, detected_edges);
    //cv::imshow( "Canny Edges" , dst );
    //cv::waitKey(0);

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
    for (int i=0; i<4; i++)
        motionVect[i] = new double[lenVectors];
     
    double DScomputations;
    motionEstDS(A, B, BLOCK_SIZE, 2*BLOCK_SIZE, motionVect, DScomputations);

    if (Index == 0)
    {
        double variance1;
        double variance2;

        cv::Scalar sum1 = cv::sum(channelsCurrImage[0]);
        variance1 = ((double)sum1.val[0,0])/(double)(numBlocks*BLOCK_SIZE*BLOCK_SIZE);

        cv::Mat Aabs = cv::abs(A);
        cv::Scalar sum2 = cv::sum(Aabs);
        variance2 = (double)sum2.val[0,0]/(double)(numBlocks*BLOCK_SIZE*BLOCK_SIZE);

        if ((variance1 < 80) && (variance2 < 20))
            thres1 = 2;
        else
            thres1 = 3;
    }
   thres1= 4;

    /*
    cv::imshow("oldFrame", oldFrame);
    cv::waitKey(1);
    cv::imshow("newFrame", oldFrame);
    cv::waitKey(1);
    */
    /*
    if (Index == 1)
    {
    cv::imwrite("C:\\Users\\Kumar\\Desktop\\oldFrame.png", oldFrame);
    cv::imwrite("C:\\Users\\Kumar\\Desktop\\newFrame.png", newFrame);
    }
    */

    
    int localCount=0;
    for (int i=2; i< (int)((row+1)/2 - 2); i++)
    {
        for (int j= 2; j<=row-2; j++)
        {
            if (motionVect[2][i*row+j] > thres1)
            {
                 localCount = localCount+1;
            }
        }
    }

    for (int i=0; i<4; i++)
        delete[] motionVect[i];
         
    //free(motionVect);
    
    return localCount;

}

void BlinkDetector::StateMachine(cv::Rect faceRegion)
{
    int row = faceRegion.width/BLOCK_SIZE;
    int numBlocks = row*row;
    int minEyeBlocks;
    minEyeBlocks = (int)((float)(10*numBlocks/800) + 0.5);
    int minEyeBlocks1 = minEyeBlocks;
    if (minEyeBlocks < 10)
        minEyeBlocks = 10;
    int prevcount1;
    double thres1;
    

    count1 = doMotionEstimation(currFrameInfo->frame, prevFrameInfo->frame, faceRegion,0, thres1);
    printf("blinkState= %d, count1= %d . FrameNum= %d\n",blinkState, count1, FrameNum);
    /*
    if (refStartFrame.data != NULL)
    {
        
        cv::String fileName = "C:\\Users\\Kumar\\Desktop\\frame"+ std::to_string(FrameNum)+".png";
        cv::imwrite(fileName, refStartFrame);
    }
    */
    switch (blinkState)
    {
            case 0:
                if ((count1 > minEyeBlocks1) || (count1 + prevCount >= (2*minEyeBlocks)) )
                {
                    blinkState = 1;
                    //prevCount  = (prevCount + count1)/4;
                    if (count1 > minEyeBlocks)
                        prevCount  = count1;
                    
                    if (prevCount < 2)
                        prevCount = 2;

                    framesInCurrState = 0;
                    refStartFrame= prevFrameInfo->frame.clone();
                }
                else
                {
                    prevCount = count1;
                }
                break;

            case 1:
                framesInCurrState = framesInCurrState + 1;
                if ((count1 < (prevCount+1)) && (abs(prevCount-count1) >= minEyeBlocks))
                {
                    blinkState = 3;
                    framesInCurrState = 0;
                    prevCount = count1;
                }
                else
                {
                    if (count1 > minEyeBlocks)
                    {
                        prevCount  = count1;
                        framesInCurrState = 0;
                    }
                    if (framesInCurrState > 2)
                    {
                        if (count1 > minEyeBlocks)
                            blinkState = 1;
                        else
                            blinkState = 0;
                    }
                }
                break;

            case 3:
                framesInCurrState = framesInCurrState + 1;
                if (count1 + prevCount>= (minEyeBlocks))
                {
                    //eye open starts
                    blinkState = 4;
                    framesInCurrState = 0;
                }
                else
                {
                    prevCount = count1;
                    if (framesInCurrState > 2)
                   {
                        blinkState = 0;
                        prevCount = 0;
                    }
                }
                break;

            case 4:
            {
                framesInCurrState = framesInCurrState+1;
                prevcount1 = count1;
                //%do motion estimation
                count1 = doMotionEstimation(currFrameInfo->frame, refStartFrame, faceRegion,1, thres1);
                printf("blinkState= %d, count1= %d . minEyeBlocks= %d\n",blinkState, count1, minEyeBlocks);
                if ((count1 <= 5*minEyeBlocks && prevcount1 <= (prevCount+1)) || count1 <= minEyeBlocks)
                {
                    printf("***********BLINK DETECTED************");
                    blinkState = 1;
                }

                if (blinkState == 4 && framesInCurrState > 3)
                {
                    blinkState = 0;
                }
                prevCount = prevcount1;
            }
                break;

    }

}


cv::Rect BlinkDetector::fObtainRoiUnion()
{
    cv::Rect rectangle(-1,-1,0,0);
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
    count= 0;
    blinkState= 0;
    count1= 0;
    prevCount= 0;
    framesInCurrState= 0;
    //prevFrame;    // Needs to be initialised
    refStartFrame = NULL;
}
