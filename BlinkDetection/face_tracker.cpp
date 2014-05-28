#include "blink_detector.h"
#include "getAffineForPoints.h"


void getFeaturePoints(cv::Mat imgGray, std::vector<cv::Point2f> &featurePoints, cv::Rect roi)
{
  /// Detector parameters
  int ED_blockSize = 2;
  int ED_apertureSize = 3;
  double ED_k = 0.04;
  int ED_thresh = 120;
  int ED_max_thresh = 300;

  cv::Mat imgGrayRoi = imgGray(cv::Rect(roi));
  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros( imgGray.size(), CV_32FC1 );

  /// Detecting corners
  //cv::cornerHarris( frame2gray, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT );
  cv::cornerMinEigenVal(imgGrayRoi, dst, ED_blockSize, ED_apertureSize, cv::BORDER_DEFAULT);

  /// Normalizing
  cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
  cv::convertScaleAbs( dst_norm, dst_norm_scaled );

  /// Drawing a circle around corners
  for( int j = 0; j < dst_norm.rows ; j++ )
  {
    for( int i = 0; i < dst_norm.cols; i++ )
    {
      if( (int) dst_norm.at<float>(j,i) > ED_thresh )
      {
        featurePoints.push_back(cv::Point2f(i+roi.x , j+roi.y));
      }
    }
  }

}

void getFeaturePoints2(cv::Mat imgGray, std::vector<cv::Point2f> &outFeaturePoints, cv::Rect roi)
{
  cv::Mat imgGrayRoi = imgGray(roi);
  int MAX_NUM_CORNERS= 100;
  double qualityLevel = 0.01;   // Parameter characterizing the minimal accepted quality of image corners
  double minDistance = 12;  // Minimum possible Euclidean distance between the returned corners
  int blockSize = 3;    // Size of an average block for computing a derivative covariation matrix over each pixel neighborhood
  bool useHarrisDetector = false;
  double k = 0.04;    // Free parameter of the Harris detector
  cv::TermCriteria termcrit(CV_TERMCRIT_ITER|
                      cv::TermCriteria::EPS, 20, 0.3);
  cv::Size subPixWinSize(10,10);
  cv::Size winSize(21,21);

  //Feature detection is performed here...
  goodFeaturesToTrack(imgGrayRoi, outFeaturePoints, MAX_NUM_CORNERS,
                      qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);
  cornerSubPix(imgGrayRoi, outFeaturePoints, subPixWinSize,
                cv::Size(-1,-1), termcrit);

  for (int i=0; i<outFeaturePoints.size(); i++)
  {
    outFeaturePoints[i].x +=  roi.x;
    outFeaturePoints[i].y +=  roi.y;
  }

  return;
}



cv::Mat BlinkDetector::extractWarpedFace(cv::Mat frameRGB, bool &flagNewFace)
{
    cv::Mat retMat = cv::Mat();
    flagNewFace = false;
    // First do face detection and lock onto a face

    if (faceBoxLockedKLT)
    {
        // Check for reset conditions;
        bool resetKLT = isFeatureResetRequired(frameRGB, pointsFaceBoxTransformed, prevFeaturePoints);    // Verify if minimum number of points lie on face
    
        if (!resetKLT)
        {
            framesInKLTLoop++;
            cv::Mat grayframe2;
            cv::cvtColor(frameRGB, grayframe2, CV_BGR2GRAY);

            std::vector<unsigned char> status;    // 1 if the flow for the corresponding features has been found, otherwise, it is set to 0
            std::vector<float> errVector;        // output vector of errors
            const cv::Size searchWinSize = cv::Size(21,21);     // size of the search window at each pyramid level.
            const int maxPyramidLevels= 3;
            //cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER, 30, (0.01));
            cv::TermCriteria termCriteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, (0.01));
            int flags = 0;
            double minEigThreshold = 0.0001;

            currFeaturePoints = std::vector<cv::Point2f>();

            cv::calcOpticalFlowPyrLK(prevFrameTracked, grayframe2, prevFeaturePoints, currFeaturePoints, status, errVector,
                                    searchWinSize, maxPyramidLevels, termCriteria, flags,minEigThreshold);

            std::vector<unsigned int> inliers;
            cv::Mat aff;

            getAffineForPoints(prevFeaturePoints, currFeaturePoints, inliers, aff);
            //getAffineForPointCloud(prevFeaturePoints, currFeaturePoints, inliers, aff);

            transformPointsForward(pointsFaceBoxTransformed, pointsFaceBoxTransformed, aff);

            /*
            for (int i=0; i< status.size(); i++)
            {
              if (inliers[i] == 1)
                circle( frameRGB, cv::Point2f( currFeaturePoints[i].x , currFeaturePoints[i].y ), 1,  cv::Scalar(0,255,0), 2, 8, 0 );
              else
                circle( frameRGB, cv::Point2f( currFeaturePoints[i].x , currFeaturePoints[i].y ), 1,  cv::Scalar(0,0,255), 2, 8, 0 );
            }
            */

            cv::Mat affInv;

            // cv::invertAffineTransform takes only matrix of size 2x3 as input
            cv::invertAffineTransform(aff(cv::Rect(0,0,3,2)), affInv);

            cv::Mat affInvTemp = cv::Mat::zeros(cv::Size(3,3), CV_32F);
            cv::Mat affInvTempSection = affInvTemp(cv::Rect(0,0,3,2));
            affInvTempSection = affInvTempSection + affInv;
            affInvCumulative = affInvTemp * affInvCumulative;

            int borderMode = cv::BORDER_CONSTANT;   // = 0. 
            // BORDER_TRANSPARENT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101= BORDER_DEFAULT, BORDER_ISOLATED
            cv::Mat frameOut;
            cv::warpAffine(frameRGB.clone(), frameOut, affInvCumulative (cv::Rect(0,0,3,2)), frameRGB.size(), 1, borderMode);

            bool reInitOutOfBoundFB = false;
            cv::Rect faceBoxSection =  findFaceBox(frameOut, faceBoxFromFD, reInitOutOfBoundFB, affInvCumulative, pointsFaceBoxTransformed);
            if (!reInitOutOfBoundFB)
            {
                //cv::imshow( "Frames", frameOut(faceBoxSection) );
                retMat = frameOut(faceBoxSection).clone();      // clone is required since we would draw a pink box in the original frame.

                //cv::imshow("Face Region",retMat);
                //cv::waitKey(1);

                prevFrameTracked = grayframe2.clone();
                prevFeaturePoints.clear();

                for (int i=0; i< currFeaturePoints.size(); i++)
                {
                    if (inliers[i] == 1)
                    {
                        prevFeaturePoints.push_back(currFeaturePoints[i]);
                    }
                }

                drawFaceBox(frameRGB, pointsFaceBoxTransformed);
            }
            
            // Also reinit if the box gets overly distorted.
            if (prevFeaturePoints.size() < 4 || framesInKLTLoop >1000 || reInitOutOfBoundFB)
            {
                faceBoxLockedKLT = false;
            }
            // Update frame states for next frame
        }
        else
        {
            faceBoxLockedKLT = false;
        }

    }

    if (!faceBoxLockedKLT && ((mFrameNum & 0x03) == 0x00))		// Do face detection every fourth frame.
    {
        std::vector<cv::Rect> faces;
        cv::Rect currFaceFromFD;
        cv::Mat frame_gray;

        cv::cvtColor(frameRGB, frame_gray, CV_BGR2GRAY);

        //-- Detect faces
        face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

        updateFaceSizeForBlockProc(faces);

        prevFeaturePoints = std::vector<cv::Point2f>();
        currFaceFromFD = postProcessFaces(faces);

		//printf("Current face width= %d\n", currFaceFromFD.width);
		//std::cout << std::endl;
		//std::cout.flush();

        if (currFaceFromFD.width >0)
        {
            // Potential candidate for KLT based tracking.
            // Find number of features. If it is greater than 4, move on to tracking from FD
            getFeaturePoints2(frame_gray, prevFeaturePoints, currFaceFromFD);

            /*
            /// Drawing a circle around corners
            for (int i=0; i< prevFeaturePoints.size(); i++)
               circle( frameRGB, prevFeaturePoints[i], 1,  cv::Scalar(0,255,0), 2, 8, 0 );
            */
            if (prevFeaturePoints.size() < 4)
            {
                faceBoxLockedKLT = false;
            }
            else
            {
                // If we have a minimum of 6 features detected, only then use it for initialization.
                flagNewFace = true;
                faceBoxLockedKLT = true;
                prevFrameTracked = frame_gray;
                faceBoxFromFD = currFaceFromFD;
                resetFaceData();
                framesInKLTLoop = 0;
                affInvCumulative = cv::Mat::eye(cv::Size(3,3), CV_32F);
                retMat = frameRGB(faceBoxFromFD);         // Actually no need for this delay
            }

            if (faceBoxLockedKLT)
            {
                pointsFaceBoxTransformed = std::vector <cv::Point2f>();
                pointsFaceBoxTransformed.push_back(cv::Point2f((int)currFaceFromFD.x, (int)currFaceFromFD.y));
                pointsFaceBoxTransformed.push_back(cv::Point2f((int)(currFaceFromFD.x + currFaceFromFD.width - 1), (int)currFaceFromFD.y));
                pointsFaceBoxTransformed.push_back(cv::Point2f((int)(currFaceFromFD.x + currFaceFromFD.width - 1), (int)(currFaceFromFD.y + currFaceFromFD.height - 1)));
                pointsFaceBoxTransformed.push_back(cv::Point2f((int)currFaceFromFD.x, (int)(currFaceFromFD.y + currFaceFromFD.height - 1)));
            
                //cv::polylines(frame, pointsFaceBox, true, cv::Scalar(255,0,255),3);
                drawFaceBox(frameRGB, pointsFaceBoxTransformed);
            }
        }
    
    }
        


    return retMat;

}


