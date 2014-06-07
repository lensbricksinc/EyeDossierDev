#include "blink_detector.h"
#include "utilityFuncs.h"
#include "motionEstDS.h"
#include <opencv2\opencv.hpp>
#include <fstream>


BlinkDetector::BlinkDetector()
{
	cv::string face_cascade_name1 = "cascades\\haarcascade_frontalface_alt2.xml";
	cv::string eye_cascade_name = "cascades\\haarcascade_mcs_lefteye.xml";// = "cascades\\haarcascade_eye_tree_eyeglasses.xml";
	
	BlinkDetector::BlinkDetector(face_cascade_name1, eye_cascade_name);
/*
	cv::string version = VERSION;
	cv::string name = "chintak";
	cv::string dumpHOGFileName = "C:\\Users\\chintak\\Desktop\\testcases\\dataset\\"+name+"\\"+name+"_eye.dat";
	cv::string blinkMarkFileName = "C:\\Users\\chintak\\Desktop\\testcases\\dataset\\"+name+"\\"+name+".csv";
	cv::string readSVMModelName = "C:\\Users\\chintak\\Downloads\\svm_light_windows32\\test\\svm_model"+version+".xml";
	cv::string dumpResSVMName1 = "C:\\Users\\chintak\\Desktop\\testcases\\dataset\\"+name+"\\"+name+"01\\";
	std::string modelName = "C:\\Users\\chintak\\Downloads\\libsvm-3.18\\windows\\test\\model"+version+".dat";
	BlinkDetector::BlinkDetector(face_cascade_name1, eye_cascade_name, modelName,
		dumpHOGFileName, blinkMarkFileName, readSVMModelName, dumpResSVMName1);*/
}


BlinkDetector::~BlinkDetector()
{
	if (blinkSVM != NULL)
	{
		delete blinkSVM;
	}
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
#ifdef DUMP_HOG
	dumpHOGFile.close();
	blinkMarkFileRead.close();
#endif

	return;
}

//BlinkDetector::BlinkDetector(cv::string face_cascade_file1,  
//							 cv::string eye_cascade_file, 
//							 cv::string modelName,
//							 cv::string dumpHOGFileName,
//							 cv::string blinkCountReadName,
//							 cv::string readSVMModelName,
//							 cv::string dumpResSVMName1)
BlinkDetector::BlinkDetector(cv::string face_cascade_file1,  
							 cv::string eye_cascade_file)
{
    prevFrameInfo = NULL;
    currFrameInfo = NULL;
    mFrameNum= 0;
    faceBoxFromFD = cv::Rect(-1,-1,-1,-1);
    
    faceArray = new FaceTrackingInfo[5];
    isReset = false;
    motionStats = new MotionRegionOnSAD();

	blinkStats = NULL;
    //blinkStats = new BLINK_STATS();
    face_cascade_name = "--";
    if (!face_cascade.load(face_cascade_file1))
    {
        printf("Unable to load face cascade");
        return;
    }
    if (!eye_cascade.load(eye_cascade_file))
    {
        printf("Unable to load eye cascade");
        return;
    }

#ifdef TEST_SVM
	std::string modelName = "svm_model.dat";
	model = svm_load_model(modelName.c_str());
#endif

	closedEyeCount = 0;

    faceBoxLockedKLT = false;
    prevFrameTracked = cv::Mat();

#ifdef DUMP_HOG
	blinkMarkFileRead.open(blinkCountReadName);
	if(!blinkMarkFileRead.is_open()) {
		std::cout << "Cannot open blink mark CSV file\n";
		cv::waitKey(0);
	}
	std::string line;
	std::getline(blinkMarkFileRead, line);  // discard the first line
	std::getline(blinkMarkFileRead, line);
	blinkFrameCount = std::stoi(line);

	/*dumpHOGFile.open(dumpHOGFileName);
	if(!dumpHOGFile.is_open()) {
		std::cout << "Cannot write HOG feature file\n";
		cv::waitKey(0);
	}*/
#endif

#ifdef TEST_AND_DUMP_SVM_RESULTS
	accuracy = 0.0;
	totalNumSamples = 0;
	dumpResSVMName = dumpResSVMName1;
#endif

#ifdef USE_HOG_DETECTOR
	cv::Size win_size(48, 32);
		cv::Size block_size(12, 8);
		cv::Size block_stride(6, 4);
		cv::Size cell_size(6, 4);
		int nbins = 9;
		double win_sigma = -1; // DEFAULT_WIN_SIGMA
		double threshold_L2hys = 0.2;
		bool gamma_correction = false;
		int nlevels = 16; // DEFAULT_NLEVELS
		hog = cv::HOGDescriptor(win_size, block_size, block_stride, 
						cell_size, nbins, win_sigma, 
						threshold_L2hys, gamma_correction, nlevels);
#endif

	blinkSVM = new BlinkDetectorML(BLINK_ML_FEATURES_PER_FRAME);

	resetBlinkStates();
EXIT:
    return;
}

#ifdef DUMP_HOG
void BlinkDetector::computeAndDumpHOG(cv::Mat image, int label, int frameCount){
	std::stringstream fileName;
	if (label==1) {
		dumpHOGFile << "+1 ";
		//fileName << "C:\\Users\\chintak\\Desktop\\testcases\\dataset\\chintak\\chintak_" << frameCount << "_pos.jpg";
	}
	else {
		dumpHOGFile << "+1 ";
		//fileName << "C:\\Users\\chintak\\Desktop\\testcases\\dataset\\chintak\\chintak_" << frameCount << "_neg.jpg";
	}

	cv::imwrite(fileName.str(), image);

	std::vector<float> des = computeHOG(image, true);
	for (int i = 0; i < des.size(); i++)
	{
		if (des[i] > 0.000001)
		{
			dumpHOGFile << i+1 << ":" << des[i] << " ";
		}
	}
	dumpHOGFile << "# " << frameCount << std::endl;
}
#endif


BlinkDetectorReturnType BlinkDetector::blink_detect(cv::Mat frame, int blinkStateFromFile)
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
	cv::Mat faceWarped_gray;
	cv::vector<cv::Rect> curEyes, detEyes;
	cv::Rect eyeCheck;
	/*cv::string name = "chintak";
	cv::String eyename = "C:\\Users\\chintak\\Desktop\\testcases\\dataset\\"+name+"\\";*/
	//cv::FileStorage eyeCoordFile("C:\\Users\\chintak\\Desktop\\testcases\\dataset\\chintak\\chintak_eye.xml", cv::FileStorage::WRITE);

    if (faceWarpedImage.rows > 0)
    {
#ifdef DUMP_HOG
		if(mFrameNum > blinkFrameCount+10) {
			std::string line;
			std::getline(blinkMarkFileRead, line);
			if(line.size() == 0) goto EXIT;
			blinkFrameCount = std::stoi(line);
		}
		if (mFrameNum < blinkFrameCount-10) goto EXIT;
#endif

		// Eye detection part
		cv::cvtColor(faceWarpedImage, faceWarped_gray, CV_BGR2GRAY);
		//cv::equalizeHist( faceWarped_gray, faceWarped_gray );
		cv::Rect roi(0, 0, faceWarpedImage.cols, faceWarpedImage.rows / 2 + faceWarpedImage.rows*.2);

#ifdef USE_HOG_DETECTOR
		std::vector<cv::Rect> loc;
		
		hog.detectMultiScale(faceWarpedImage(roi), loc, -bias, cv::Size(24, 16));

		if(loc.size() != 0) {
			for (int i = 0; i < loc.size(); i++)
			{
				cv::rectangle(faceWarpedImage, loc[i], cv::Scalar(255, 0, 255), 2);
			}
		}
		goto EXIT;
#endif

		eye_cascade.detectMultiScale(faceWarped_gray(roi), detEyes, 
			1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

		if(prevEyes.size() <= 1)
			curEyes = detEyes;
		else {
			curEyes.clear();
			for (int i = 0; i < detEyes.size(); i++)
			{
				eyeCheck = detEyes[i];
				for (int j = 0; j < prevEyes.size(); j++)
				{
					if (std::abs(eyeCheck.x - prevEyes[j].x) < 10)
						if (std::abs(eyeCheck.y - prevEyes[j].y) < 10) 
							if (std::abs(eyeCheck.width - prevEyes[j].width) < 20)
								if (std::abs(eyeCheck.height - prevEyes[j].height) < 20) {
									curEyes.push_back(eyeCheck);
									break;
								}
				}
			}
		}
		prevEyes = curEyes;
		if(curEyes.size() > 0) {
			
			//int c = cv::waitKey(0);

			for(int j=0; j < curEyes.size(); j++) {
#ifdef TEST_SVM
				std::vector<float> des = computeHOG(faceWarpedImage(curEyes[j]), false);
				struct svm_node * x = (struct svm_node *) malloc(1765*sizeof(struct svm_node));
				for (int i = 0; i < 1765; i++)
				{
					x[i].index = i;
					x[i].value = des[i];
				}

				double y = svm_predict(model, x);
				if(y>0)
					std::cout << "Eye state: Closed" << std::endl;
				else
					std::cout << "Eye state: Open" << std::endl;

				if(y > 0)
					closedEyeCount += 1;
				else
					closedEyeCount = 0;

				bool blink;

				if(closedEyeCount >= CONSECUTIVE_CLOSED_EYE_FOR_BLINK) {
					blink = true;
					closedEyeCount = 0;
				}

				if (blink)
					ret.outState = OUTSTATE_BLINKDETECT_FRAME_BLINK;
				else
					ret.outState = OUTSTATE_BLINKDETECT_FRAME_NOBLINK;

				if(blink)
					std::cout << "****** Blink Detected! ******" << std::endl;

#ifdef TEST_AND_DUMP_SVM_RESULTS
				bool closed = (y > DETECT_THRESHOLD);
				bool unsure = (y <= DETECT_THRESHOLD) & (y >= -DETECT_THRESHOLD);
				std::stringstream path;
#ifdef TEST_AND_DUMP_OPEN
				cv::Mat img;
				cv::resize(faceWarpedImage(curEyes[j]), img, cv::Size(48, 32));
				if(closed && !unsure)
					path << dumpResSVMName << "open\\incorrect\\" << name << "_" << mFrameNum << "_" << j << ".png";
				else if (!unsure) {
					accuracy += 1;
					path << dumpResSVMName << "open\\correct\\" << name << "_" << mFrameNum << "_" << j << ".png";
				}
				else
					path << dumpResSVMName << "open\\unsure\\" << name << "_" << mFrameNum << "_" << j << ".png";
				totalNumSamples += 1;
				cv::imwrite(path.str(), img);
				std::cout << "Accuracy: " << accuracy / totalNumSamples << std::endl;
				std::cout << "Correct: " << accuracy << std::endl;
				std::cout << "Incorrect: " << totalNumSamples - accuracy << std::endl;
#endif

#ifdef TEST_AND_DUMP_CLOSED
				cv::Mat img;
				cv::resize(faceWarpedImage(curEyes[j]), img, cv::Size(48, 32));
				if(!closed && !unsure)
					path << dumpResSVMName << "closed\\incorrect\\" << name << "_" << mFrameNum << "_" << j << ".png";
				else if (!unsure) {
					accuracy += 1;
					path << dumpResSVMName << "closed\\correct\\" << name << "_" << mFrameNum << "_" << j << ".png";
				}
				else
					path << dumpResSVMName << "closed\\unsure\\" << name << "_" << mFrameNum << "_" << j << ".png";
				totalNumSamples += 1;
				cv::imwrite(path.str(), img);
				std::cout << "Accuracy: " << accuracy / totalNumSamples << std::endl;
				std::cout << "Correct: " << accuracy << std::endl;
				std::cout << "Incorrect: " << totalNumSamples - accuracy << std::endl;
#endif

#endif

				//std::cout << "Label predicted for frame " << mFrameNum << " : " << y << std::endl;
				//cv::waitKey(0);
				/*if(blink) {
					cv::rectangle(frame, cv::Rect(0, 0, 50, 50), cv::Scalar(0, 255, 0), -1);
				}*/
#endif

#ifdef DUMP_HOG
				std::stringstream outName;
				if(mFrameNum <= blinkFrameCount+1 && mFrameNum >= blinkFrameCount-1)
					outName << eyename << "closed\\" << name << "_" << frameCount << "_" << j << ".png";
				else
					outName << eyename << "open\\" << name << "_" << frameCount << "_" << j << ".png";
				cv::Mat im;
				cv::resize(faceWarpedImage(curEyes[j]), im, cv::Size(48, 32));
				cv::imwrite(outName.str(), im); 
#endif
			}
			for(int j=0; j < curEyes.size(); j++) {
				cv::rectangle(faceWarpedImage, curEyes[j], cv::Scalar(0, 255, 0), 2);
			}

		}
EXIT:
//		cv::imshow("Gray face", faceWarped_gray(roi));
		//cv::imshow("Warped face box", faceWarpedImage);

		/*
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
#if 0
            //state = StateMachine(cv::Rect(0, 0, faceWarpedImage.rows, faceWarpedImage.cols) );
#else
			if (blinkStateFromFile == -1)
				state = BlinkDetectionMLPredict(cv::Rect(0, 0, faceWarpedImage.rows, faceWarpedImage.cols) );
			else
			{
				state = BlinkDetectionMLSaveFeaturePoints(cv::Rect(0, 0, faceWarpedImage.rows, faceWarpedImage.cols), blinkStateFromFile);
				state = blinkStateFromFile;
			}
#endif
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
		*/
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

/*
BlinkDetectorReturnType BlinkDetector::blink_detect_new(cv::Mat frame, int blinkStateFromFile)
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
#if 0
            //state = StateMachine(cv::Rect(0, 0, faceWarpedImage.rows, faceWarpedImage.cols) );
#else
			if (blinkStateFromFile == -1)
				state = BlinkDetectionMLPredict(cv::Rect(0, 0, faceWarpedImage.rows, faceWarpedImage.cols) );
			else
			{
				state = BlinkDetectionMLSaveFeaturePoints(cv::Rect(0, 0, faceWarpedImage.rows, faceWarpedImage.cols), blinkStateFromFile);
				state = blinkStateFromFile;
			}
#endif
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
*/


int BlinkDetector::doMotionEstimation(cv::Mat newFrame, cv::Mat oldFrame, cv::Rect faceRegion, int Index, double &thres1,double minEyeBlocks, int &motion, double *extraOutput)
{
    int row = faceRegion.width/BLOCK_SIZE;
    int numBlocks = row*row;
    

    //double thres1;



#if 1
    cv::Mat tempImage = newFrame(faceRegion);
    cv::Mat YCurrImage;
    matlabRGB2Y(tempImage, YCurrImage);

    tempImage = oldFrame(faceRegion);
    cv::Mat YRefImage;
    matlabRGB2Y(tempImage, YRefImage);

    cv::Scalar meanCurrImage = cv::mean(YCurrImage);
    cv::Scalar meanRefImage = cv::mean(YRefImage);

    cv::Mat srcMat = YCurrImage;
    cv::Mat A = cv::Mat(srcMat.rows,srcMat.cols,CV_32F);
    A = srcMat - (float)meanCurrImage[0,0];

    cv::Mat srcMat2 = YRefImage;
    cv::Mat B = cv::Mat(srcMat2.rows, srcMat2.cols,CV_32F);
    B = srcMat2 - (float)meanRefImage[0,0];

#else
    cv::Mat tempImage = newFrame(faceRegion);
    cv::Mat currImage;
    cv::Mat t1,t2,t3;
    matlabRGB2YCrCb(tempImage, t1,t2,t3, currImage);
    std::vector<cv::Mat> channelsCurrImage(3);
    cv::split(currImage, channelsCurrImage);

    tempImage = oldFrame(faceRegion);
    cv::Mat refImage1;
    matlabRGB2YCrCb(tempImage, t1,t2,t3, refImage1);
    std::vector<cv::Mat> channelsRefImage(3);
    cv::split(refImage1, channelsRefImage);

    cv::Scalar meanCurrImage = cv::mean(channelsCurrImage[0]);
    cv::Scalar meanRefImage = cv::mean(channelsRefImage[0]);

    //Implementing cv::Mat A = channelsCurrImage[0] - meanCurrImage[0,0]; 
    cv::Mat srcMat = channelsCurrImage[0];
    cv::Mat A = cv::Mat(srcMat.rows,srcMat.cols,CV_32F);
    for (int a=0; a<srcMat.rows; a++)
        for (int b=0; b<srcMat.cols; b++)
            A.at<float>(a,b) = ((float) srcMat.at<uchar>(a,b)) - (float)meanCurrImage[0,0];

    // Implementing cv::Mat B = channelsRefImage[0] - meanRefImage[0,0];
    cv::Mat srcMat2 = channelsRefImage[0];
    cv::Mat B = cv::Mat(srcMat2.rows, srcMat2.cols,CV_32F);
    for (int a=0; a< srcMat2.rows; a++)
        for (int b=0; b< srcMat2.cols; b++)
            B.at<float>(a,b) = ((float) srcMat2.at<uchar>(a,b)) - (float)meanRefImage[0,0];
#endif
    //printf("Scalar values= %f, %f\n", (float)meanCurrImage[0,0], (float)meanRefImage[0,0]);

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

	if (extraOutput != nullptr)
		motionStats->updateStats(motionVect, lenVectors, 1.9, motionMask, Index, extraOutput + 8);
	else
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
    motion = motionStats->analyzeMotion(mask,row,minEyeBlocks,localCount,blinkState,row,row, extraOutput);

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
                //if (prevMotion == 0 && motion == 1)
                //{
                //    count1 = 2000; //cannot be blink
                //} //for later
                
                if ((count1 < maxCount) && ((count1 <= 2 * minEyeBlocks && motion == 0) && count1 < prevCount && framesInCurrState > 1) || (framesInCurrState == 1 && (double)count1 <= (double)(minEyeBlocks / 2) && prevCount > minEyeBlocks && motion == 0 && count1 < prevCount))
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

		if (blinkSVM != NULL)
			blinkSVM->MLReset();
    }
}


