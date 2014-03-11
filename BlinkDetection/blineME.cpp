
#include "motionEstDS.h"
#include "getFileNames.h"
#include "utilityFuncs.h"

#define COUNT_BLINK_DB 32
#define COUNT_NOBLINK_DB 100

using namespace std;

void temp()
{
    cv::Mat tempImg = cv::imread("D:\\work\\blinkdetection\\matlab\\frame_000120.jpg");

    cv::Mat Y;
    cv::Mat cb;
    cv::Mat cr;
    cv::Mat YCrCb;
    matlabRGB2YCbCr(tempImg, Y, cb, cr, YCrCb);

    cv::imwrite("D:\\work\\blinkdetection\\matlab\\output\\cppoutput3.png", YCrCb);

    //cv::imwrite("", YCr

    /*
    double minValTemp[3];
    double maxValTemp[3];
    cv::Point minIdxTemp;
    cv::Point maxIdxTemp;

    //cv::minMaxIdx(tempImg, minValTemp, maxValTemp, &minIdxTemp, &maxIdxTemp);
    cv::minMaxLoc(tempImg, minValTemp, maxValTemp, &minIdxTemp, &maxIdxTemp, cv::Mat());
    //printf("MinVal= %f. \n Maxval= %f\n MinIndex=(%d,%d,%d). MaxIndex=(%d,%d,%d)", minValTemp[0],maxValTemp[0]);

    cv::Mat refImage1;
    cv::cvtColor(tempImg, refImage1, CV_BGR2YCrCb );
    std::vector<cv::Mat> channelsRefImage(3);
    cv::split(refImage1, channelsRefImage);

    cv::Scalar meanRefImage = cv::mean(channelsRefImage[0]);

    double minVal[3];
    double maxVal[3];
    int minIdx[3];
    int maxIdx[3];
    cv::minMaxIdx(channelsRefImage[0], minVal, maxVal,
                      minIdx, maxIdx);

    //printf("minVal= %f, maxval= %f, minIdx=%d, maxIdx=%d, mean= %d \n", *minVal, *maxVal, *minIdx, *maxIdx, (int) meanRefImage[0,0]);


    unsigned char* ptrImg;
    ptrImg = channelsRefImage[0].data;
    int minVal2= 65536;
    int maxVal2= -1;
    for (int i=0; i<channelsRefImage[0].rows; i++)
    {
        for (int j=0; j<channelsRefImage[0].cols; j++)
        {
            unsigned char val = ptrImg[j];
            int valInt = (int)val;
            if (valInt < minVal2)
                minVal2 = valInt;

            if (valInt > maxVal2)
                maxVal2 = valInt;
        }
        ptrImg+= channelsRefImage[0].step;
    }

    printf("minVal= %d, maxval= %d", minVal2, maxVal2);
    */

}




void blineME (cv::String rootDir) 
{
    

    
    //cv::String rootDir = "D:\\work\\blinkdetection\\BlinkDetectionPranav\\BlinkData\\blink\\blink02\\";

    

    /*
    int faceArray[10][4]=
    {{481, 76, 480, 480},
    {483, 77, 480, 480},
    {481, 77, 480, 480},
    {480, 76, 464, 464},
    {468, 66, 480, 480},
    {463, 66, 464, 464},
    {465, 63, 448, 448},
    {462, 58, 448, 448},
    {462, 45, 448, 448},
    {455, 28, 448, 448}};
    */

    /*
    int faceArray[10][4] =
    {
        {580, 308, 304, 304},
        {579, 310, 304, 304},
        {568, 312, 320, 320},
        {570, 314, 320, 320},
        {554, 326, 304, 304},
        {542, 328, 320, 320},
        {529, 333, 320, 320},
        {518, 335, 320, 320},
        {504, 334, 320, 320},
        {502, 332, 320, 320}
    };
    */
    std::vector<std::string> listFiles;
    cv::String fileExt = "*.jpg";
    //cv::String fileExt = "*.*";
    listFiles = listFilesInDirectory(rootDir+fileExt);
    if (listFiles.size() == 1)
    {
        fileExt = "*.png";
        listFiles = listFilesInDirectory(rootDir+fileExt);
    }
    int numFiles=0;
    for each (string str in listFiles)
    {
        numFiles++;
        cout << str << endl;
    }

    int count    = 0;
    int blink    = 0;
    int blinkState = 0;
    int count1     = 0;
    int prevCount = 0;
    int framesInCurrState = 0;
    cv::Mat prevFrame;    // Needs to be initialised
    cv::Mat refStartFrame;
    int BLOCK_SIZE_LOC= 16;

    for (int fileIndex= 0; fileIndex<numFiles; fileIndex++){

        cv::String fileFullPath = rootDir + listFiles[fileIndex];
        cv::Mat image = cv::imread(fileFullPath);

        cv::CascadeClassifier face_cascade;     // To be initialised
        
        cv::Mat image_gray;
        std::vector<cv::Rect> faces;

        cv::cvtColor(image, image_gray, CV_BGR2GRAY);

        cv::String face_cascade_name = "cascades\\haarcascade_frontalface_alt2.xml";
        if (!face_cascade.load(face_cascade_name))
        {
            printf("Unable to load face cascade");
            return;
        };

        //-- Detect faces
        face_cascade.detectMultiScale( image_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );
    
        if (faces.size() == 0)
            continue;

        int maxWidth     = 0;
        int index   = 0;
        // Find face with maximum size
        for (int i=0; i<faces.size(); i++)
        {
            if (faces[i].width > maxWidth)
            {
                maxWidth = faces[i].width;
                index = i;
            }
        }
        cv::Rect maxFace = faces[index];

        if (maxFace.width %32 != 0)
        {
            int extra = maxFace.width%32;
            int extra_2 = extra/2;
            maxFace.x += extra_2;
            maxFace.y +=extra_2;
            maxFace.width -= extra;
            maxFace.height -=extra;
        }

        /*
        cv::Mat dispImg = image.clone();
        // Draw rectangle only for largest face
        cv::Point lefttop(maxFace.x, maxFace.y);
        cv::Point rightbottom( (maxFace.x + maxFace.width), (maxFace.y + maxFace.height));
        cv::rectangle(dispImg, lefttop, rightbottom, cv::Scalar( 255, 0, 255 ));
        */
        //cv::imshow("Frame", dispImg);
        //cv::waitKey(0);
       
        /*
        maxFace.x = faceArray[fileIndex][0];
        maxFace.y = faceArray[fileIndex][1];
        maxFace.width = faceArray[fileIndex][2];
        maxFace.height = faceArray[fileIndex][3];
        */

        // Make face width a remainder of 32.
        //BB(index,3)=BB(index,3) - rem(BB(index,3),32)-1; % Making it a (remainder of 32) - 1
        //BB(index,4)=BB(index,4) - rem(BB(index,4),32)-1;
        int row = maxFace.width/16;       // 16 is block size?


        
        if (count!=0)
        {
            //%do motion estimation
            cv::Mat tempImage = image(maxFace);
            cv::Mat currImage;
            //cv::cvtColor(tempImage, currImage, CV_BGR2YCrCb );
            cv::Mat t1,t2,t3;
            matlabRGB2YCrCb(tempImage, t1,t2,t3, currImage);
            std::vector<cv::Mat> channelsCurrImage(3);
            cv::split(currImage, channelsCurrImage);

            tempImage = prevFrame(maxFace);
            cv::Mat refImage1;
            //cv::cvtColor(tempImage, refImage1, CV_BGR2YCrCb );
            matlabRGB2YCrCb(tempImage, t1,t2,t3, refImage1);
            std::vector<cv::Mat> channelsRefImage(3);
            cv::split(refImage1, channelsRefImage);

            cv::Scalar meanCurrImage = cv::mean(channelsCurrImage[0]);
            cv::Scalar meanRefImage = cv::mean(channelsRefImage[0]);

            //Implementing cv::Mat A = channelsCurrImage[0] - meanCurrImage[0,0];
            cv::Mat srcMat = channelsCurrImage[0];
            cv::Mat A = cv::Mat(srcMat.rows, srcMat.cols, CV_32F);
            for (int a=0; a<srcMat.rows; a++)
            {
                for (int b=0; b<srcMat.cols; b++)
                {
                    A.at<float>(a,b) = ((float) srcMat.at<uchar>(a,b)) - (float)meanCurrImage[0,0];
                }
            }

            // Implementing cv::Mat B = channelsRefImage[0] - meanRefImage[0,0];
            cv::Mat srcMat2 = channelsRefImage[0];
            cv::Mat B = cv::Mat(srcMat2.rows, srcMat2.cols, CV_32F);
            for (int a=0; a<channelsRefImage[0].rows; a++)
            {
                for (int b=0; b<channelsRefImage[0].cols; b++)
                {
                    B.at<float>(a,b) =  ((float) srcMat2.at<uchar>(a,b)) - (float)meanRefImage[0,0];
                }
            }

            double *motionVect[4];
            int imgRows = A.rows;
            int imgCols = A.cols;
            int lenVectors = imgRows*imgCols/((BLOCK_SIZE_LOC*BLOCK_SIZE_LOC));
            for (int i=0; i<4; i++)
            {
                motionVect[i] = new double[lenVectors];
            }
            double DScomputations;
            if(fileIndex==1)
            {
                fileIndex = 1;
            }
            motionEstDS(A, B, BLOCK_SIZE_LOC, BLOCK_SIZE_LOC, motionVect, DScomputations);

            // Calculating variance at this step
            cv::Mat temp = A.clone();
            temp = temp.mul(temp);
            cv::Scalar sumMat = cv::sum(temp);
            double variance = sumMat.val[0,0]/(A.rows * A.cols - 1);
            variance = variance / 300;

            if (variance < 2)
                variance = 2;
            else
                variance = 3;

            count1=0;

            for (int i=2; i<= (int)((row+1)/2 - 1); i++)
            {
                for (int j= 2; j<=row-1; j++)
                {
                   if (motionVect[2][i*row+j-1] > variance)
                   {
                        count1 = count1+1;
                        //printf("i= %d, j= %d, count= %d\n",i,j,i*row+j);
                   }
                 
                }
            }

            
            for (int i=0; i<4; i++)
            {
                delete[] motionVect[i];
                motionVect[i] = nullptr;
            }
            

            //free(motionVect);

            prevFrame = image;
        }
        else
        {
            prevFrame = image;
        }

        printf("count1= %d\n", count1);
        count = count+1;

        switch (blinkState)
        {
            case 0:
                if (count1 + prevCount> 10)
                {
                    blinkState = 1;
                    prevCount  = (prevCount + count1)/4;
                    if (prevCount < 2)
                        prevCount = 2;

                    framesInCurrState = 0;
                    refStartFrame=image;
                }
                else
                {
                    prevCount = count1;
                }
                break;

            case 1:
                framesInCurrState = framesInCurrState + 1;
                if (count1 < (prevCount+1))
                {
                    blinkState = 3;
                    framesInCurrState = 0;
                    prevCount = 0;
                }
                else
                {
                    if (framesInCurrState > 2)
                    {
                        if (count1 > 10)
                            blinkState = 1;
                        else
                            blinkState = 0;
                    }
                }
                break;

            case 3:
                framesInCurrState = framesInCurrState + 1;
                if (count1 + prevCount>= 5)
                {
                    //eye open starts
                    blinkState = 4;
                    framesInCurrState = 0;
                }
                else
                {
                    prevCount = count1;
                    if (framesInCurrState > 4)
                   {
                        blinkState = 0;
                        prevCount = 0;
                    }
                }
                break;

            case 4:
            {
                framesInCurrState = framesInCurrState+1;

                cv::Mat tempImage = image(maxFace);
                cv::Mat currImage;
                cv::Mat t4,t5,t6;
                //cv::cvtColor(tempImage, currImage, CV_BGR2YCrCb );
                matlabRGB2YCrCb(tempImage, t4,t5,t6, currImage);

                std::vector<cv::Mat> channelsCurrImage(3);
                cv::split(currImage, channelsCurrImage);

                tempImage = refStartFrame(maxFace);
                cv::Mat refFrame1;
                //cv::cvtColor(tempImage,refFrame1, CV_BGR2YCrCb);
                matlabRGB2YCrCb(tempImage, t4,t5,t6, refFrame1);
                std::vector<cv::Mat> channelsRefImage(3);
                cv::split(refFrame1, channelsRefImage);

                cv::Scalar meanCurrImage = mean(channelsCurrImage[0]);
                cv::Scalar meanRefImage = mean(channelsRefImage[0]);

                //Implementing cv::Mat A = channelsCurrImage[0] - meanCurrImage[0,0]; 
                cv::Mat A = cvCreateMat(channelsCurrImage[0].rows,channelsCurrImage[0].cols,CV_32F);
                //printf("Type of A= %d\n", A.type());        // = 3
                unsigned char* ptrSrc = channelsCurrImage[0].data;
                for (int a=0; a<channelsCurrImage[0].rows; a++)
                {
                    for (int b=0; b<channelsCurrImage[0].cols; b++)
                    {
                        A.at<float>(a,b) = ((float) ptrSrc[b]) - (float)meanCurrImage[0,0];
                    }
                    ptrSrc += channelsCurrImage[0].step;
                }

                // Implementing cv::Mat B = channelsRefImage[0] - meanRefImage[0,0];
                cv::Mat B = cvCreateMat(channelsRefImage[0].rows,channelsRefImage[0].cols,CV_32F);
                ptrSrc = channelsRefImage[0].data;
                for (int a=0; a<channelsRefImage[0].rows; a++)
                {
                    for (int b=0; b<channelsRefImage[0].cols; b++)
                    {
                        B.at<float>(a,b)  = ((float) ptrSrc[b]) - (float)meanRefImage[0,0];
                    }
                    ptrSrc += channelsRefImage[0].step;
                }

                double *motionVect[4];
                int imgRows = A.rows;
                int imgCols = A.cols;
                int lenVectors = imgRows*imgCols/((BLOCK_SIZE_LOC*BLOCK_SIZE_LOC));
                for (int i=0; i<4; i++)
                {
                    motionVect[i] = new double[lenVectors];
                }
                double DScomputations;

                motionEstDS(A, B, BLOCK_SIZE_LOC, BLOCK_SIZE_LOC, motionVect, DScomputations);
                //prevcount1 = count1;
                count1=0;
                //double sum = 0;
                for (int i=2; i<= (int)((row+1)/2 -1); i++)
                {
                    for (int j = 2; j<=row-1; j++)
                    {
                        //sum = sum+motionVect[3][i*row+j-1];
                        if (motionVect[2][i*row+j-1] > 3)
                       {
                            count1 = count1+1;
                           
                        }
                    }
                }

                
                for (int i=0; i<4; i++)
                {
                    delete[] motionVect[i];
                    motionVect[i] = nullptr;
                }

                //free(motionVect);
                

                if (count1 < 50)
                {
                    blink = 1;
                    blinkState = 0;
                }

                if (blinkState == 4 && framesInCurrState > 3)
                {
                    blinkState = 0;
                }
            }
                break;



        }

        
    }

    if (blink == 0)
        std::cout<< "No blink detected in " <<rootDir<<endl;
    else
        std::cout<< "Blink detected in " <<rootDir<<endl;

    std::cout<<std::endl<<std::endl;

    return;
}




void blineMEBatch()
{
    int numFolders;
    bool blinkDatabase=1;
    cv::String baseDir ="D:\\work\\blinkdetection\\BlinkDetectionPranav\\BlinkData\\";
    cv::String addDir;
    if (blinkDatabase)
    {
        numFolders=COUNT_BLINK_DB;
        addDir="blink\\blink";
    }
    else
    {
        numFolders=COUNT_NOBLINK_DB;
        addDir="nonblink\\noblink";
    }

    for (int i=32; i<=numFolders; i++)
    {
        cv::String fileNameDigit = std::to_string(i);
        if (fileNameDigit.size() == 1)
            fileNameDigit = "0"+ fileNameDigit;

        cv::String fullFilePath = baseDir + addDir + fileNameDigit + "\\";
        blineME(fullFilePath);
    
    }

    cv::Mat temp = cv::Mat(480, 640, CV_8U);
    cv::imshow("Frame2", temp);
    cv::waitKey(0);
};