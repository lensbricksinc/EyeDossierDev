
#include <iostream>
#include <opencv2\opencv.hpp>
#include <mutex>


class FRAMEDUMP {
#define SIZE_FRAME_ARRAY 100
private:
    cv::Mat FrameArray[SIZE_FRAME_ARRAY];
    int     FrameNumber[SIZE_FRAME_ARRAY];
    int currIndex;      // The frame at currIndex is either NULL, or the oldest one.
    std::mutex mutFrameAccess;
    int mInputFrameCount;
public:

    FRAMEDUMP()
    {
        for (int i=0; i<SIZE_FRAME_ARRAY; i++)
        {
            FrameArray[i] = cv::Mat();
            FrameNumber[i] = -1;
        }
        currIndex= 0;
        mInputFrameCount= 0;
    }

    int AddFrameToInternalMemory(cv::Mat frame)
    {

        mutFrameAccess.lock();
        mInputFrameCount++;
        FrameArray[currIndex] = frame.clone();
        FrameNumber[currIndex] = mInputFrameCount;
        currIndex++;
        
        if (currIndex >= SIZE_FRAME_ARRAY)
        {
            currIndex= 0;
        }
        mutFrameAccess.unlock();

        return 0;
    }
    
    int DumpFrameArrayToMemory(cv::String sDir)
    {

        mutFrameAccess.lock();
        char lastChar = *sDir.rbegin();     // get the last character in the string
        if (strcmp(&lastChar, "\\") != 0)
            sDir= sDir + "\\";

        if (FrameArray[currIndex].rows != 0)
        {
            for (int i=currIndex; i<SIZE_FRAME_ARRAY; i++)
            {
                //std::ostringstream fileName;
                char fileName[20];
                sprintf(fileName,"frame_%08d.png", FrameNumber[i]);
                cv::String sOutFileName = sDir + "frame_" + std::string(fileName);
                cv::imwrite(sOutFileName, FrameArray[i]);
            }
        }

        if (currIndex>0)
        {
            for (int i=0; i<currIndex; i++)
            {
                char fileName[20];
                sprintf(fileName,"frame_%08d.png", FrameNumber[i]);
                cv::String sOutFileName = sDir + std::string(fileName);
                cv::imwrite(sOutFileName, FrameArray[i]);
            }
        }
        

        /*
        // Reset dumped data array (is not really necessary)
        for (int i=0; i<SIZE_FRAME_ARRAY; i++)
        {
            FrameArray[i] = cv::Mat();
        }
        currIndex= 0;
        */
        mutFrameAccess.unlock();


        return 0;
    }

};