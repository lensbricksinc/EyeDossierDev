
#include "utilityFuncs.h"

void matlabRGB2YCbCr(cv::Mat &rgb, cv::Mat &Y, cv::Mat &cb, cv::Mat &cr, cv::Mat &YCrCb)
{
    int imgWidth = rgb.cols;
    int imgHeight = rgb.rows;

    std::vector<cv::Mat> rgbChannels(3);
    cv::split(rgb, rgbChannels);

    cv::Mat R= cv::Mat(imgHeight, imgWidth, CV_32F);
    cv::Mat G= cv::Mat(imgHeight, imgWidth, CV_32F);
    cv::Mat B= cv::Mat(imgHeight, imgWidth, CV_32F);

   // unsigned char *ptrB = rgbChannels[0].data;
    //unsigned char *ptrG = rgbChannels[1].data;
    //unsigned char *ptrR = rgbChannels[2].data;
    for (int i=0; i<imgHeight; i++){
        for (int j=0; j<imgWidth; j++){
            //R.at<float>(i,j) = (float) ptrR[j];
            //G.at<float>(i,j) = (float) ptrG[j];
            //B.at<float>(i,j) = (float) ptrB[j];
            R.at<float>(i,j) = (float) rgbChannels[2].at<unsigned char>(i,j);
            G.at<float>(i,j) = (float) rgbChannels[1].at<unsigned char>(i,j);
            B.at<float>(i,j) = (float) rgbChannels[0].at<unsigned char>(i,j);
        }
        //ptrR += rgbChannels[0].step;
        //ptrG += rgbChannels[1].step;
        //ptrB += rgbChannels[2].step;
    }

    //printf("Initial values: R=%d, G=%d, B= %d\n", (int)R.at<float>(199,199), (int)G.at<float>(199,199), (int)B.at<float>(199,199));

    float T[3][3] = 
    {{65.481, 128.553, 24.966},
     {-37.797, -74.203, 112},
     {112, -93.786, -18.214}};

    for (int i=0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            T[i][j] = T[i][j]/255.0;
        }
    }
    
    float origOffsetArray[3] = {16.0, 128.0, 128.0};

    cv::Mat Y = (R*T[0][0] + G*T[0][1] + B*T[0][2] + origOffsetArray[0]);
    cv::Mat cb = (R*T[1][0] + G*T[1][1] + B*T[1][2]  + origOffsetArray[1]);
    cv::Mat cr = (R*T[2][0] + G*T[2][1] + B*T[2][2]  + origOffsetArray[2]);

    //printf("Initial values: Y=%d, Cb=%d, Cr= %d\n", (int)Y.at<float>(199,199), (int)cb.at<float>(199,199), (int)cr.at<float>(199,199));

    YCrCb = cv::Mat::zeros(imgHeight, imgWidth, CV_8UC3);


    for (int i=0; i< imgHeight; i++)
    {
        for (int j=0; j<imgWidth; j++)
        {
            YCrCb.at<cv::Vec3b>(i,j)[0] = (unsigned int)(Y.at<float>(i,j)+0.5);
            YCrCb.at<cv::Vec3b>(i,j)[1] = (unsigned int)(cb.at<float>(i,j)+0.5);
            YCrCb.at<cv::Vec3b>(i,j)[2] = (unsigned int)(cr.at<float>(i,j)+0.5);
        }
    }

    return;
}



void matlabRGB2YCrCb(cv::Mat &rgb, cv::Mat &Y, cv::Mat &cb, cv::Mat &cr, cv::Mat &YCrCb)
{
    int imgWidth = rgb.cols;
    int imgHeight = rgb.rows;

    std::vector<cv::Mat> rgbChannels(3);
    cv::split(rgb, rgbChannels);

    cv::Mat R= cv::Mat(imgHeight, imgWidth, CV_32F);
    cv::Mat G= cv::Mat(imgHeight, imgWidth, CV_32F);
    cv::Mat B= cv::Mat(imgHeight, imgWidth, CV_32F);

   // unsigned char *ptrB = rgbChannels[0].data;
    //unsigned char *ptrG = rgbChannels[1].data;
    //unsigned char *ptrR = rgbChannels[2].data;
    for (int i=0; i<imgHeight; i++){
        for (int j=0; j<imgWidth; j++){
            //R.at<float>(i,j) = (float) ptrR[j];
            //G.at<float>(i,j) = (float) ptrG[j];
            //B.at<float>(i,j) = (float) ptrB[j];
            R.at<float>(i,j) = (float) rgbChannels[2].at<unsigned char>(i,j);
            G.at<float>(i,j) = (float) rgbChannels[1].at<unsigned char>(i,j);
            B.at<float>(i,j) = (float) rgbChannels[0].at<unsigned char>(i,j);
        }
        //ptrR += rgbChannels[0].step;
        //ptrG += rgbChannels[1].step;
        //ptrB += rgbChannels[2].step;
    }

    //printf("Initial values: R=%d, G=%d, B= %d\n", (int)R.at<float>(199,199), (int)G.at<float>(199,199), (int)B.at<float>(199,199));

    float T[3][3] = 
    {{65.481, 128.553, 24.966},
     {-37.797, -74.203, 112},
     {112, -93.786, -18.214}};

    for (int i=0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            T[i][j] = T[i][j]/255.0;
        }
    }
    
    float origOffsetArray[3] = {16.0, 128.0, 128.0};

    Y = cv::Mat(imgHeight, imgWidth, CV_32F);
    cb = cv::Mat(imgHeight, imgWidth, CV_32F);
    cr = cv::Mat(imgHeight, imgWidth, CV_32F);

    /*
    for (int i=0; i<imgHeight; i++)
    {
        for (int j=0; j<imgWidth; j++)
        {
            Y.at<float>(i,j) = R.at<float>(i,j) * T[0][0] + G.at<
        }
    }
    */

    Y = (R*T[0][0] + G*T[0][1] + B*T[0][2] + origOffsetArray[0]);
    cb = (R*T[1][0] + G*T[1][1] + B*T[1][2]  + origOffsetArray[1]);
    cr = (R*T[2][0] + G*T[2][1] + B*T[2][2]  + origOffsetArray[2]);

    //printf("Initial values: Y=%d, Cb=%d, Cr= %d\n", (int)Y.at<float>(199,199), (int)cb.at<float>(199,199), (int)cr.at<float>(199,199));

    YCrCb = cv::Mat::zeros(imgHeight, imgWidth, CV_8UC3);


    for (int i=0; i< imgHeight; i++)
    {
        for (int j=0; j<imgWidth; j++)
        {
            YCrCb.at<cv::Vec3b>(i,j)[0] = (unsigned int)(Y.at<float>(i,j) + 0.5);
            YCrCb.at<cv::Vec3b>(i,j)[1] = (unsigned int)(cr.at<float>(i,j)+0.5);
            YCrCb.at<cv::Vec3b>(i,j)[2] = (unsigned int)(cb.at<float>(i,j)+0.5);
        }
    }

    return;
}



void matlabRGB2Y(cv::Mat &rgb, cv::Mat &Y)
{
    int imgWidth = rgb.cols;
    int imgHeight = rgb.rows;

    std::vector<cv::Mat> rgbChannels(3);
    cv::split(rgb, rgbChannels);

    cv::Mat R= cv::Mat(imgHeight, imgWidth, CV_32F);
    cv::Mat G= cv::Mat(imgHeight, imgWidth, CV_32F);
    cv::Mat B= cv::Mat(imgHeight, imgWidth, CV_32F);

#if 1
    rgbChannels[2].convertTo(R,CV_32F);
    rgbChannels[1].convertTo(G,CV_32F);
    rgbChannels[0].convertTo(B,CV_32F);
#else
    for (int i=0; i<imgHeight; i++){
        for (int j=0; j<imgWidth; j++){
            R.at<float>(i,j) = (float) rgbChannels[2].at<unsigned char>(i,j);
            G.at<float>(i,j) = (float) rgbChannels[1].at<unsigned char>(i,j);
            B.at<float>(i,j) = (float) rgbChannels[0].at<unsigned char>(i,j);
        }
    }
#endif

    float T[3][3] = 
    {{65.481, 128.553, 24.966},
     {-37.797, -74.203, 112},
     {112, -93.786, -18.214}};

    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            T[i][j] = T[i][j]/255.0;
    
    float origOffsetArray[3] = {16.0, 128.0, 128.0};

    Y = cv::Mat(imgHeight, imgWidth, CV_32F);
    Y = (R*T[0][0] + G*T[0][1] + B*T[0][2] + origOffsetArray[0]);       // +0.5 might be required for bringing results close to usage of matlabRGB2YCrCb

    return;
}

void dumpFrames(cv::Mat img)
{
    static int frameCount = 0;

}

/*
void dumpToFile(cv::Mat img, cv::String filename)
{



}
*/