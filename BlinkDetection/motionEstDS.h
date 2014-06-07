
#ifndef __MOTIONESTDS_H__
#define __MOTIONESTDS_H__

#include <opencv2\opencv.hpp>

void motionEstDS(cv::Mat imgP, cv::Mat imgI, int mbSize, int p, double *motionVect[4], double &DScomputations);


#endif

