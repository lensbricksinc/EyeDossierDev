#include <opencv2\opencv.hpp>


#ifndef __MOTIONDS_H__

#define __MOTIONSDS_H__
void motionEstDS(cv::Mat imgP, cv::Mat imgI, int mbSize, int p, double *motionVect[4], double &DScomputations);


#endif

