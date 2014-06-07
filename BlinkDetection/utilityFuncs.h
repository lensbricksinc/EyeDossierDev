#ifndef __UTILITYFUNCS_H__
#define __UTILITYFUNCS_H__


#include <opencv2\opencv.hpp>

void matlabRGB2YCrCb(cv::Mat &rgb, cv::Mat &Y, cv::Mat &cb, cv::Mat &cr, cv::Mat &YCrCb);
void matlabRGB2YCbCr(cv::Mat &rgb, cv::Mat &Y, cv::Mat &cb, cv::Mat &cr, cv::Mat &YCrCb);
void matlabRGB2Y(cv::Mat &rgb, cv::Mat &Y);

#endif
