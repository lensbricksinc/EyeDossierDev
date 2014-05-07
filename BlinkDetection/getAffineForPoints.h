
#include <opencv2/opencv.hpp>

void getAffineForPoints(std::vector<cv::Point2f> &points1,
                            std::vector<cv::Point2f> &points2,
                            std::vector<unsigned int> &inliers,
                            cv::Mat &outAffMatx);

cv::Mat computeTForm(int sampleSize, 
                  std::vector<cv::Point2f> &points1,
                  std::vector<cv::Point2f> &points2,
                  std::vector<int> indices);

int computeLoopNumberSVD(int sampleSize, int confidence, int pointNum, int inlierNum);

void normalizePoints(std::vector<cv::Point2f> &points,
                     std::vector<int> indices,
                     std::vector<cv::Point2f> &samples,
                     cv::Mat & normMat);


void drawFaceBox(cv::Mat frame, std::vector<cv::Point2f> faceBox);

cv::Rect findFaceBox(cv::Mat frame, cv::Rect origSizeFaceBox, bool &outBoolForceReinit,
                     cv::Mat affMatInv, std::vector<cv::Point2f> pointsFaceBoxTransformed);

bool isFeatureResetRequired(cv::Mat fullFrame, cv::vector<cv::Point2f> faceBox, std::vector<cv::Point2f> featurePoints);

void computerSimilaritySVD(std::vector<cv::Point2f> &points1,
                            std::vector<cv::Point2f> &points2,
                            cv::Mat &outAffMatx);

void transformPointsForward(std::vector<cv::Point2f> &points1,
                     std::vector<cv::Point2f> &points2,
                     cv::Mat affineMat);

void transformPointsForward(std::vector<cv::Point2i> &points1,
                     std::vector<cv::Point2i> &points2,
                     cv::Mat affineMat);
