//#include <opencv2\opencv.hpp>

class MotionRegionOnSAD
{
private:
	double *meanS;
	double *varS;
    int numBlocks;
    int framesCurrRun;
	int find(int** mask, int start, int end, int* row, int* col, int endRow);
	double mean(int* array, int count);
	double var(int* array, double mean, int count);
	int calculateEyeBlocks(int** mask, int start, int end, int* row, int* col, int endRow, double meanX, double meanY, double varX, double varY);

public:
	double RHO;
	double INIT_VARIANCE;
	double prevLeftX, prevLeftY;
	double prevRightX, prevRightY;

	MotionRegionOnSAD();
	~MotionRegionOnSAD();

    void updateStats(double *motionVect[4], int lenVectors, float VAR_FACTOR, int* motionMask, int blinkState);
    void getMotionMask(double *motionVect[4], float VAR_FACTOR, int* motionMask, int blinkState);
	void resetMotionStats();
	int  analyzeMotion(int** mask, int row, double minEyeBlocks, int motionCount, int blinkState, int imgRows, int imgCols);
	
};
