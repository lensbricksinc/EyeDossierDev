//#include <opencv2\opencv.hpp>

class MotionRegionOnSAD
{
private:
	double *meanS;
	double *varS;
    int numBlocks;
    int framesCurrRun;

public:
	double RHO;
	double INIT_VARIANCE;

	MotionRegionOnSAD();
	~MotionRegionOnSAD();

    void updateStats(double *motionVect[4], int lenVectors, float VAR_FACTOR, int* motionMask, int blinkState);
    void getMotionMask(double *motionVect[4], float VAR_FACTOR, int* motionMask, int blinkState);
	void resetMotionStats();

};
