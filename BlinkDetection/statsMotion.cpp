
#include "statsMotion.h";


MotionRegionOnSAD::MotionRegionOnSAD()
{
    //framesCurrRun = 0;
    meanS = nullptr;
    varS = nullptr;
	RHO = 0.05;
	INIT_VARIANCE = 0.05;

    resetMotionStats();
    return;
};


MotionRegionOnSAD::~MotionRegionOnSAD()
{
    if (meanS != nullptr)
    {
        delete[] meanS;
        meanS = nullptr;
    }

    if (varS != nullptr)
    {
        delete[] varS;
        varS = nullptr;
    }

    return;
};


void MotionRegionOnSAD::updateStats(double *motionVect[4], int lenVectors, float VAR_FACTOR, int* motionMask, int blinkState)
{
    double *motionSAD = motionVect[2];
    numBlocks = lenVectors;

    framesCurrRun++;

    if (meanS == nullptr || varS == nullptr)
    {
        
        if (meanS != nullptr)
        {
            delete[] meanS;
            meanS = nullptr;
        }

        if (varS != nullptr)
        {
            delete[] varS;
            varS = nullptr;
        }

        meanS = new double[lenVectors];
        for (int i = 0; i < lenVectors; i++)
            meanS[i] = motionSAD[i];

        varS = new double[lenVectors];
        for (int i = 0; i < lenVectors; i++)
            varS[i] = INIT_VARIANCE;

        for (int i = 0; i < lenVectors; i++)
            motionMask[i] = 0;
        
        return;
    }


    for (int i = 0; i < lenVectors; i++)
    {
        double currDiff = (meanS[i] - motionSAD[i]);
        currDiff = currDiff*currDiff;
        if (blinkState || framesCurrRun > 30)
        {
        
            meanS[i] = (1 - RHO)*meanS[i] + RHO*motionSAD[i];
            varS[i] = (1 - RHO)*varS[i] + RHO*currDiff;
        }

        if (currDiff >(VAR_FACTOR * varS[i]))
            motionMask[i] = 1;
        else
            motionMask[i] = 0;
    }

    
    return;
};


void MotionRegionOnSAD::getMotionMask(double *motionVect[4], float VAR_FACTOR, int* motionMask, int blinkState)
{
    if (motionMask == nullptr || meanS == nullptr || varS == nullptr)
        return;

    double *motionSAD = motionVect[2];
    for (int i = 0; i < numBlocks; i++)
    {
        double currDiff = meanS[i] - motionSAD[i];
        currDiff = currDiff*currDiff;
        if (currDiff > (VAR_FACTOR* varS[i]))
            motionMask[i] = 1;
        else
            motionMask[i] = 0;
    }

    return;
};


void MotionRegionOnSAD::resetMotionStats()
{
    if (meanS != nullptr)
    {
        delete[] meanS;
        meanS = nullptr;
    }
    if (varS != nullptr)
    {
        delete[] varS;
        varS = nullptr;
    }
    framesCurrRun = 0;
    numBlocks = -1;
    return;
};

