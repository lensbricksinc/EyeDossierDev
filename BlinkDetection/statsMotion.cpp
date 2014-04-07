
#include <math.h>
#include <stdio.h>
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
        double currDiff = motionSAD[i];
        currDiff = currDiff*currDiff;
        if (blinkState == 0 || framesCurrRun < 30)
        {
        
            //meanS[i] = (1 - RHO)*meanS[i] + RHO*motionSAD[i];
			//meanS[i] = 0;
			varS[i] = (1 - RHO)*varS[i] + RHO*currDiff;
        }

		if (currDiff >(VAR_FACTOR * varS[i]))
		{
			motionMask[i] = 1;
		}
        else
            motionMask[i] = 0;
    }

    
    return;
};

int  MotionRegionOnSAD::find(int** mask, int start, int end, int* row, int* col, int endRow)
{
	int i, j;
	int count;
	count = 0;
	for (j = 0; j < endRow; j++)
	for (i = start; i <= end; i++)
	{
		if (mask[j][i] > 0)
		{
			row[count] = j;
			col[count] = i;
			count = count + 1;
		}
	}
	return count;

}

double MotionRegionOnSAD:: mean(int* array, int count)
{
	int i,sum;
	double mean;
	sum = 0;
	for (i = 0; i < count; i++)
		sum = sum + array[i];
	if (count != 0)
		mean = (0.5 + sum* 1.0 / count);
	else
		mean = 0;
	return mean;

}

double MotionRegionOnSAD::var(int* array, double mean, int count)
{
	int i;
	double var;
	double sum;
	sum = 0;
	for (i = 0; i < count; i++)
		sum = sum + (array[i] - mean)*(array[i]-mean);
	if (count != 0)
		var = sqrt(sum*1.0 / count);
	else
		var = 0;
	if (var < 2)
		var = 2;

	return var;
}

int MotionRegionOnSAD::calculateEyeBlocks(int** mask, int start, int end, int* row, int* col, int endRow, double meanX, double meanY, double varX, double varY)
{
	int count;
	int i, j;
	count = 0;
	for (j = 0; j < endRow; j++)
	for (i = start; i <= end; i++)
	{
		if (mask[j][i] == 255)
		{
			if (j < (meanX - varX) || j > (meanX + varX) || i < (meanY - varY) || i > (meanY + varY))
				mask[j][i] = 0;
			else
				count = count + 1;
		}
	}
	return count;
}

int MotionRegionOnSAD::analyzeMotion(int** mask, int row, double minEyeBlocks, int motionCount, int blinkState, int imgRows, int imgCols)
{
	int motion = 0;
	int *sum;
	int i, j;
	int N;
	int *leftIndexRow, *leftIndexCol;
	int *rightIndexRow, *rightIndexCol;
	double leftX, leftY,leftXD,leftYD;
	double rightX, rightY, rightXD,rightYD;
	double varLeftX, varLeftY;
	double varRightX, varRightY;
	double countLeft, countRight, totalCount;
	double leftCount=0, rightCount=0;
	N = imgCols / 2;
	sum = new int[imgCols]();
	
	leftIndexRow = new int[imgRows*imgCols]();
	leftIndexCol = new int[imgCols*imgRows]();
	rightIndexRow = new int[imgRows*imgCols]();
	rightIndexCol = new int[imgCols*imgRows]();

	for (j = 0; j < imgCols; j++)
	for (i = 0; i < imgRows; i++)
	{
		sum[j] += mask[i][j];
	}
#if 1	
	for (i = 2; i <= 5; i++)
	{
		if ((sum[i] - sum[i + 1] >= sum[i] - 2 * 255) && (sum[i] > 2 * 255))
		{
			for (j = 0; j < imgRows; j++)
				mask[j][i] = 0;
			motionCount -= (sum[i]/255);
		}
	}

	for (i = row-5; i <= row-2; i++)
	{
		if ((sum[i] - sum[i - 1] >= sum[i] - 2 * 255) && (sum[i] > 2 * 255))
		{
			for (j = 0; j < imgRows; j++)
				mask[j][i] = 0;
			motionCount -= (sum[i]/255);
		}
	}
#endif
	leftCount = find(mask, 2, N - 1, leftIndexRow, leftIndexCol, imgRows);
	rightCount= find(mask, N, row - 2, rightIndexRow, rightIndexCol, imgRows);
	leftX = mean(leftIndexRow,leftCount);
	leftY = mean(leftIndexCol,leftCount);
	rightX = mean(rightIndexRow,rightCount);
	rightY = mean(rightIndexCol,rightCount);

	varLeftX = var(leftIndexRow,leftX,leftCount);
	varLeftY = var(leftIndexCol,leftY,leftCount);
	varRightX = var(rightIndexRow,rightX,rightCount);
	varRightY = var(rightIndexCol,rightY,rightCount);

	countLeft = calculateEyeBlocks(mask,2,N-1,leftIndexRow,leftIndexCol,imgRows,leftX,leftY,varLeftX,varLeftY);
	countRight = calculateEyeBlocks(mask, N, row - 2, rightIndexRow, rightIndexCol,imgRows, rightX, rightY, varRightX, varRightY);
	
	leftCount = find(mask, 2, N - 1, leftIndexRow, leftIndexCol, imgRows);
	rightCount = find(mask, N, row - 2, rightIndexRow, rightIndexCol, imgRows);
	leftX = mean(leftIndexRow, leftCount);
	leftY = mean(leftIndexCol, leftCount);
	rightX = mean(rightIndexRow, rightCount);
	rightY = mean(rightIndexCol, rightCount);
	varLeftX = var(leftIndexRow, leftX, leftCount);
	varLeftY = var(leftIndexCol, leftY, leftCount);
	varRightX = var(rightIndexRow, rightX, rightCount);
	varRightY = var(rightIndexCol, rightY, rightCount);

	motion = 0;
	totalCount = countLeft + countRight;
    printf("1 -> totalCount = %lf leftX = %lf rightX = %lf rightY = %lf leftY = %lf row = %d\n", totalCount, leftX, rightX, rightY, leftY,row);
	if (blinkState == 0)
	{

		if (totalCount >= (double)(0.3*minEyeBlocks) && (leftX >= row / 4) && (rightX >= row / 4) && (rightY - leftY) <= 0.65*row && (rightY - leftY) >= 0.25*row)
		{
			motion = 1;
			prevLeftX = leftX;
			prevLeftY = leftY;
			prevRightX = rightX;
			prevRightY = rightY;
		}
		else
		{

			if (totalCount > 0.9*minEyeBlocks && (leftX >= row / 4 || varLeftX < 2) && (rightX >= row / 4 || varRightX < 2) && (rightY - leftY) <= 0.65*row && (rightY - leftY) >= 0.25*row)
			{
				prevLeftX = leftX;
				prevLeftY = leftY;
				prevRightX = rightX;
				prevRightY = rightY;
				motion = 1;
			}
		}
	}
	else
	{

#if 0
		if ((totalCount> 0.3*minEyeBlocks) && (leftX >= row / 4) && (rightX >= row / 4) && (rightY - leftY) <= 0.65*row && (rightY - leftY) >= 0.25*row)
		{
			motion = 1;
		}
		else
		if ((totalCount > 0.5*minEyeBlocks) && (leftX >= row / 4 || varLeftX < 2) && (rightX >= row / 4 || varRightX < 2) && (rightY - leftY) <= 0.65*row && (rightY - leftY) >= 0.25*row)
		{
			motion = 1;
		}
#else
		leftXD = abs(leftX - prevLeftX);
		leftYD = abs(leftY - prevLeftY);
		rightXD = abs(rightX - prevRightX);
		rightYD = abs(rightY - prevRightY);
		printf("2 ->  leftXD = %lf rightXD = %lf rightYD = %lf leftYD = %lf \n",  leftXD, rightXD, rightYD, leftYD);
		if (leftXD < 1.5 && leftYD < 2 && rightXD < 1.5 && rightYD < 2)
		{
			if (totalCount > 0.5*minEyeBlocks || totalCount > 10)
					motion = 1;
		}
#endif
	}
	/*
	if (blinkState == 0)
	{
		if (totalCount >= (double)(minEyeBlocks)*1.0/ 2 && (leftX >= (((double)(row)*1.0 / 4) || varLeftX < 2)) && ((rightX >= (double)(row) / 4) || varRightX < 2)&& ((rightY <= 3.0 * (double)(row) / 4) || varRightY < 2 )&& ((leftY >= (double)(row) / 4) || varLeftY < 2) && (rightY - leftY) <= (3 * (double)row) / 5 /*&& (rightY - leftY) >= (2 * row) / 5)
		{
			motion = 1;
		}
	}
	else
	{
		if (((leftX >= (double)row / 4) || varLeftX < 2) && ((rightX >= (double)row / 4)||varRightX < 2) && (rightY - leftY) <= (3 * (double)row) / 5 && ((rightY <= 3 * (double)row / 4) || varRightY < 2) && ((leftY >= (double)row / 4) || varLeftY < 2) /*&& (rightY - leftY) >= (2 * row) / 5)
				motion = 1;
	}
	*/
	delete[] sum;

	delete[] leftIndexRow;
	delete[] leftIndexCol;
	delete[] rightIndexRow;
	delete[] rightIndexCol;

	return motion;
}

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

