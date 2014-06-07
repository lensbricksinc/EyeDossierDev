


#ifndef __ML_CORE_H__
#define __ML_CORE_H__

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#define PCA_MODE

class BlinkDetectorML {

private:
	static const int ML_HISTORY_SIZE = 10;
	int ML_FEATURES_PER_FRAME;
	float **historyMLFeatures;
	int mCurrHead;
	int mNumValues;
	std::ofstream mOutFileForTraining;
	bool isReset;
	CvSVM *blinkSVM;
	std::string fileNameDataDump;
	std::string fileNameCSVDump;
	std::string fileNameDataDumpPCA;
	void MLInitFeatureDumpFile();

public:
	void MLInit();
		//Machine learning
	void MLReset();
	void MLAddEntry(float* featureEntries);
	void MLTrain();
	int MLPredict();
	void MLDumpFeaturesToFileForTraining(int state);

	BlinkDetectorML(int mNumFeatures);
};

#endif

