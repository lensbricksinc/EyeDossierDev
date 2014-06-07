#include "ml_core.h"
#include <opencv2/opencv.hpp>


void BlinkDetectorML::MLInit()
{
	if (!isReset)
		MLReset();

	historyMLFeatures = new float* [ML_HISTORY_SIZE];
	for (int i=0; i< ML_HISTORY_SIZE; i++)
	{
		historyMLFeatures[i] = new float [ML_FEATURES_PER_FRAME];
	}

	isReset = false;
	
	return;
}

void BlinkDetectorML::MLInitFeatureDumpFile()
{
	mOutFileForTraining.open(fileNameDataDump, 'w');

	if (mOutFileForTraining.is_open())
	{
		mOutFileForTraining << "STATE:Vectors," << ML_HISTORY_SIZE<<"," <<ML_FEATURES_PER_FRAME<< std::endl;
	}

}


void BlinkDetectorML::MLReset()
{
	/*
	if (blinkSVM != NULL)
	{
		blinkSVM->clear();
		delete blinkSVM;
		blinkSVM = NULL;
	}
	*/

	/*
	if (mOutFileForTraining.is_open())
	{
		mOutFileForTraining.flush();
		mOutFileForTraining.close();
	}
	*/

	if (historyMLFeatures!= nullptr)
	{
		for (int i=0; i< ML_HISTORY_SIZE; i++)
		{
			if (historyMLFeatures[i] != nullptr)
			{
				delete[] historyMLFeatures[i];
				historyMLFeatures[i] = nullptr;
			}
		}
		delete[] historyMLFeatures;
		historyMLFeatures = nullptr;
	}
	mCurrHead=0;
	mNumValues=0;
	isReset = true;
}

BlinkDetectorML::BlinkDetectorML(int mNumFeatures)
	{
		ML_FEATURES_PER_FRAME = mNumFeatures;
		mCurrHead = 0;
		mNumValues = 0;
		historyMLFeatures = nullptr;
		blinkSVM = NULL;

		MLReset();
		blinkSVM = new CvSVM();

		std::stringstream ssfileNameDataDump;
		ssfileNameDataDump <<  "C:\\Users\\chintak\\Desktop\\testcases\\manualmarkings\\train_data\\pranav01_" <<mNumFeatures <<".csv";
		fileNameDataDump = ssfileNameDataDump.str();

		std::stringstream ssfileNameDataDumpPCA;
		ssfileNameDataDumpPCA <<  "C:\\Users\\chintak\\Desktop\\testcases\\manualmarkings\\train_data\\pranav01_pca_" <<mNumFeatures <<".csv";
		fileNameDataDumpPCA = ssfileNameDataDumpPCA.str();

		std::stringstream ssfileNameCSVDump;
		ssfileNameCSVDump <<  "C:\\Users\\chintak\\Desktop\\testcases\\manualmarkings\\train_data\\pranav01_rbf_" <<mNumFeatures <<".xml";
		fileNameCSVDump = ssfileNameCSVDump.str();


		
		
		//MLInitFeatureDumpFile();
	};


void BlinkDetectorML::MLAddEntry(float* featureEntries)
{
	if (isReset)
		MLInit();

	if (featureEntries != NULL)
	{
		// Add entry at mCurrHead
		memcpy(historyMLFeatures[mCurrHead] , featureEntries, ML_FEATURES_PER_FRAME * sizeof(float));
		mNumValues = MIN(mNumValues + 1, ML_HISTORY_SIZE);
		mCurrHead++;
		if (mCurrHead == ML_HISTORY_SIZE)
			mCurrHead = 0;
	}
}

void BlinkDetectorML::MLDumpFeaturesToFileForTraining(int state)
{
	if (mNumValues == ML_HISTORY_SIZE)
	{
		if (!mOutFileForTraining.is_open())
			MLInitFeatureDumpFile();

		mOutFileForTraining << state << ":" ;
		int k= mCurrHead;
		for (int i=0; i<ML_HISTORY_SIZE-1; i++)
		{
			k--;
			if (k < 0)
				k = ML_HISTORY_SIZE-1;
			for (int j=0 ; j<ML_FEATURES_PER_FRAME; j++)
			{
				mOutFileForTraining << historyMLFeatures[k][j] << ",";
			}
			
		}

		k--;
		if (k < 0)
			k = ML_HISTORY_SIZE-1;

		for (int j=0; j<ML_FEATURES_PER_FRAME-1; j++)
			mOutFileForTraining << historyMLFeatures[k][j] << ",";
		mOutFileForTraining << historyMLFeatures[k][ML_FEATURES_PER_FRAME-1] << std::endl;
	}
}


void BlinkDetectorML::MLTrain()
{

	std::ifstream fileInputTraining;
	fileInputTraining.open(fileNameDataDump);
	int numEntries = 0;
	std::string unused;
	std::getline(fileInputTraining, unused);			// Discard the first line which contains the header
	while ( std::getline(fileInputTraining, unused) )
	{
		if (!unused.empty())
			++numEntries;
	}
	fileInputTraining.close();
#ifdef PCA_MODE
	std::string line;

	fileInputTraining.open(fileNameDataDump);
	std::getline(fileInputTraining, line);			// Discard the first line which contains the header

	std::vector<cv::Mat> matArray;
	int indexRow = 0;
	while ( std::getline(fileInputTraining, line) )
	{
		
		if (!line.empty())
		{
			int locColon = line.find(":");
			int state = atoi(line.substr(0,locColon).c_str());

			if (state == 1)
			{
				std::stringstream lineStream(line.substr(locColon+1,-1));	// Substr till end of string
				std::string lineSubString;
				cv::Mat entry = cv::Mat::zeros(ML_FEATURES_PER_FRAME*ML_HISTORY_SIZE,1, CV_32FC1);
				int indexCol = 0;
				while (std::getline(lineStream, lineSubString, ','))
				{
					if (!lineSubString.empty()){
						float entryVal = atof(lineSubString.c_str());
						//printf("Row=%d, Col=%d\n", indexRow, indexCol);
						entry.at<float>(indexCol,0) = entryVal;
						//printf("%f,", entryVal);
						indexCol++;
					}
				}
				if (indexCol != ML_FEATURES_PER_FRAME*ML_HISTORY_SIZE)
				{
					printf("Error in input parameters for training\n");
					while(1);
				}
				matArray.push_back(entry);
			}
			indexRow++;
		}
		
	}
	fileInputTraining.close();

	int componentsVec = ML_FEATURES_PER_FRAME*ML_HISTORY_SIZE;
	int numVecs = matArray.size();
	cv::Mat pcaInput = cv::Mat(componentsVec, numVecs, CV_32FC1);

	for (int i=0; i< numVecs; i++)
	{
		cv::Mat entryPoint = pcaInput.col(i);
		matArray[i].copyTo(entryPoint);
	}

	int numPrincipalComponents = 5;
	// Do the PCA:
    cv::PCA pca(pcaInput, cv::Mat(), CV_PCA_DATA_AS_COL, numPrincipalComponents);

	std::cout << "Eigenvalues:\n" << pca.eigenvalues << std::endl;
	cv::Mat weights = pca.eigenvalues.clone();
	weights = (weights)/(cv::sum(weights)[0,0]);

	std::cout << "Weights:\n" << weights << std::endl;

	std::string filename = "D:\\pcaeigenvectors.xml";
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	fs << "pcaeigenvectors" << pca.eigenvectors;			// numPrincipalComponenets x componentsVec
	fs << "pcaeigenvalues" << pca.eigenvalues;
	fs << "weights" << weights;
	fs << "pcameans" << pca.mean;
	fs.release();

	cv::Mat labelsMat = cv::Mat::zeros(numEntries, 1, CV_32FC1);
	labelsMat = labelsMat - 1;		// Initialize to -1 by default. Should be treated as an invalid state.
	cv::Mat trainingDataMat = cv::Mat::zeros(numEntries, numPrincipalComponents, CV_32FC1);

	fileInputTraining.open(fileNameDataDump);
	std::getline(fileInputTraining, line);

	std::ofstream fileOutDataDumpPCA(fileNameDataDumpPCA);
	fileOutDataDumpPCA << "STATE:Vectors," << numPrincipalComponents << std::endl;
	indexRow = 0;
	while ( std::getline(fileInputTraining, line) )
	{
		if (!line.empty())
		{
			int locColon = line.find(":");
			int state = atoi(line.substr(0,locColon).c_str());
			labelsMat.at<float>(indexRow,0) = (float)state;

			fileOutDataDumpPCA << state <<":";

			std::stringstream lineStream(line.substr(locColon+1,-1));	// Substr till end of string
			std::string lineSubString;
			cv::Mat entryPreProcessed = cv::Mat::zeros(ML_FEATURES_PER_FRAME*ML_HISTORY_SIZE,1, CV_32FC1);
			int indexCol = 0;
			while (std::getline(lineStream, lineSubString, ','))
			{
				if (!lineSubString.empty()){
					float entryVal = atof(lineSubString.c_str());
					entryPreProcessed.at<float>(indexCol, 0) = entryVal;
					indexCol++;
				}
			}
			if (indexCol != ML_FEATURES_PER_FRAME*ML_HISTORY_SIZE)
			{
				printf("Error in input parameters for training\n");
				while(1);
			}
			
			// Make the entry after projecting it according to PCA vectors.
			cv::Mat entryPostProcessed = pca.project(entryPreProcessed);
			entryPostProcessed.reshape(1,1).copyTo(trainingDataMat.row(indexRow));		// Reshape to contain only one row. reshape contains only new header files for cv::Mat.

 			double valNorm = (cv::norm(entryPostProcessed)/cv::norm(entryPreProcessed - pca.mean))*100.0;

			for (int i=0; i< entryPostProcessed.rows; i++)
			{
				fileOutDataDumpPCA << entryPostProcessed.at<float>(i,0) << ",";
			}
			fileOutDataDumpPCA <<":"<< valNorm;
			fileOutDataDumpPCA << std::endl;
			indexRow++;
		}
		
	}

	fileOutDataDumpPCA.flush();
	fileOutDataDumpPCA.close();
	fileInputTraining.close();
	if (indexRow != numEntries)
	{
		printf("Error in input parameters for training\n");
		while(1);
	}
#else
	std::string line;
	fileInputTraining.open(fileNameDataDump);
	std::getline(fileInputTraining, line);			// Discard the first line which contains the header

	cv::Mat labelsMat = cv::Mat::zeros(numEntries, 1, CV_32FC1);
	labelsMat = labelsMat - 1;		// Initialize to -1 by default. Should be treated as an invalid state.
	cv::Mat trainingDataMat = cv::Mat::zeros(numEntries, ML_FEATURES_PER_FRAME*ML_HISTORY_SIZE, CV_32FC1);

	int indexRow = 0;
	while ( std::getline(fileInputTraining, line) )
	{
		int indexCol = 0;
		if (!line.empty())
		{
			int locColon = line.find(":");
			int state = atoi(line.substr(0,locColon).c_str());
			labelsMat.at<float>(indexRow,0) = (float)state;
			std::stringstream lineStream(line.substr(locColon+1,-1));	// Substr till end of string
			std::string entry;
			while (std::getline(lineStream, entry, ','))
			{
				if (!entry.empty()){
					float entryVal = atof(entry.c_str());
					//printf("Row=%d, Col=%d\n", indexRow, indexCol);
					trainingDataMat.at<float>(indexRow, indexCol) = entryVal;
					//printf("%f,", entryVal);
					indexCol++;
				}
			}
			//printf("\n");
			if (indexCol != ML_FEATURES_PER_FRAME*ML_HISTORY_SIZE)
			{
				printf("Error in input parameters for training\n");
				while(1);
			}
			indexRow++;
		}
		
	}
	fileInputTraining.close();
	if (indexRow != numEntries)
	{
		printf("Error in input parameters for training\n");
		while(1);
	}
#endif
	

	CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;		// enum { C_SVC=100, NU_SVC=101, ONE_CLASS=102, EPS_SVR=103, NU_SVR=104 };
    params.kernel_type = CvSVM::RBF;		// enum { LINEAR=0, POLY=1, RBF=2, SIGMOID=3 };
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	params.degree = 15;			// Rrequired only for POLY kernel type
	//params.coef0				// Can be set for POLY type and SIGMOID type
	// SVM params type -- enum { C=0, GAMMA=1, P=2, NU=3, COEF=4, DEGREE=5 };

	//CvSVM SVM;
    //SVM.train(trainingDataMat, labelsMat, cv::Mat(), cv::Mat(), params);

	
    // Train the SVM
    blinkSVM->train(trainingDataMat, labelsMat, cv::Mat(), cv::Mat(), params);
	blinkSVM->save(fileNameCSVDump.c_str());
	// Make a large matrix which will store all these values to be feeded for training
	
}

int BlinkDetectorML::MLPredict()
{
	int retState=-1;
	if( 0 == blinkSVM->get_support_vector_count())
		blinkSVM->load(fileNameCSVDump.c_str());
#ifdef PCA_MODE
	cv::Mat pcaEigenVectors;
	cv::Mat pcaEigenValues;
	cv::Mat pcaWeights;
	cv::Mat pcaMeans;
	std::string filename = "D:\\pcaeigenvectors.xml";
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	fs["pcaeigenvectors"] >> pcaEigenVectors;
	fs["pcaeigenvalues"] >> pcaEigenValues;
	fs["weights"] >> pcaWeights;
	fs["pcameans"] >> pcaMeans;
	fs.release();
#endif
	if (mNumValues == ML_HISTORY_SIZE)
	{
		cv::Mat currentFeatureVecMat = cv::Mat::zeros(1, ML_FEATURES_PER_FRAME*ML_HISTORY_SIZE, CV_32FC1);
		int k= mCurrHead;
		for (int i=0; i<ML_HISTORY_SIZE; i++)
		{
			k--;
			if (k < 0)
				k = ML_HISTORY_SIZE-1;
			for (int j=0 ; j<ML_FEATURES_PER_FRAME; j++)
			{
			    currentFeatureVecMat.at<float>(0,i*ML_FEATURES_PER_FRAME + j) = historyMLFeatures[k][j];
			}
			
		}
		double valNorm = (cv::norm((currentFeatureVecMat - pcaMeans.t()) * pcaEigenVectors.t())/cv::norm(currentFeatureVecMat - pcaMeans.t()))*100.0;
		std::cout << "Percentage= " << valNorm << " %"<< std::endl;
#ifdef PCA_MODE
		retState = blinkSVM->predict((currentFeatureVecMat - pcaMeans.t()) * pcaEigenVectors.t());
#else
		retState = blinkSVM->predict(currentFeatureVecMat);
#endif
	}
	return retState;
}

