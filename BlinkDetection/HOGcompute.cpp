#include <opencv2\opencv.hpp>
cv::Mat get_hogdescriptor_visual_image(cv::Mat& origImg,
                                   std::vector<float>& descriptorValues,
                                   cv::Size winSize,
                                   cv::Size cellSize,                                   
                                   int scaleFactor,
                                   double viz_factor);


std::vector<float> computeHOG(cv::Mat image, bool visualize) {

	// HOG parameters
	cv::Size win_size(48, 32);
	cv::Size block_size(12, 8);
	cv::Size block_stride(6, 4);
	cv::Size cell_size(6, 4);
	int nbins = 9;
	double win_sigma = -1; // DEFAULT_WIN_SIGMA
	double threshold_L2hys = 0.2;
	bool gamma_correction = false;
	int nlevels = 16; // DEFAULT_NLEVELS

	cv::HOGDescriptor hog(win_size, block_size, block_stride, 
					  cell_size, nbins, win_sigma, 
					  threshold_L2hys, gamma_correction, nlevels);
	std::vector<float> des;
	std::vector<cv::Point> loc;
	cv::Mat image_gray;
	
	cv::resize(image, image, win_size);
	cv::cvtColor(image, image_gray, CV_BGR2GRAY);
	hog.compute(image_gray, des, cv::Size(0,0), cv::Size(0,0), loc);

	/*std::cout << "HOG descriptor size is " << hog.getDescriptorSize() << std::endl;
	std::cout << "img dimensions: " << image.cols << " width x " << image.rows << "height" << std::endl;
	std::cout << "Found " << des.size() << " descriptor values" << std::endl;*/
	
	if (visualize) {
		cv::Mat visual = get_hogdescriptor_visual_image(image, des, win_size, cell_size, 5, 2);
		cv::imshow("Visualize HOG", visual);
	}

	return des;
}

