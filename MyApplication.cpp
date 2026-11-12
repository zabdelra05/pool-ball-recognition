
#include "Utilities.h"



Mat getTransformation(Mat& image) {
	Mat transformed;
	
	// Warped image dimensions
	float height = 683;
	float width = 1434;

	Point2f src_points[4] = { // Table corner source points (from brief)
		{337, 35},
		{1570, 21},
		{1888, 451},
		{38, 472}
	};

	Point2f dest_points[4] = { // Destination points for the transformed image
		{0, 0},
		{width - 1, 0},
		{width - 1, height - 1},
		{0, height - 1}
	};

	// Compute the perspective transformation matrix and apply it to the image
	Mat transformation = cv::getPerspectiveTransform(src_points, dest_points);
	warpPerspective(image, transformed, transformation, Size(width, height));

	return transformed;
}

Mat findEdges(Mat& image) {
	Mat hsvImage, satImage, blurred, binary, edges, closed;
	// Convert to hsv
	cvtColor(image, hsvImage, COLOR_BGR2HSV);

	// Split channels and use the Saturation channel
	vector<Mat> hsvChannels;
	split(hsvImage, hsvChannels);
	satImage = hsvChannels[1];

	// Apply Gaussian blur
	GaussianBlur(satImage, blurred, Size(5, 5), 0);

	// Find optimal threshold using Otsu thresholding
	double otsu_thresh = threshold(blurred, binary, 0, 255, THRESH_BINARY | THRESH_OTSU) * 0.8;

	// Apply Canny edge detection
	Canny(blurred, edges, otsu_thresh / 3, otsu_thresh, 3, true);

	// Perform a closing morphological operation on the resulting edges
	morphologyEx(edges, closed, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
	return edges;
}

void MyApplication()
{
	char* file_location = "Media/Pool/";
	char* pool_files[] = {
		"AllBalls.jpeg",
		"Table1.jpeg",
		"Table2.jpeg",
		"Table3.jpeg",
		"Table4.jpeg",
		"Table5.jpeg",
		"Table6.jpeg"
	};

	// Load images
	int number_of_images = sizeof(pool_files) / sizeof(pool_files[0]);
	Mat* pool_image = new Mat[number_of_images];
	//Mat* new_pool_image = new Mat[number_of_images];
	for (int file_no = 0; (file_no < number_of_images); file_no++)
	{
		cout << pool_files[file_no] << endl;
		string filename(file_location);
		filename.append(pool_files[file_no]);
		pool_image[file_no] = imread(filename, -1);

		if (pool_image[file_no].empty())
		{
			cout << "Could not open " << pool_image[file_no] << endl;
		}
		Mat transformed_image = getTransformation(pool_image[file_no]);
		Mat edges = findEdges(transformed_image);
		imshow(pool_files[file_no], transformed_image);
		imshow("Edges", edges);
		char pause = cv::waitKey();  // This introduces a delay between frame.  The length of the delay/number can be reduced.
	}

}

