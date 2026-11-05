
#include "Utilities.h"



Mat GetTransformation(Mat& image) {
	Mat transformed;
	
	Point2f src_points[4] = {
		{337, 35},
		{1570, 21},
		{1888, 451},
		{38, 472}
	};

	Point2f dest_points[4] = {
		{0, 0},
		{1433, 0},
		{1433, 682},
		{0, 682}
	};

	Mat transformation = cv::getPerspectiveTransform(src_points, dest_points);

	warpPerspective(image, transformed, transformation, Size(1434, 683));

	return transformed;
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
		Mat transformed_image = GetTransformation(pool_image[file_no]);
		imshow(pool_files[file_no], transformed_image);

		char pause = cv::waitKey();  // This introduces a delay between frame.  The length of the delay/number can be reduced.
	}

}

