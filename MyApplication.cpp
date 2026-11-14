
#include "Utilities.h"

vector<vector<int>> TRAINING_COORDS = {
	{908, 253},
	{959, 250},
	{1013, 255},
	{1072, 256},
	{1132, 254},
	{1184, 262},
	{1244, 256},
	{1307, 253},
	{797, 293},
	{300, 300}
};

vector<vector<int>> TRAINING_DATA = vector<vector<int>>(TRAINING_COORDS.size(), vector<int>(3, 0));

vector<vector<vector<int>>> GROUND_TRUTH = {
	{
		{1092, 309}, {1212, 429}, {1131, 383}, {1050, 335},
		{1211, 285}, {1170, 261}, {1173, 352}, {1132, 331},
		{1210, 240}, {1170, 310}, {1169, 407}, {1135, 281},
		{1212, 374}, {1091, 360}, {1213, 331}, {29, 288}
	},
	{
		{1103, 282}, {1224, 620}, {1121, 391}, {1046, 315},
		{1305, 310}, {621, 202}, {1173, 361}, {1132, 331},
		{725, 177}, {1269, 185}, {798, 665}, {438, 296},
		{1235, 53}, {1090, 362}, {1205, 399}, {1169, 507}
	},
	{
		{908, 23}, {1224, 616}, {1199, 447}, {29, 336},
		{1406, 305}, {621, 210}, {1294, 260}, {1252, 226},
		{725, 177}, {1270, 171}, {798, 665}, {351, 249},
		{1233, 53}, {901, 169}, {1201, 399}, {29, 434}
	},
	{
		{908, 23}, {1407, 546}, {1204, 444}, {-1, -1},
		{1406, 305}, {621, 210}, {1294, 260}, {1252, 226},
		{724, 181}, {1270, 171}, {798, 665}, {351, 249},
		{1231, 58}, {901, 169}, {1201, 399}, {1392, 88}
	},
	{
		{23, 535}, {1407, 546}, {1204, 444}, {-1, -1},
		{1406, 305}, {25, 131}, {1294, 260}, {1252, 226},
		{72, 538}, {1270, 171}, {798, 665}, {351, 249},
		{-1, -1}, {901, 169}, {1201, 399}, {26, 608}
	},
	{
		{23, 535}, {1364, 477}, {1204, 444}, {-1, -1},
		{1406, 305}, {25, 131}, {1294, 260}, {1252, 226},
		{72, 538}, {1270, 171}, {-1, -1}, {351, 249},
		{-1, -1}, {901, 169}, {1201, 399}, {947, 629}
	}
};

void calculateTrainingData(Mat& trainingImage, vector<vector<int>>& trainingCoords, vector<vector<int>>& trainingData) {
	Mat hsvImage;
	cvtColor(trainingImage, hsvImage, COLOR_BGR2HSV);
	vector<Mat> hsvChannels;
	split(hsvImage, hsvChannels);
	int radius = 21;

	for (int i = 0; i < trainingCoords.size(); i++) {
		int x = trainingCoords[i][0];
		int y = trainingCoords[i][1];

		vector<int> hueBins(180, 0);
		int total = 0;
		int sat = 0;
		int val = 0;

		// Gather Hue, Saturation, and Value data for the training data in the specified radius around the training coordinates
		for (int dx = -radius; dx <= radius; dx++) {
			for (int dy = -radius; dy <= radius; dy++) {
				int nx = x + dx;
				int ny = y + dy;
				if (sqrt(pow(x - nx, 2) + pow(y - ny, 2)) <= radius) {
					if (nx >= 0 && nx < trainingImage.cols && ny >= 0 && ny < trainingImage.rows) {
						total++;
						sat += hsvChannels[1].at<uchar>(ny, nx);
						val += hsvChannels[2].at<uchar>(ny, nx);
						hueBins[hsvChannels[0].at<uchar>(ny, nx)]++;
					}
				}
			}
		}

		// Average out the saturation and value, and find the most common hue in the area
		trainingData[i][1] = sat / total;
		trainingData[i][2] = val / total;

		for (int h = 0; h < 180; h++) {
			if (hueBins[h] > hueBins[trainingData[i][0]]) {
				trainingData[i][0] = h;
			}
		}

	}

}

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


void printMetrics(int tp, int fp, int fn) {
	// Calculate and print out some metrics for performance evaluation of the algorithms
	float accuracy = (float)tp / (tp + fp + fn);
	float precision = (float)tp / (tp + fp);
	float recall = (float)tp / (tp + fn);
	float f1_score = 2 * (precision * recall) / (precision + recall);
	
	cout << endl << "Values" << endl;
	cout << "True Positive: " << tp << endl;
	cout << "False Positive: " << fp << endl;
	cout << "False Negative: " << fn << endl << endl;

	cout << "Metrics" << endl;
	cout << "Accuracy: " << accuracy << endl;
	cout << "Precision: " << precision << endl;
	cout << "Recall: " << recall << endl;
	cout << "F1 Score: " << f1_score << endl << endl;
}

vector<Vec3f> testDetection(vector<Vec3f>& circles, vector<vector<int>>& truth, int radius) {
	vector<bool> matched = vector<bool>(truth.size(), false);
	vector<Vec3f> matches = {};

	int tp = 0;
	int fp = 0;
	int fn = 0;

	for (int i = 0; i < circles.size(); i++) {
		int x = cvRound(circles[i][0]);
		int y = cvRound(circles[i][1]);

		bool false_circle = true;
		for (int j = 0; j < truth.size(); j++) {
			int tx = cvRound(truth[j][0]);
			int ty = cvRound(truth[j][1]);

			if (tx == -1 || ty == -1) continue;

			if (sqrt(pow(tx - x, 2) + pow(ty - y, 2)) <= radius + 2 && !matched[j]) { // Match a circle if it is within the radius of the ground truth and hasn't been matched yet
				matched[j] = true;
				matches.push_back(circles[i]);
				tp++;
				false_circle = false;
				break;
			}
		}
		if (false_circle) fp++;
	}

	for (int m = 0; m < matched.size(); m++) {
		if (!matched[m] && !(truth[m][0] == -1 || truth[m][1] == -1)) {
			cout << "Missed: " << truth[m][0] << " " << truth[m][1] << endl;
			fn++;
		}
	}
	
	cout << "Detection Results" << endl;
	printMetrics(tp, fp, fn);

	return matches;

}



Mat findCircles(Mat& image, int table_num) {
	Mat hsvImage, satImage, blurred, binary, edges, closed, blurredEdges, output;
	image.copyTo(output); // Our output image (to be used later)

	// Our circles vector
	vector<Vec3f> circles;
	int radius = 43 / 2;
	// Convert to hsv
	cvtColor(image, hsvImage, COLOR_BGR2HSV);

	// Split channels and use the Saturation channel
	vector<Mat> hsvChannels;
	split(hsvImage, hsvChannels);
	satImage = hsvChannels[1];

	// Apply Gaussian blur
	GaussianBlur(satImage, blurred, Size(5, 5), 0);

	// Find optimal threshold using Otsu thresholding
	double otsu_thresh = threshold(blurred, binary, 0, 255, THRESH_BINARY | THRESH_OTSU) * 0.6;

	// Apply Canny edge detection to simulate HoughCircles edge detection for development purposes
	//Canny(blurred, edges, otsu_thresh / 2, otsu_thresh, 3, true);


	// Detect and return circles detected based on radius and otsu threshold value
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, radius * 1.9, otsu_thresh, radius * 0.5, radius-1, radius + 4);

	vector<Vec3f> matched = testDetection(circles, GROUND_TRUTH[table_num], radius);
	for (auto& p : circles) {
		circle(output, Point(cvRound(p[0]), cvRound(p[1])), cvRound(p[2]), Scalar(255, 0, 0), 2);
	}

	return output;
}



void MyApplication()
{
	char* file_location = "Media/Pool/";
	char* pool_files[] = {
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

	Mat training_image = imread("Media/Pool/AllBalls.jpeg", -1);
	Mat transformed_training_image = getTransformation(training_image);
	calculateTrainingData(transformed_training_image, TRAINING_COORDS, TRAINING_DATA);

	for(int i = 0; i < TRAINING_DATA.size()-1; i++) {
		cout << "Training Data for Ball " << i + 1 << ": Hue = " << TRAINING_DATA[i][0] << ", Saturation = " << TRAINING_DATA[i][1] << ", Value = " << TRAINING_DATA[i][2] << endl;
	}
	cout << "Training Data for Background" << ": Hue = " << TRAINING_DATA[TRAINING_DATA.size() - 1][0] << ", Saturation = " << TRAINING_DATA[TRAINING_DATA.size() - 1][1] << ", Value = " << TRAINING_DATA[TRAINING_DATA.size() - 1][2] << endl;

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
		Mat circles = findCircles(transformed_image, file_no);
		imshow(pool_files[file_no], circles);
		char pause = cv::waitKey();  // This introduces a delay between frame.  The length of the delay/number can be reduced.
	}

}

