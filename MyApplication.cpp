
#include "Utilities.h"

vector<vector<int>> TRAINING_COORDS = {
	{908, 253},
	{959, 250},
	{1013, 255},
	{1072, 256},
	{1132, 254},
	{1184, 262},
	{1244, 256},
	{1307, 253}, // White Ball
	{797, 293}, // Black Ball

	// Background Data (random non-ball samples)
	{300, 300},
	{30, 60},
	{100, 25},
	{715, 30},
	{1410, 50}
};


vector<vector<int>> TRAINING_DATA = vector<vector<int>>(TRAINING_COORDS.size(), vector<int>(3, 0)); // Training data for each ball and background containing Hue, Saturation, and Value

vector<vector<vector<int>>> GROUND_TRUTH = { // Ground truth coordinates for each table image
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
vector<vector<float>> RESULTS(3, vector<float>(4, 0)); // Cumulative accuracy, precision, recall and F1 scores for each task across all pictures (to be averaged)

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


void printMetrics(int tp, int fp, int fn, int resultIndex) {
	// Calculate and print out some metrics for performance evaluation of the algorithms
	float accuracy = (float)tp / (tp + fp + fn);
	float precision = (float)tp / (tp + fp);
	float recall = (float)tp / (tp + fn);
	float f1_score = 2 * (precision * recall) / (precision + recall);

	RESULTS[resultIndex][0] += accuracy;
	RESULTS[resultIndex][1] += precision;
	RESULTS[resultIndex][2] += recall;
	RESULTS[resultIndex][3] += f1_score;

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

vector<Vec3f> testDetection(vector<Vec3f>& circles, vector<vector<int>>& truth) {
	vector<bool> matched = vector<bool>(truth.size(), false);
	vector<Vec3f> matches = {};

	int tp = 0;
	int fp = 0;
	int fn = 0;

	for (int i = 0; i < circles.size(); i++) {
		int x = cvRound(circles[i][0]);
		int y = cvRound(circles[i][1]);
		int radius = cvRound(circles[i][2]);

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

	for (int m = 0; m < matched.size(); m++) { // Count false negatives for any unmatched ground truth circles
		if (!matched[m] && !(truth[m][0] == -1 || truth[m][1] == -1)) {
			cout << "Missed: " << truth[m][0] << " " << truth[m][1] << endl;
			fn++;
		}
	}
	
	cout << endl << "Circle Detection Results" << endl;
	printMetrics(tp, fp, fn, 0);

	return matches;

}

void testClassification(vector<Vec3f>& balls, vector<vector<int>>& truth, int resultIndex) {
	int tp = 0;
	int fp = 0;
	int fn = 0;

	for (int i = 0; i < balls.size(); i++) {

		int x = cvRound(balls[i][0]);
		int y = cvRound(balls[i][1]);
		int radius = cvRound(balls[i][2]);

		int tx = cvRound(truth[i][0]);
		int ty = cvRound(truth[i][1]);

		if ((x != -1 || y != -1) && (tx == -1 || ty == -1)) { // False positive, detected a ball that doesn't exist
			fp++;
			continue;
		}
		else if ((x == -1 || y == -1) && (tx != -1 || ty != -1)) { // False negative, failed to detect a ball that exists
			fn++;
			continue;
		}

		if (tx == -1 || ty == -1) continue;

		if (sqrt(pow(tx - x, 2) + pow(ty - y, 2)) <= radius + 2)  // Match if the correct ball centre within threshold distance of the detected ball centre
			tp++;
		else
			fp++;
	}

	printMetrics(tp, fp, fn, resultIndex);
}

int hueDifference(int hue1, int hue2) { // Considers wrap-around distance value for Hue
	int diff = abs(hue1 - hue2);
	return min(diff, 180 - diff);
}


struct Ball { // Ball struct used to store information about a circle for classification
	Vec3f position;
	float blackRatio; // Black pixel ratio
	float whiteRatio; // White pixel ratio
	float bgRatio; // Background pixel ratio
	int maxHue; // Most frequent Hue
	float avgValue; // Avg Lighting (for segregating similar colours such as red and maroon)
	Ball(int x, int y, int radius, float blackRatio, float whiteRatio, float bgRatio, int maxHue, float avgValue)
		: position(x, y, radius), blackRatio(blackRatio), whiteRatio(whiteRatio), bgRatio(bgRatio), maxHue(maxHue), avgValue(avgValue) {
	}
};

vector<Ball> evaluateBalls(vector<Vec3f>& circles, vector<vector<int>>& trainingData, Mat& image) {
	Mat hsvImage;
	cvtColor(image, hsvImage, COLOR_BGR2HSV);
	vector<Mat> hsvChannels;
	split(hsvImage, hsvChannels);

	vector<Ball> balls;

	for (auto& circle : circles) {
		int x = cvRound(circle[0]);
		int y = cvRound(circle[1]);
		int radius = cvRound(circle[2]);

		vector<int> hueBins(180, 0);
		int blackPixels = 0;
		int whitePixels = 0;
		int bgPixels = 0;
		int totalPixels = 0;

		// Used for brightness comparison between red and maroon balls
		int totalValue = 0;
		int sumValue = 0;

		for (int dx = -radius + 1; dx <= radius - 1; dx++) {
			for (int dy = -radius + 1; dy <= radius - 1; dy++) {
				int nx = x + dx;
				int ny = y + dy;
				if (sqrt(pow(x - nx, 2) + pow(y - ny, 2)) <= radius - 1) { // Extract pixels from circle and their HSV values
					if (nx >= 0 && nx < hsvImage.cols && ny >= 0 && ny < hsvImage.rows) {
						totalPixels++;
						int hue = hsvChannels[0].at<uchar>(ny, nx);
						int sat = hsvChannels[1].at<uchar>(ny, nx);
						int val = hsvChannels[2].at<uchar>(ny, nx);

						if (abs(sat - trainingData[7][1]) <= 40 && abs(val - trainingData[7][2]) <= 40) // Match with black ball S and V
							blackPixels++;
						else if (abs(sat - trainingData[8][1]) <= 40 && abs(val - trainingData[8][2]) <= 40) // Match with white ball S and V
							whitePixels++;
						else { // Ignore pixels matching background signature
							bool isBg = false;
							for (int bg = 9; bg < trainingData.size(); bg++) {
								if (abs(trainingData[bg][2] - val) <= 30 && abs(trainingData[bg][1] - sat) <= 30 && hueDifference(hue, trainingData[bg][0]) <= 10) {
									isBg = true;
									bgPixels++;
									break;
								}
							} // Add hue to relevant bin if S and V are sufficient
							if (!isBg && sat >= 80 && val >= 40) {
								hueBins[hue]++;
								totalValue++;
								sumValue += val;
							}
								
						}
					}
				}
			}
		}

		// Calculate ratios
		float blackRatio = (float)blackPixels / totalPixels;
		float whiteRatio = (float)whitePixels / totalPixels;
		float bgRatio = (float)bgPixels / totalPixels;
		float avgValue = (float)sumValue / totalValue;

		int maxHue = 0; // Find most frequent Hue
		for (int h = 0; h < 180; h++) {
			if (hueBins[h] > hueBins[maxHue]) {
				maxHue = h;
			}
		}

		balls.push_back(Ball(x, y, radius, blackRatio, whiteRatio, bgRatio, maxHue, avgValue));
	}
	return balls;
}

vector<Vec3f> classifyBalls(vector<Ball>& balls, vector<vector<int>>& trainingData) {
	vector<Vec3f> classifiedBalls(16, Vec3f(-1, -1, -1)); // Initialise balls as not found
	Vec3f whiteBall, blackBall;

	vector<vector<Ball>> ballGroups(7); // Ball colour groups
	float whiteBallRatio = 0.0f, blackBallRatio = 0.0f;
	int hueThreshold = 10;

	for (auto& ball : balls) {
		//cout << ball.position[0] << " " << ball.position[1] << " " << ball.whiteRatio << " " << ball.avgValue << endl;
		if (ball.whiteRatio > whiteBallRatio) { // Find whitest and blackest balls to classify as white and black balls before finding coloured balls
			whiteBall = ball.position;
			whiteBallRatio = ball.whiteRatio;
		}
		if (ball.blackRatio > blackBallRatio) {
			blackBall = ball.position;
			blackBallRatio = ball.blackRatio;
		}
	}

	//cout << "Black Ball: " << blackBall[0] << ", " << blackBall[1] << ", " << blackBall[2] << endl;
	//cout << "White Ball: " << whiteBall[0] << ", " << whiteBall[1] << ", " << whiteBall[2] << endl;

	classifiedBalls[7] = blackBall;
	classifiedBalls[15] = whiteBall;

	for (auto& ball : balls) {
		if (ball.position == whiteBall || ball.position == blackBall || ball.bgRatio >= 0.4) { // Ignore white and black balls and circles with too much background
			//cout << "Ignoring ball at (" << ball.position[0] << ", " << ball.position[1] << ") due to background ratio or being white/black ball." << endl;
			continue;
		}
			
		int group = -1;
		int minHueDiff = 180;
		for (int i = 0; i < 7; i++) {
			int hueDiff = hueDifference(ball.maxHue, trainingData[i][0]); // Find closest matching colour
			 
			if (hueDiff <= hueThreshold && hueDiff < minHueDiff) {
				group = i;
				minHueDiff = hueDiff;
			}

		}
		if (group == 2 || group == 6) { // Differentiate between red and maroon based on average brightness
			float redDist = abs(ball.avgValue - trainingData[2][2]);
			float maroonDist = abs(ball.avgValue - trainingData[6][2]);

			group = (redDist < maroonDist) ? 2 : 6;
			ballGroups[group].push_back(ball);
		}
		else if (group != -1) // No matching group found, ignore the ball
			ballGroups[group].push_back(ball);
	}

	for (int i = 0; i < 7; i++) {
		if (ballGroups[i].empty()) continue;

		sort(ballGroups[i].begin(), ballGroups[i].end(), [&trainingData, i](const Ball& a, const Ball& b) { // Sort based on hue difference to training data
			float distA = hueDifference(a.maxHue, trainingData[i][0]);
			float distB = hueDifference(b.maxHue, trainingData[i][0]);

			return distA < distB;
		});

		if (ballGroups[i].size() >= 2) {
			sort(ballGroups[i].begin(), ballGroups[i].begin() + 2, [](const Ball& a, const Ball& b) { // Larger white ratio is more likely to be the striped ball
				return a.whiteRatio < b.whiteRatio;
			});
			if (ballGroups[i].size() > 2) {
				//cout << "Adding: " << ballGroups[i][0].position[0] << ", " << ballGroups[i][0].position[1] << " to group " << i << endl;
				//cout << "Adding: " << ballGroups[i].back().position[0] << ", " << ballGroups[i].back().position[1] << " to group " << i << endl;

				for (int j = 1; j < ballGroups[i].size() - 1; j++) {
					
					//cout << "Dropping (" << ballGroups[i][j].position[0] << ", " << ballGroups[i][j].position[1] << ", " << ballGroups[i][j].position[2] << ") from group" << i << " Max Hue: " << ballGroups[i][j].maxHue <<  endl;
				}
			}
			
			classifiedBalls[i] = ballGroups[i][0].position;
			classifiedBalls[i + 8] = ballGroups[i][1].position;
		}
		else if (ballGroups[i].size() == 1) {
			if (ballGroups[i][0].whiteRatio > 0.3f) { // Stripe whiteness threshold
				classifiedBalls[i + 8] = ballGroups[i][0].position;
			}
			else {
				classifiedBalls[i] = ballGroups[i][0].position;
			}
		}
	}

	return classifiedBalls;
}


vector<Mat> findCircles(Mat& image, int table_num) {
	Mat hsvImage, satImage, blurred, binary, edges, closed, blurredEdges, output1, output2, output3;
	image.copyTo(output1); // Detection matrix
	image.copyTo(output2); // Matched classification matrix
	image.copyTo(output3); // Raw Classification matrix

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

	// Find automated threshold using Otsu thresholding
	double otsu_thresh = threshold(blurred, binary, 0, 255, THRESH_BINARY | THRESH_OTSU) * 0.6;

	// Apply Canny edge detection to simulate HoughCircles edge detection for development purposes
	//Canny(blurred, edges, otsu_thresh / 2, otsu_thresh, 3, true);


	// Detect and return circles detected based on radius and otsu threshold value
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, radius * 1.9, otsu_thresh, radius * 0.45, radius-1, radius + 4);

	vector<Vec3f> matched = testDetection(circles, GROUND_TRUTH[table_num]);
	vector<Vec3f> classifiedRaw = classifyBalls(evaluateBalls(circles, TRAINING_DATA, image), TRAINING_DATA);
	vector<Vec3f> classifiedMatched = classifyBalls(evaluateBalls(matched, TRAINING_DATA, image), TRAINING_DATA);

	cout << "Matched Circles Classification Results" << endl;
	testClassification(classifiedMatched, GROUND_TRUTH[table_num], 1);

	cout << "Raw Circles Classification Results" << endl;
	testClassification(classifiedRaw, GROUND_TRUTH[table_num], 2);

	// Draw relevant text and circles on appropriate output matrices for each task
	for (auto& p : circles) {
		if (p[0] == -1 || p[1] == -1 || p[2] == -1) continue;
		circle(output1, Point(cvRound(p[0]), cvRound(p[1])), cvRound(p[2]), Scalar(255, 0, 0), 2);
	}

	for (int i = 0; i < classifiedMatched.size(); i++) {
		auto& p = classifiedMatched[i];
		if (p[0] == -1 || p[1] == -1 || p[2] == -1) continue;
		putText(output2, to_string(i+1), Point(cvRound(p[0]) - 10, cvRound(p[1]) + 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 2);
		circle(output2, Point(cvRound(p[0]), cvRound(p[1])), cvRound(p[2]), Scalar(255, 0, 0), 2);
	}

	for (int i = 0; i < classifiedRaw.size(); i++) {
		auto& p = classifiedRaw[i];
		if (p[0] == -1 || p[1] == -1 || p[2] == -1) continue;
		putText(output3, to_string(i+1), Point(cvRound(p[0]) - 10, cvRound(p[1]) + 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 2);
		circle(output3, Point(cvRound(p[0]), cvRound(p[1])), cvRound(p[2]), Scalar(255, 0, 0), 2);
	}
	
	vector<Mat> output = { output1, output2, output3 };

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

	for(int i = 0; i < 8; i++) 
		cout << "Training Data for Ball " << i + 1 << ": Hue = " << TRAINING_DATA[i][0] << ", Saturation = " << TRAINING_DATA[i][1] << ", Value = " << TRAINING_DATA[i][2] << endl;
	
	cout << endl;

	for (int i = 8; i < TRAINING_DATA.size(); i++)
		cout << "Training Data for Background Sample " << i - 7 << ": Hue = " << TRAINING_DATA[i][0] << ", Saturation = " << TRAINING_DATA[i][1] << ", Value = " << TRAINING_DATA[i][2] << endl;

	//cout << "Training Data for Background" << ": Hue = " << TRAINING_DATA[TRAINING_DATA.size() - 1][0] << ", Saturation = " << TRAINING_DATA[TRAINING_DATA.size() - 1][1] << ", Value = " << TRAINING_DATA[TRAINING_DATA.size() - 1][2] << endl;

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
		vector<Mat> circles = findCircles(transformed_image, file_no);
		imshow("Detections", circles[0]);
		imshow("Matched Classifications", circles[1]);
		imshow("Raw Classifications", circles[2]);
		char pause = cv::waitKey();  // This introduces a delay between frame.  The length of the delay/number can be reduced.
	}

	// Performance metrics averaged over all images for each task
	for (int i = 0; i < RESULTS.size(); i++) {
		string type = "";
		if (i == 0) type = "Detection";
		else if (i == 1) type = "Matched Classification";
		else if (i == 2) type = "Raw Classification";

		cout << "Results for " << type << ":" << endl;
		cout << "Average Accuracy: " << RESULTS[i][0] / number_of_images << endl;
		cout << "Average Precision: " << RESULTS[i][1] / number_of_images << endl;
		cout << "Average Recall: " << RESULTS[i][2] / number_of_images << endl;
		cout << "Average F1 Score: " << RESULTS[i][3] / number_of_images << endl;
		cout << endl;
	}
}