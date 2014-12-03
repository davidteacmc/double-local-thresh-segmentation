#include "util.h"

RNG rng(54321);

// print type of Mat
void printType(Mat mat)
{
	string r;
	int type = mat.type();
	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch(depth){
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	cout << r;
}

// wrapper to show an image in the window
void showImage(const string& winname, Mat img, int autosize, int delay)
{
	namedWindow(winname, autosize);
	imshow(winname, img);
	waitKey(delay);
}

void putNumOnImage(Mat img, double num, int precision, Point org, int fontScale, Scalar color, int thickness)
{
	if(!img.data)
		return;
	stringstream ss;
	ss << setprecision(precision) << num;
	putText(img, ss.str(), org, FONT_HERSHEY_PLAIN, fontScale, color, thickness);
}

// wrapper to draw one contour in the image
void drawOneContour(Mat img, const vector<Point>& contour, Scalar color, int thickness)
{
	if(!img.data || contour.empty())
		return;
	vector<vector<Point>> contourVec;
	contourVec.push_back(contour);
	drawContours(img, contourVec, -1, color, thickness);
}

// wrapper to find contours in a grayscale image
vector<vector<Point>> extractContours(const Mat& img)
{
	Mat imgC1 = img.clone();
	if(img.channels() != 1)
		cvtColor(img, imgC1, COLOR_BGR2GRAY);
	
	vector<vector<Point> > contours;
	Mat tempImg = imgC1.clone();
	findContours(tempImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	tempImg.release();
	return contours;
}

// convert a binary number to decimal
int binToDec(string number)
{
	int result = 0, pow = 1;
	for(int i = int(number.length()) - 1; i >= 0; --i, pow <<= 1)
		result += (number[i] - '0') * pow;

	return result;
}

// convert a decimal number to binary
string decToBin(int number)
{
	if(number == 0) return "0";
	if(number == 1) return "1";
	if(number % 2 == 0)
		return decToBin(number/2) + "0";
	return decToBin(number/2) + "1";
}

// Gaussian function value given x (zero-mean) and standard deviation
double Gaussian(double x, double stdev)
{
	return exp( -pow(x, 2) / (2*pow(stdev, 2)) ) / stdev / sqrt(2*CV_PI);
}

// generate oriented bounding box
// find the rotation angle by principal component analysis (PCA)
RotatedRect orientedBoundingBox(const vector<Point>& contour)
{
	if(contour.empty())
		return RotatedRect();

	RotatedRect orientedBox;
	if(contour.size() <= 2){
		if(contour.size() == 1){
			orientedBox.center = contour[0];
		}
		else{
			orientedBox.center.x = 0.5f*(contour[0].x + contour[1].x);
			orientedBox.center.y = 0.5f*(contour[0].x + contour[1].x);
			double dx = contour[1].x - contour[0].x;
			double dy = contour[1].y - contour[0].y;
			orientedBox.size.width = (float)sqrt(dx*dx + dy*dy);
			orientedBox.size.height = 0;
			orientedBox.angle = (float)atan2(dy, dx) * 180 / CV_PI;
		}
		return orientedBox;
	}

	Mat data = Mat::zeros(2, contour.size(), CV_32F);
	for(int j = 0; j < contour.size(); ++j){
		data.at<float>(0, j) = contour[j].x;
		data.at<float>(1, j) = contour[j].y;
	}

	// find the principal components
	PCA pcaObj (data, noArray(), CV_DATA_AS_COL);
	Mat result;
	pcaObj.project(data, result);

	// find two endpoints in principal component's direction      
	float maxU = 0, maxV = 0;
	float minU = 0, minV = 0;

	for(int j = 0; j < result.cols; ++j){
		float u = result.at<float>(0, j);
		float v = result.at<float>(1, j);
		if(u > 0 && u > maxU) 
			maxU = u;
		else if(u < 0 && u < minU)
			minU = u;

		if(v > 0 && v > maxV)
			maxV = v;  
		else if(v < 0 && v < minV)
			minV = v;
	}

	float cenU = 0.5*(maxU + minU);
	float cenV = 0.5*(maxV + minV);

	Mat cenUVMat = (Mat_<float>(2, 1) << cenU, cenV);
	Mat cenXYMat = pcaObj.backProject(cenUVMat);

	Point cen(cenXYMat.at<float>(0, 0), cenXYMat.at<float>(1, 0));

	// get width and height of the oriented bounding box
	float width = maxU - minU;
	float height = maxV - minV;

	Mat pc = pcaObj.eigenvectors;

	float pcx = pc.at<float>(0, 0);
	float pcy = pc.at<float>(0, 1);
	float theta = atan2(pcy, pcx) * 180 / 3.1415927;

	// define the oriented bounding box
	orientedBox.center = cen;
	orientedBox.size = Size2f(width, height);
	orientedBox.angle = theta;
	return orientedBox;
}
