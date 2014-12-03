//////////////////////////////////////////////////////////////////////////
//
//  FGExtraction.cpp
//  Date:   Jun/10/2014
//  Author: Meng-Che Chuang, University of Washington
//

#include <iostream>

#include "FGExtraction.h"

//********** class FGExtraction **************************************************

FGExtraction::FGExtraction(double minArea, double maxArea, double minVar, 
                           double pHigh, double pLow,
                           double theta, int nbins,
						   int gradSESize, int areaSESize, int postSESize)
	: _minArea(minArea), _maxArea(maxArea), _minVar(minVar),
      _pHigh(pHigh), _pLow(pLow),
      _theta(theta), _binCount(nbins),
	  _gradSESize(gradSESize), _areaSESize(areaSESize), _postSESize(postSESize)
{
	
}

FGExtraction::~FGExtraction()
{

}

// Extract foreground objects (i.e. segmentation) from input image
//     src - input image
//     dst - output image, binary object mask
//
void FGExtraction::extractForeground(InputArray src, OutputArray dst) 
{
	if(!src.obj) return;
	Mat inImg = src.getMat().clone();
	dst.create(inImg.size(), CV_8U);
	Mat outImg = dst.getMat();
	
    // convert input image to grayscale if it is color
	if(inImg.channels() > 1)
		cvtColor(inImg, inImg, COLOR_BGR2GRAY);
	
    // coarse object localization using morphological gradient
	Mat gradImg;
	Mat se = getStructuringElement(MORPH_ELLIPSE, Size(_gradSESize, _gradSESize));
	morphologyEx(inImg, gradImg, MORPH_GRADIENT, se);
	threshold(gradImg, gradImg, 20, 255, THRESH_BINARY);

	vector<vector<Point>> contours = extractContours(gradImg);
	for (size_t i = 0; i < contours.size(); i++){
		double area = contourArea(contours[i]);
		if(area < _minArea || area > _maxArea){
			drawContours(gradImg, contours, i, Scalar(0), -1);
		}
	}


	// get object contours from coarse localization
	contours.clear();
	contours = extractContours(gradImg);
    
    // for each object, apply double local thresholding and then histogram backprojection
	Mat fgImg = Mat::zeros(gradImg.size(), CV_8U); 
	Mat mask = Mat::zeros(gradImg.size(), CV_8U); 
	for(size_t i = 0; i < contours.size(); ++i)
	{
		// create the "local region" mask
        Rect uprightBox = boundingRect(contours[i]);
		RotatedRect orientedBox = orientedBoundingBox(contours[i]);
		orientedBox.size.width *= 1.5;
        orientedBox.size.height *= 1.5;
		ellipse(mask, orientedBox, Scalar(255), -1);
		
		// double local thresholding
		Mat fgHighImg, fgLowImg;
		doubleLocalThreshold(inImg, fgHighImg, fgLowImg, mask);

		// remove noise by a median filter
		medianBlur(fgHighImg, fgHighImg, 3);
		medianBlur(fgLowImg, fgLowImg, 3);
		
		// merge two masks using histogram backprojection
		updateByHistBackproject(inImg, fgHighImg, fgLowImg, fgImg, mask);

		// remove the used local region from mask
		ellipse(mask, orientedBox, Scalar(0), -1);
	}

	
    // thresholding by area and variance
	thresholdByAreaVar(inImg, fgImg, fgImg);

	// post-processing
    postProcessing(fgImg, fgImg);
	
    // discard connected components with small or large area
	contours.clear();
	contours = extractContours(fgImg);
	for (size_t i = 0; i < contours.size(); i++){
		double area = contourArea(contours[i]);
		if(area < _minArea || area > _maxArea){
			drawOneContour(fgImg, contours[i], Scalar(0), -1);
		}
	}
	
	fgImg.copyTo(outImg);
	return;
}

// Double local thresholding algorithm
//     src     - input grayscale image
//     dstHigh - binary object mask produced by high threshold
//     dstLow  - binary object mask produced by low threshold
//     roiMask - ROI binary mask
//
void FGExtraction::doubleLocalThreshold(InputArray src, OutputArray dstHigh, OutputArray dstLow, Mat roiMask)
{
	if(!src.obj) return;
	Mat inImg = src.getMat();
	dstHigh.create(inImg.size(), CV_8U);
	Mat highFgImg = dstHigh.getMat();
	dstLow.create(inImg.size(), CV_8U);
	Mat lowFgImg = dstLow.getMat();

	int u = 0;
	int thresh = getOtsuThreshold(inImg, 0, 255, &u, roiMask);
	int highThresh = thresh - int(_pHigh*(thresh - u));
	int lowThresh = thresh - int(_pLow*(thresh - u));

	Mat maskedInImg;
	bitwise_and(inImg, roiMask, maskedInImg);
	
	// generate high and low look-up tables
	Mat highLUT(1, 256, CV_8U);
	Mat lowLUT(1, 256, CV_8U);
    uchar* highData = highLUT.data; 
	uchar* lowData = lowLUT.data;
    for(int i = 0; i < 256; ++i){
        highData[i] = i <= highThresh ? 0 : 255;
		lowData[i] = i <= lowThresh ? 0 : 255;
	}

	// threshold by using look-up tables
	LUT(maskedInImg, highLUT, highFgImg);
	LUT(maskedInImg, lowLUT, lowFgImg);
}

// Computes the threshold using Otsu's method
//     src       - input image
//     lowerVal  - lower bound of pixel value
//     upperVal  - upper bound of pixel value
//     u1Ptr     - pointer to receive the mean of lower class
//     roiMask   - ROI binary mask
//
//     returns : Otsu threshold value
//
int FGExtraction::getOtsuThreshold(InputArray src, int lowerVal, int upperVal, int* u1Ptr, Mat roiMask)
{
	if(!src.obj) return -1;
	Mat inImg = src.getMat();
	
	int nbins = 256;
	int channels[] = {0};
	const int histSize[] = {nbins};
    float range[] = {0, 255};
    const float* ranges[] = {range};
	Mat hist;
    cv::calcHist(&inImg, 1, channels, roiMask, hist, 1, histSize, ranges);
	
	Mat_<float> hist_(hist);
	float size = float(sum(hist)[0]);

	float w1, w2, u1, u2;
  	float max = -1;
	int index = 1;
	float u1max = -1;
	float histMax = 0;
	int mode = 0;
	float count = 0;

	for (int i = lowerVal+1; i < upperVal; ++i){	
		if(hist_(i,0) > histMax) {
			histMax = hist_(i,0);
			mode = i;
		}
		w1 = 0;
		
		for (int j = lowerVal+1; j <= i; ++j){
			w1 = w1 + hist_(j-1,0);
		}
		w1 = w1 / size;
		w2 = 1 - w1;

		u1 = 0;
		count = 0;
		for (int j = lowerVal; j <= i-1; ++j){
			u1 = u1 + j*hist_(j,0);
			count += hist_(j,0);
		}
		u1 /= count;

		u2 = 0;
		count = 0;
		for (int j = i; j <= upperVal; ++j){
			u2 = u2 + j*hist_(j, 0);
			count += hist_(j, 0);
		}
		u2 /= count;

		if (w1 * w2 * (u1-u2) * (u1-u2) > max){
			max = w1 * w2 * (u1-u2) * (u1-u2);
			index = i;
			u1max = u1;
		}
		else{
			max = max;
			index = index;
		}
	}
	
	*u1Ptr = (int)(u1max + 0.5);
	return index;
}

// merges low and high FG mask using histogram backprojection
//    src       - input image
//    srcHigh   - high object mask
//    srcLow    - low object mask
//    roiMask   - ROI binary mask
//
void FGExtraction::updateByHistBackproject(InputArray src, InputArray srcHigh, InputArray srcLow, InputOutputArray dst, Mat roiMask)
{
	if(!src.obj || !srcHigh.obj || !srcLow.obj || !dst.obj) return;
	Mat inImg = src.getMat();
	Mat highMask = srcHigh.getMat();
	Mat lowMask = srcLow.getMat();
	Mat fgImg = dst.getMat();

	// generate histograms of two foregrounds
    int channels[] = {0};
	const int histSize[] = {_binCount};
    float range[] = {0, 255};
    const float* ranges[] = {range};
	Mat highHist;
    Mat lowHist;
    cv::calcHist(&inImg, 1, channels, highMask, highHist, 1, histSize, ranges);
	cv::calcHist(&inImg, 1, channels, lowMask, lowHist, 1, histSize, ranges);
	
	// get the ratio histogram
	Mat ratioHist = highHist / lowHist;
	threshold(ratioHist, ratioHist, 1.0, 1.0, THRESH_TRUNC);

	// backproject the ratio histogram to image plane
	Mat ratioHistBP = histBackProject(inImg, ratioHist, roiMask);

	// thresholding on the backprojection
	threshold(ratioHistBP, ratioHistBP, _theta, 255, THRESH_BINARY);
	Mat ratioHistBP_8U;
	ratioHistBP.convertTo(ratioHistBP_8U, CV_8U);
	bitwise_or(ratioHistBP_8U, fgImg, fgImg, roiMask);
}

// Performs histogram backprojection
//    src        - input image
//    hist       - histogram to be backprojected
//    roiMask    - ROI mask
//
//    return : histogram backproject image
//
Mat FGExtraction::histBackProject(InputArray src, Mat hist, Mat roiMask)
{
	if(!src.obj) return Mat();
	Mat inImg = src.getMat();
	int binWidth = hist.rows;

	// generate high and low look-up tables
	Mat lookUpTable(1, 256, CV_32F);
    for(int i = 0; i < 256; ++i)
        lookUpTable.at<float>(0, i) = hist.at<float>(i/binWidth, 0);

	// perform histogram backprojection via the look-up table
	Mat backProj = Mat(inImg.size(), CV_32F);
	LUT(inImg, lookUpTable, backProj);
	
	return backProj;
}

// Selects the foreground objects based on area and variance
//     src   - input grayscale image
//     srcFg - original binary object mask
//     dst   - resultant binary object mask
//
void FGExtraction::thresholdByAreaVar(InputArray src, InputArray srcFg, OutputArray dst)
{
    if(!src.obj || !srcFg.obj) return;
    Mat inImg = src.getMat();
    Mat fgImg = srcFg.getMat();
    dst.create(fgImg.size(), CV_8U);
    Mat outImg = dst.getMat();
    
	// connect separate parts before finding connected components
	Mat closeSE = getStructuringElement(MORPH_ELLIPSE, Size(_areaSESize, _areaSESize));
	morphologyEx(fgImg, outImg, MORPH_CLOSE, closeSE);
    
	// extract contours of targets
    vector<vector<Point>> contours = extractContours(outImg);

	for(size_t i = 0; i < contours.size(); ++i){
		bool passArea = false;
        bool passVar = false;
	
		// check if the area is within the desired range
		double area = contourArea(contours[i]);
        if (area >= _minArea && area <= _maxArea) 
			passArea = true;
		
		// check if the variance of pixel exceeds the threshold
		Mat objFgImg = Mat::zeros(outImg.size(), CV_8U);
        drawContours(objFgImg, contours, i, Scalar(255), -1);
        
        int n = 0;
        double mean = 0;
        double SSD = 0;
        double tempVar = 0;
        
        for(int y = 0; y < objFgImg.rows; ++y){
            for(int x = 0; x < objFgImg.cols; ++x){
                if(objFgImg.at<uchar>(y, x) > 0){
                    ++n;
                    uchar px = inImg.at<uchar>(y, x);
                    double delta = px - mean;
                    mean += delta / n;
                    SSD += delta*(px - mean);
                    tempVar = SSD / (n == 1 ? n : (n-1));
                }
            }
        }

        double var = tempVar;
        passVar = var >= _minVar;
        
		// remove the target if any of the tests fails
		if(!passArea || !passVar){
			drawContours(outImg, contours, i, Scalar(0), -1);
		}
	}
}

// performs post-processing based on morphological operations
//     src  - input image
//     dst  - output image
//
void FGExtraction::postProcessing(InputArray src, OutputArray dst)
{
    if(!src.obj) return;
	Mat inImg = src.getMat();
    dst.create(inImg.size(), inImg.type());
    Mat outImg = dst.getMat();
	
	Mat se = getStructuringElement(MORPH_ELLIPSE, Size(_postSESize, _postSESize));
    
    Mat tempImg;
	morphologyEx(inImg, tempImg, MORPH_CLOSE, se);
	morphologyEx(tempImg, tempImg, MORPH_OPEN, se);
	morphologyEx(tempImg, outImg, MORPH_CLOSE, se);
}
