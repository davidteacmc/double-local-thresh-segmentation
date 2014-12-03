//////////////////////////////////////////////////////////////////////////
//
//  test_main.cpp
//  Date:   Jun/10/2014
//  Author: Meng-Che Chuang, University of Washington
//
//  This file serves only as a test script for the double local thresholding
//  algorithm, rather than the main function of a executable program.
//
//  All runtime arguments are taken via the command-line interface.
//

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "util.h"
#include "FGExtraction.h"

//********** main functions **************************************************************************

int main(int argc, char** argv)
{
	// read input image
    string filename = "14860.jpg";
	Mat inImg = imread(filename, 0);
	Mat fgImg = Mat::zeros(inImg.size(), CV_8U);

	// set parameters
	double minArea = 1000;
	double maxArea = inImg.rows * inImg.cols;
	double minVar = 30;
	double pHigh = 0.7;
	double pLow = 1;
	double theta = 0.3;
	int nbins = 16;
	int gradSESize = 5;
	int areaSESize = 7;
	int postSESize = 5;

	// apply object segmentation
	FGExtraction segMgr(minArea, maxArea, minVar, pHigh, pLow, theta, nbins, gradSESize, areaSESize, postSESize);
	segMgr.extractForeground(inImg, fgImg);

	// show and save the result
	imshow("input", inImg);
	imshow("segmentation", fgImg);
	waitKey(0);
	filename = "seg_" + filename;
	imwrite(filename, fgImg);

	return 0;
}