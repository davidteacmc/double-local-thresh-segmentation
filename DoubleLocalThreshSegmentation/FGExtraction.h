//////////////////////////////////////////////////////////////////////////
//
//  FGExtraction.h
//  Date:   Jun/10/2014
//  Author: Meng-Che Chuang, University of Washington
//
//  For details of the algorithm, please refer to this paper:
//
//  M.-C. Chuang et al., "Automatic fish segmentation via double local
//  thresholding for trawl-based underwater cameras," Proc. of IEEE 
//  International Conference on Image Processing, Sep. 2011.

#ifndef _FGEXTRACTION_H_
#define _FGEXTRACTION_H_

#include <iostream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "util.h"

using namespace std;
using namespace cv;

//********** class FGExtraction **************************************************

class FGExtraction
{
public:
	FGExtraction(double minArea, double maxArea, double minVar, 
				 double pHigh, double pLow, 
                 double theta, int nbins,
				 int gradSESize, int areaSESize, int postSESize);
	~FGExtraction();

	// object segmentation method
	void extractForeground(InputArray inImg, OutputArray fgImg);

private:
	double	_minArea;
	double	_maxArea;
    double	_minVar;
	double  _pHigh;
	double  _pLow;
    double  _theta;
    double  _binCount;
	int     _gradSESize;
    int     _areaSESize;
	int     _postSESize;
	
	// double local thresholding methods
	void doubleLocalThreshold(InputArray src, OutputArray dstHigh, OutputArray dstLow, Mat roiMask);
	int getOtsuThreshold(InputArray src, int lowerVal, int upperVal, int* u1Ptr, Mat roiMask);

	// histogram backprojection methods
	void updateByHistBackproject(InputArray src, InputArray srcHigh, InputArray srcLow, InputOutputArray dst, Mat roiMask);
	Mat histBackProject(InputArray src, Mat hist, Mat roiMask);

	// threshold by area and variance
	void thresholdByAreaVar(InputArray src, InputArray srcFg, OutputArray dst);

	// post processing
	void postProcessing(InputArray src, OutputArray dst);
};

#endif
