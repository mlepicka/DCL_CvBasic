/*!
 * \file
 * \brief
 * \author Tomek Kornuta,,,
 */

#include <memory>
#include <string>

#include "CvSIFT.hpp"
#include "Common/Logger.hpp"

#include <boost/bind.hpp>

#include "opencv2/opencv_modules.hpp"
#include <stdio.h>
#include "opencv2/core/core.hpp"

#include "opencv2/xfeatures2d.hpp"

namespace Processors {
namespace CvSIFT {

CvSIFT::CvSIFT(const std::string & name) :
		Base::Component(name)  {

}

CvSIFT::~CvSIFT() {
}

void CvSIFT::prepareInterface() {
	// Register handlers with their dependencies.
	registerHandler("onNewImage", boost::bind(&CvSIFT::onNewImage, this));
	addDependency("onNewImage", &in_img);

	// Input and output data streams.
	registerStream("in_img", &in_img);
	registerStream("out_features", &out_features);
	registerStream("out_descriptors", &out_descriptors);
}

bool CvSIFT::onInit() {

	return true;
}

bool CvSIFT::onFinish() {
	return true;
}

bool CvSIFT::onStop() {
	return true;
}

bool CvSIFT::onStart() {
	return true;
}

void CvSIFT::onNewImage()
{
	LOG(LTRACE) << "CvSIFT::onNewImage";
	try {
		// Input: a grayscale image.
		cv::Mat input = in_img.read();

		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors = cv::Mat();

		cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create();
		sift->detectAndCompute(input, cv::noArray(), keypoints, descriptors);


/*
		//-- Step 1: Detect the keypoints.
	    cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();
	    std::vector<cv::KeyPoint> keypoints;
	    detector->detect(input, keypoints);

		//-- Step 2: Calculate descriptors (feature vectors).
		cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
		cv::Mat descriptors=cv::Mat();
		extractor->compute(input, keypoints, descriptors);
*/
		// Write results to outputs.

	    Types::Features features(keypoints);
	    
	    LOG(LINFO) << "CvSIFT::FEATURES SIZE "<< features.features.size();
	    LOG(LINFO) << "CvSIFT::DESC SIZE "<< descriptors.total();
	    
	    sift.release();
	    
	    LOG(LINFO) << "CvSIFT::Released ";

		out_features.write(features);
		out_descriptors.write(descriptors);
	} catch (...) {
		LOG(LERROR) << "CvSIFT::onNewImage failed\n";
	}
}


} //: namespace CvSIFT
} //: namespace Processors
