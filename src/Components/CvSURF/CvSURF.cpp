/*!
 * \file
 * \brief
 * \author Tomek Kornuta,,,
 */

#include <memory>
#include <string>

#include "CvSURF.hpp"
#include "Common/Logger.hpp"

#include <boost/bind.hpp>
# include "opencv2/xfeatures2d/nonfree.hpp"

namespace Processors {
namespace CvSURF {

CvSURF::CvSURF(const std::string & name) :
		Base::Component(name),
        minHessian("minHessian", 400, "minHessian")
{
	// Register properties.
	registerProperty(minHessian);
}

CvSURF::~CvSURF() {
}

void CvSURF::prepareInterface() {
	// Register handlers with their dependencies.
	registerHandler("onNewImage", boost::bind(&CvSURF::onNewImage, this));
	addDependency("onNewImage", &in_img);

	// Input and output data streams.
	registerStream("in_img", &in_img);
	registerStream("out_features", &out_features);
	registerStream("out_descriptors", &out_descriptors);
}

bool CvSURF::onInit() {

	return true;
}

bool CvSURF::onFinish() {
	return true;
}

bool CvSURF::onStop() {
	return true;
}

bool CvSURF::onStart() {
	return true;
}

void CvSURF::onNewImage()
{
	LOG(LTRACE) << "CvSURF::onNewImage\n";
	try {
		// Input: a grayscale image.
		cv::Mat input = in_img.read();


		//-- Step 1: Detect the keypoints using SURF Detector.
		cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> detector = cv::xfeatures2d::SurfFeatureDetector::create(minHessian);
	
		
		std::vector<KeyPoint> keypoints;
		detector->detect( input, keypoints );

		//-- Step 2: Calculate descriptors (feature vectors).
        cv::Ptr<cv::xfeatures2d::SurfDescriptorExtractor> extractor;
		Mat descriptors;
		extractor->compute( input, keypoints, descriptors);

		// Write features to the output.
	    Types::Features features(keypoints);
		out_features.write(features);

		// Write descriptors to the output.
		out_descriptors.write(descriptors);
	} catch (...) {
		LOG(LERROR) << "CvSURF::onNewImage failed\n";
	}
}



} //: namespace CvSURF
} //: namespace Processors
