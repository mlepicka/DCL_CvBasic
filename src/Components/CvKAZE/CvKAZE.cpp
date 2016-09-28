#include <memory>
#include <string>

#include "CvKAZE.hpp"
#include "Common/Logger.hpp"
#include "Common/Timer.hpp"
#include <sstream>
#include <fstream>
#include <boost/bind.hpp>

#include <opencv2/features2d.hpp>
#include "Types/KAZE.hpp"
#include "Types/AKAZE.hpp"

namespace Processors {
namespace CvKAZE {

CvKAZE::CvKAZE(const std::string & name) :
		Base::Component(name), prop_calc_path("Calculations.path",
				std::string("")), prop_diffusivity("Diffusivity", 1) {
	registerProperty(prop_calc_path);
	registerProperty(prop_diffusivity);
}

CvKAZE::~CvKAZE() {
}

void CvKAZE::prepareInterface() {
	// Register handlers with their dependencies.
	registerHandler("calculate_features", boost::bind(&CvKAZE::calculate_features, this));
	addDependency("calculate_features", &in_img);

	// Input and output data streams.
	registerStream("in_img", &in_img);
	registerStream("out_features", &out_features);
	registerStream("out_descriptors", &out_descriptors);
}

bool CvKAZE::onInit() {

	return true;
}

bool CvKAZE::onFinish() {
	return true;
}

bool CvKAZE::onStop() {
	return true;
}

bool CvKAZE::onStart() {
	return true;
}

void CvKAZE::calculate_features() {
	//tu property czy kaze czy akaze
	//kaze nie extended

	LOG(LTRACE)<< "CvKAZE::calculate_features\n";
	try {
		// Input: a grayscale image.
		cv::Mat input = in_img.read();

		std::ofstream feature_calc_time;
		if(!string(prop_calc_path).empty()) {
			feature_calc_time.open((string(prop_calc_path)+string("czas_wyznaczenia_cech_kaze.txt")).c_str(), ios::out|ios::app);
		}
		std::vector<cv::KeyPoint> keypoints;
		Mat descriptors;
		Common::Timer timer;

		timer.restart();
		Ptr<AKAZE> kaze = AKAZE::create(prop_diffusivity); //create(false);
		kaze->detectAndCompute(input, cv::noArray(), keypoints, descriptors);

		if(!string(prop_calc_path).empty()) {
			feature_calc_time << timer.elapsed() << endl;
		}
		// Write results to outputs.
		Types::Features features(keypoints, "KAZE");

		LOG(LTRACE) << "CvKaze:: features found: " << features.features.size() << " descriptors found: " << descriptors.size() << "\n";
		//features.type = "KAZE";
		LOG(LTRACE) <<"FEATURES TYPE: " << features.type;
		out_features.write(features);
		out_descriptors.write(descriptors);

	} catch (...) {
		LOG(LERROR) << "CvKAZE::onNewImage failed\n";
	}
}

} //: namespace CvKAZE
} //: namespace Processors
