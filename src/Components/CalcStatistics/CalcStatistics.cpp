/*!
 * \file
 * \brief
 * \author Marta Lepicka
 */

#include <memory>
#include <string>
#include <cmath>

#include "CalcStatistics.hpp"
#include "Common/Logger.hpp"
#include <stdlib.h>
#include <boost/bind.hpp>
#include "Types/Objects3D/Object3D.hpp"
#include "Types/HomogMatrix.hpp"
#include "Types/CameraInfo.hpp"
#include <Common/Timer.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <sstream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <Types/HomogMatrix.hpp>
namespace Processors {
namespace CalcStatistics {

CalcStatistics::CalcStatistics(const std::string & name) :
		Base::Component(name), prop_calc_path("Calculations_path",std::string(".")) {
	registerProperty(prop_calc_path);
}

CalcStatistics::~CalcStatistics() {
}

void CalcStatistics::prepareInterface() {
	// Register data streams, events and event handlers HERE!
	registerStream("in_homog_matrix", &in_homog_matrix);
	registerStream("out_homog_matrix", &out_homog_matrix);
	registerStream("in_trigger",&in_trigger);
	registerStream("in_homog_matrix_right", &in_homog_matrix_right);

	// Register handlers
	registerHandler("calculate", boost::bind(&CalcStatistics::calculate,this));
	//addDependency("calculate", &in_homogMatrix);
	addDependency("calculate",&in_trigger);
}

bool CalcStatistics::onInit() {
	//cumulatedHomogMatrix;
	cumulated_rvec = cv::Mat_<double>::zeros(3,1);
	cumulated_tvec = cv::Mat_<double>::zeros(3,1);
	cumulated_axis = cv::Mat_<double>::zeros(3,1);
	cumulated_fi = 0;
	counter = 0;

	sum_deviation_fi=0;
	sum_deviation_axis_X=0;
	sum_deviation_axis_Y=0;
	sum_deviation_axis_Z=0;

	return true;
}

bool CalcStatistics::onFinish() {
	return true;
}

bool CalcStatistics::onStop() {
	return true;
}

bool CalcStatistics::onStart() {
	return true;
}

/**
 * a lot of unecessary computes
 */
void CalcStatistics::calculate() {
	 int i;
	 std::ofstream output_all, error_fi, error_trans, error_dist;

	 output_all.open((string(prop_calc_path)+string("output.txt")).c_str(), ios::out|ios::app);
	 error_fi.open((string(prop_calc_path)+string("blad_kata_fi.txt")).c_str(), ios::out|ios::app);
	 error_trans.open((string(prop_calc_path)+string("blad_translacji.txt")).c_str(), ios::out|ios::app);
	 error_dist.open((string(prop_calc_path)+string("blad_odleglosci.txt")).c_str(), ios::out|ios::app);

	in_trigger.read();

	CLOG(LDEBUG)<<"in calculate()";
	output_all <<"in calculate()" <<endl;

	Types::HomogMatrix homog_matrix;
	Types::HomogMatrix homog_matrix_right;
	
	cv::Mat_<double> rvec;
	cv::Mat_<double> tvec;
	cv::Mat_<double> axis;
	cv::Mat_<double> rot_matrix;
	float fi;
	std::vector<cv::Mat_<double> > sdaxis;
	std::vector<double > sdfi;
	rot_matrix= cv::Mat_<double>::zeros(3,3);
	tvec = cv::Mat_<double>::zeros(3,1);
	axis = cv::Mat_<double>::zeros(3,1);

	CLOG(LINFO)<<"Counter: "<<counter;

	//first homogMatrix on InputStream
	if(counter == 0) {
		stringstream ss;
		CLOG(LINFO)<<"Reading matrixes";
		homog_matrix=in_homog_matrix.read();
		Types::HomogMatrix homog_matrix_right_tmp;
		homog_matrix_right_tmp=in_homog_matrix_right.read();
		homog_matrix_right=homog_matrix_right_tmp;
		homog_matrix_right.matrix()=homog_matrix_right_tmp.matrix().inverse();

		Eigen::Matrix4f actual_trans = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f proper_trans = Eigen::Matrix4f::Identity();
		
		bool identity=true;

		//read matrix
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				actual_trans(i,j)=homog_matrix(i,j);
				proper_trans(i,j)=homog_matrix_right(i,j);
				rot_matrix(i,j)=homog_matrix(i,j);
				if((i==j&&homog_matrix(i,j)!=1)||(i!=j&&homog_matrix(i,j)!=0)){
					identity=false;
					CLOG(LINFO)<<"NOT identity matrix";
					output_all <<"NOT identity matrix" << endl;
				}

				ss << homog_matrix(i,j) << "  ";
			}
			actual_trans(i,3)= homog_matrix(i,3);
			proper_trans(i,3)= homog_matrix_right(i,3);
			tvec(i, 0) = homog_matrix(i,3);
			if(homog_matrix(i,3)!=0)
				identity=false;
			ss << homog_matrix(i,3) << " \n ";
		}
		if(identity){

			CLOG(LINFO)<<"identity matrix";
			output_all <<"identity matrix" << endl;
			return;
		}
		CLOG(LINFO) << "homog w static \n" << ss.str();
		output_all << "homog w static \n" << ss.str() << endl;

		Rodrigues(rot_matrix, rvec);

		cumulated_homog_matrix = homog_matrix;
		cumulated_tvec = tvec;
		cumulated_rvec = rvec;
		fi = sqrt((pow(rvec(0,0), 2) +
						pow(rvec(1,0), 2)+pow(rvec(2,0),2)));
		float fi_2 = acos((rot_matrix(1,1)+ rot_matrix(2,2)+rot_matrix(3,3)-1)/2);
		cumulated_fi = fi;
		for(int k=0;k<3;k++) {
			axis(k,0)=rvec(k,0)/fi;
		}

		Eigen::Matrix4f diff_trans = Eigen::Matrix4f::Identity();

		diff_trans = (actual_trans - proper_trans).array().abs().matrix();

		CLOG(LINFO) << "aktualny błąd: \n" << diff_trans;
		output_all << "aktualny błąd: \n" << diff_trans <<endl;
		error_trans << "aktualny błąd: \n" << diff_trans <<";"<<endl;

		//blad kata, ta miara wzieta z gejodezji
		float blad_kat = (float)acos((1-diff_trans(0,0)+1-diff_trans(1,1)+1-diff_trans(2,2)-1)/(double)2);//*(double)180/(double)3.14;

		CLOG(LINFO) << "blad kat: " << blad_kat;
		output_all << "blad kat: " << blad_kat <<endl;
		error_fi << blad_kat <<endl;


		float blad_odleglosc = sqrt(pow(diff_trans(0,3),2)+pow(diff_trans(1,3),2)+ pow(diff_trans(2,3),2));
		output_all << "bledy w xyz: " << diff_trans(0,3) << " " << diff_trans(1,3) << " "<< diff_trans(2,3) << " " << std::endl;
		CLOG(LINFO) << "blad odleglosc: " << blad_odleglosc;
		output_all << "blad odleglosc: " << blad_odleglosc <<endl;
		error_dist  << blad_odleglosc  <<endl;

		cumulated_axis = axis;
		counter=1;
		return;
	}

	homog_matrix=in_homog_matrix.read();
	Types::HomogMatrix homogMatrix_right_tmp;
	homogMatrix_right_tmp=in_homog_matrix_right.read();
	homog_matrix_right=homogMatrix_right_tmp;
	homog_matrix_right.matrix()=homogMatrix_right_tmp.matrix().inverse();

	Eigen::Matrix4f avg_trans = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f proper_trans = Eigen::Matrix4f::Identity();
	
	bool identity=true;
	stringstream ss;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			avg_trans(i,j)=homog_matrix(i,j);
			proper_trans(i,j)=homog_matrix_right(i,j);
			rot_matrix(i,j)=homog_matrix(i,j);
			if((i==j&&homog_matrix(i,j)!=1)||(i!=j&&homog_matrix(i,j)!=0)){
				identity=false;
				CLOG(LINFO)<<"NOT identity matrix";
				output_all <<"NOT identity matrix" <<endl;
			}

			ss << homog_matrix(i,j) << "  ";
		}
		avg_trans(i,3)= homog_matrix(i,3);
		proper_trans(i,3)=homog_matrix_right(i,3);
		tvec(i, 0) = homog_matrix(i,3);
		if(homog_matrix(i,3)!=0)
			identity=false;
		ss << homog_matrix(i,3) << " \n ";
	}
	if(identity){

		CLOG(LINFO)<<"identity matrix";
		output_all<<"identity matrix" << endl;
		return;
	}

	CLOG(LINFO) << "homog w static \n" << ss.str();
	output_all << "homog w static \n" << ss.str() << endl;

	Rodrigues(rot_matrix, rvec);
	CLOG(LINFO) << "nowa rotMatrix" << rot_matrix << "\n";
	output_all << "nowa rotMatrix" << rot_matrix << "\n" << endl;
	CLOG(LINFO)<<"nowy rvec "<<rvec <<"\n";
	output_all <<"nowy rvec "<<rvec <<"\n" << endl;

	float fi_2 = acos((rot_matrix(1,1)+ rot_matrix(2,2)+rot_matrix(3,3)-1)/2);
	CLOG(LINFO) << "kat a arccos: " << fi_2;
	output_all << "kat a arccos: " << fi_2 << endl;

	fi = sqrt((pow(rvec(0,0), 2) + pow(rvec(1,0), 2)+pow(rvec(2,0),2)));

	for(int k=0;k<3;k++) {
			axis(k,0)=rvec(k,0)/fi;
	}
	cumulated_fi += fi;
	cumulated_tvec += tvec;
	cumulated_rvec += rvec;
	cumulated_axis += axis;

	counter++;
	avg_fi = cumulated_fi/counter;
	avg_axis = cumulated_axis/counter;
	avg_rvec = avg_axis * avg_fi;
	avg_tvec = cumulated_tvec/counter;

	Types::HomogMatrix hm;
	cv::Mat_<double> rottMatrix;
	Rodrigues(avg_rvec, rottMatrix);

	CLOG(LINFO)<<"Uśredniona macierz z "<<counter<<" macierzy \n";
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			//
			hm(i,j) = rottMatrix(i, j);
			CLOG(LINFO) << hm(i,j) << "  ";
		}
		//avg_trans(i,3)=avgTvec(i, 0);
		hm(i,3) = avg_tvec(i, 0);
		CLOG(LINFO) << hm(i,3) <<" \n";
	}
	out_homog_matrix.write(hm);

	CLOG(LINFO)<<"Uśredniony kąt: " << avg_fi << "\n";
	output_all<<"Uśredniony kąt: " << avg_fi << "\n";
	CLOG(LINFO)<<"Uśredniona oś: " << avg_axis << "\n";
	output_all<<"Uśredniona oś: " << avg_axis << "\n";
	CLOG(LINFO)<<"Uśredniony rvec: " << avg_rvec <<"\n";
	output_all<<"Uśredniony rvec: " << avg_rvec <<"\n";
	sdaxis.push_back(axis);
	sdfi.push_back(fi);
	sum_deviation_fi=0;
	sum_deviation_axis_X=0;
	sum_deviation_axis_Y=0;
	sum_deviation_axis_Z=0;

	for(int i=0; i< sdaxis.size(); i++){
		sum_deviation_fi+=pow(avg_fi - sdfi.at(i), 2);
		sum_deviation_axis_X+=pow(avg_axis(0,0) - sdaxis.at(i)(0,0), 2);
		sum_deviation_axis_Y+=pow(avg_axis(1,0) - sdaxis.at(i)(1,0), 2);
		sum_deviation_axis_Z+=pow(avg_axis(2,0) - sdaxis.at(i)(2,0), 2);
	}
	if(counter>1){
		float standard_deviation_axis_X = sqrt(sum_deviation_axis_X/(double)(counter));
		float standard_deviation_axis_Y = sqrt(sum_deviation_axis_Y/(double)(counter));
		float standard_deviation_axis_Z = sqrt(sum_deviation_axis_Z/(double)(counter));
		float standard_deviation_fi = sqrt(sum_deviation_fi/(double)(counter));

		CLOG(LINFO)<<"Odchylenie standardowe kąta: "<< standard_deviation_fi << "\n";
		output_all<<"Odchylenie standardowe kąta: "<< standard_deviation_fi << "\n";
		CLOG(LINFO)<<"Odchylenie standardowe osi w X: "<< standard_deviation_axis_X << "\n";
		output_all<<"Odchylenie standardowe osi w X: "<< standard_deviation_axis_X << "\n";
		CLOG(LINFO)<<"Odchylenie standardowe osi w Y: "<< standard_deviation_axis_Y << "\n";
		output_all<<"Odchylenie standardowe osi w Y: "<< standard_deviation_axis_Y << "\n";
		CLOG(LINFO)<<"Odchylenie standardowe osi w Z: "<< standard_deviation_axis_Z<< "\n";
		output_all<<"Odchylenie standardowe osi w Z: "<< standard_deviation_axis_Z<< "\n";

		Eigen::Matrix4f diff_trans = Eigen::Matrix4f::Identity();

		diff_trans = (avg_trans - proper_trans).array().abs().matrix();

		CLOG(LINFO) << "aktualny błąd: \n" << diff_trans;
		output_all << "aktualny błąd: \n" << diff_trans <<endl;
		error_trans << "aktualny błąd: \n" << diff_trans <<";"<<endl;

		float blad_kat = (float)acos((1-diff_trans(0,0)+1-diff_trans(1,1)+1-diff_trans(2,2)-1)/(double)2);//*(double)180/(double)3.14;

		CLOG(LINFO) << "blad kat: " << blad_kat;
		output_all << "blad kat: " << blad_kat <<endl;
		error_fi << blad_kat <<endl;


		float blad_odleglosc = sqrt(pow(diff_trans(0,3),2)+pow(diff_trans(1,3),2)+ pow(diff_trans(2,3),2));
		output_all << "bledy w xyz: " << diff_trans(0,3) << " " << diff_trans(1,3) << " "<< diff_trans(2,3) << " " << endl;
		CLOG(LINFO) << "blad odleglosc: " << blad_odleglosc;
		output_all << "blad odleglosc: " << blad_odleglosc <<endl;
		error_dist  << blad_odleglosc  <<endl;

	}
	output_all.close();
	error_fi.close();
	error_trans.close();
	error_dist.close();
}


} //: namespace CalcStatistics
} //: namespace Processors
