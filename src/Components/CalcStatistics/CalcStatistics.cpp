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
		Base::Component(name) {
}

CalcStatistics::~CalcStatistics() {
}

void CalcStatistics::prepareInterface() {
	// Register data streams, events and event handlers HERE!
	registerStream("in_homogMatrix", &in_homogMatrix);
	registerStream("out_homogMatrix", &out_homogMatrix);
	registerStream("in_trigger",&in_trigger);
	registerStream("in_homogMatrix_right", &in_homogMatrix_right);

	// Register handlers
	registerHandler("calculate", boost::bind(&CalcStatistics::calculate,this));
	//addDependency("calculate", &in_homogMatrix);
	addDependency("calculate",&in_trigger);
}

bool CalcStatistics::onInit() {
	//cumulatedHomogMatrix;
	cumulatedRvec = cv::Mat_<double>::zeros(3,1);
	cumulatedTvec = cv::Mat_<double>::zeros(3,1);
	cumulatedAxis = cv::Mat_<double>::zeros(3,1);
	cumulatedFi = 0;
	counter = 0;

	sumDeviationFi=0;
	sumDeviationAxisX=0;
	sumDeviationAxisY=0;
	sumDeviationAxisZ=0;

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
	 std::ofstream plik1, plik2, plik3, plik4;
	 /* sad hardcoded paths :D*/
	 plik1.open("/home/mort/wyniki/wynikiAll.txt", ios::out|ios::app);
	 plik2.open("/home/mort/wyniki/wynikiFi.txt", ios::out|ios::app);
	 plik3.open("/home/mort/wyniki/wynikiTrans.txt", ios::out|ios::app);
	 plik4.open("/home/mort/wyniki/wynikiOdl.txt", ios::out|ios::app);

	in_trigger.read();

	CLOG(LDEBUG)<<"in calculate()";
	plik1 <<"in calculate()" <<endl;

	Types::HomogMatrix homogMatrix;
	Types::HomogMatrix homogMatrix_right;
	
	cv::Mat_<double> rvec;
	cv::Mat_<double> tvec;
	cv::Mat_<double> axis;
	cv::Mat_<double> rotMatrix;
	float fi;
	std::vector<cv::Mat_<double> > sdaxis;
	std::vector<double > sdfi;
	rotMatrix= cv::Mat_<double>::zeros(3,3);
	tvec = cv::Mat_<double>::zeros(3,1);
	axis = cv::Mat_<double>::zeros(3,1);

	CLOG(LINFO)<<"Counter: "<<counter;

	//first homogMatrix on InputStream
	if(counter == 0) {
		stringstream ss;
		CLOG(LINFO)<<"Reading matrixes";
		homogMatrix=in_homogMatrix.read();
		Types::HomogMatrix homogMatrix_right_tmp;
		homogMatrix_right_tmp=in_homogMatrix_right.read();
		homogMatrix_right=homogMatrix_right_tmp;
		homogMatrix_right.matrix()=homogMatrix_right_tmp.matrix().inverse();

		Eigen::Matrix4f actual_trans = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f proper_trans = Eigen::Matrix4f::Identity();
		
		bool identity=true;

		//read matrix
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				actual_trans(i,j)=homogMatrix(i,j);
				proper_trans(i,j)=homogMatrix_right(i,j);
				rotMatrix(i,j)=homogMatrix(i,j);
				if((i==j&&homogMatrix(i,j)!=1)||(i!=j&&homogMatrix(i,j)!=0)){
					identity=false;
					CLOG(LINFO)<<"NOT identity matrix";
					plik1 <<"NOT identity matrix" << endl;
				}

				ss << homogMatrix(i,j) << "  ";
			}
			actual_trans(i,3)= homogMatrix(i,3);
			proper_trans(i,3)= homogMatrix_right(i,3);
			tvec(i, 0) = homogMatrix(i,3);
			if(homogMatrix(i,3)!=0)
				identity=false;
			ss << homogMatrix(i,3) << " \n ";
		}
		if(identity){

			CLOG(LINFO)<<"identity matrix";
			plik1 <<"identity matrix" << endl;
			return;
		}
		CLOG(LINFO) << "homog w static \n" << ss.str();
		plik1 << "homog w static \n" << ss.str() << endl;

		Rodrigues(rotMatrix, rvec);

		cumulatedHomogMatrix = homogMatrix;
		cumulatedTvec = tvec;
		cumulatedRvec = rvec;
		fi = sqrt((pow(rvec(0,0), 2) +
						pow(rvec(1,0), 2)+pow(rvec(2,0),2)));
		float fi_2 = acos((rotMatrix(1,1)+ rotMatrix(2,2)+rotMatrix(3,3)-1)/2);
		cumulatedFi = fi;
		for(int k=0;k<3;k++) {
			axis(k,0)=rvec(k,0)/fi;
		}

		Eigen::Matrix4f diff_trans = Eigen::Matrix4f::Identity();

		diff_trans = (actual_trans - proper_trans).array().abs().matrix();

		CLOG(LINFO) << "aktualny błąd: \n" << diff_trans;
		plik1 << "aktualny błąd: \n" << diff_trans <<endl;
		plik3 << "aktualny błąd: \n" << diff_trans <<";"<<endl;

		//blad kata, ta miara wzieta z gejodezji
		float blad_kat = (float)acos((1-diff_trans(0,0)+1-diff_trans(1,1)+1-diff_trans(2,2)-1)/(double)2);//*(double)180/(double)3.14;

		CLOG(LINFO) << "blad kat: " << blad_kat;
		plik1 << "blad kat: " << blad_kat <<endl;
		plik2 << blad_kat <<endl;


		float blad_odleglosc = sqrt(pow(diff_trans(0,3),2)+pow(diff_trans(1,3),2)+ pow(diff_trans(2,3),2));
		plik1 << "bledy w xyz: " << diff_trans(0,3) << " " << diff_trans(1,3) << " "<< diff_trans(2,3) << " " << std::endl;
		CLOG(LINFO) << "blad odleglosc: " << blad_odleglosc;
		plik1 << "blad odleglosc: " << blad_odleglosc <<endl;
		plik4  << blad_odleglosc  <<endl;

		cumulatedAxis = axis;
		counter=1;
		return;
	}

	homogMatrix=in_homogMatrix.read();
	Types::HomogMatrix homogMatrix_right_tmp;
	homogMatrix_right_tmp=in_homogMatrix_right.read();
	homogMatrix_right=homogMatrix_right_tmp;
	homogMatrix_right.matrix()=homogMatrix_right_tmp.matrix().inverse();

	Eigen::Matrix4f avg_trans = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f proper_trans = Eigen::Matrix4f::Identity();
	
	bool identity=true;
	stringstream ss;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			avg_trans(i,j)=homogMatrix(i,j);
			proper_trans(i,j)=homogMatrix_right(i,j);
			rotMatrix(i,j)=homogMatrix(i,j);
			if((i==j&&homogMatrix(i,j)!=1)||(i!=j&&homogMatrix(i,j)!=0)){
				identity=false;
				CLOG(LINFO)<<"NOT identity matrix";
				plik1 <<"NOT identity matrix" <<endl;
			}

			ss << homogMatrix(i,j) << "  ";
		}
		avg_trans(i,3)= homogMatrix(i,3);
		proper_trans(i,3)=homogMatrix_right(i,3);
		tvec(i, 0) = homogMatrix(i,3);
		if(homogMatrix(i,3)!=0)
			identity=false;
		ss << homogMatrix(i,3) << " \n ";
	}
	if(identity){

		CLOG(LINFO)<<"identity matrix";
		plik1<<"identity matrix" << endl;
		return;
	}

	CLOG(LINFO) << "homog w static \n" << ss.str();
	plik1 << "homog w static \n" << ss.str() << endl;

	Rodrigues(rotMatrix, rvec);
	CLOG(LINFO) << "nowa rotMatrix" << rotMatrix << "\n";
	plik1 << "nowa rotMatrix" << rotMatrix << "\n" << endl;
	CLOG(LINFO)<<"nowy rvec "<<rvec <<"\n";
	plik1 <<"nowy rvec "<<rvec <<"\n" << endl;

	float fi_2 = acos((rotMatrix(1,1)+ rotMatrix(2,2)+rotMatrix(3,3)-1)/2);
	CLOG(LINFO) << "kat a arccos: " << fi_2;
	plik1 << "kat a arccos: " << fi_2 << endl;

	fi = sqrt((pow(rvec(0,0), 2) + pow(rvec(1,0), 2)+pow(rvec(2,0),2)));

	for(int k=0;k<3;k++) {
			axis(k,0)=rvec(k,0)/fi;
	}
	cumulatedFi += fi;
	cumulatedTvec += tvec;
	cumulatedRvec += rvec;
	cumulatedAxis += axis;

	counter++;
	avgFi = cumulatedFi/counter;
	avgAxis = cumulatedAxis/counter;
	avgRvec = avgAxis * avgFi;
	avgTvec = cumulatedTvec/counter;

	Types::HomogMatrix hm;
	cv::Mat_<double> rottMatrix;
	Rodrigues(avgRvec, rottMatrix);

	CLOG(LINFO)<<"Uśredniona macierz z "<<counter<<" macierzy \n";
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			//
			hm(i,j) = rottMatrix(i, j);
			CLOG(LINFO) << hm(i,j) << "  ";
		}
		//avg_trans(i,3)=avgTvec(i, 0);
		hm(i,3) = avgTvec(i, 0);
		CLOG(LINFO) << hm(i,3) <<" \n";
	}
	out_homogMatrix.write(hm);

	CLOG(LINFO)<<"Uśredniony kąt: " << avgFi << "\n";
	plik1<<"Uśredniony kąt: " << avgFi << "\n";
	CLOG(LINFO)<<"Uśredniona oś: " << avgAxis << "\n";
	plik1<<"Uśredniona oś: " << avgAxis << "\n";
	CLOG(LINFO)<<"Uśredniony rvec: " << avgRvec <<"\n";
	plik1<<"Uśredniony rvec: " << avgRvec <<"\n";
	sdaxis.push_back(axis);
	sdfi.push_back(fi);
	sumDeviationFi=0;
	sumDeviationAxisX=0;
	sumDeviationAxisY=0;
	sumDeviationAxisZ=0;

	for(int i=0; i< sdaxis.size(); i++){
		sumDeviationFi+=pow(avgFi - sdfi.at(i), 2);
		sumDeviationAxisX+=pow(avgAxis(0,0) - sdaxis.at(i)(0,0), 2);
		sumDeviationAxisY+=pow(avgAxis(1,0) - sdaxis.at(i)(1,0), 2);
		sumDeviationAxisZ+=pow(avgAxis(2,0) - sdaxis.at(i)(2,0), 2);
	}
	if(counter>1){
		float standardDeviationAxisX = sqrt(sumDeviationAxisX/(double)(counter));
		float standardDeviationAxisY = sqrt(sumDeviationAxisY/(double)(counter));
		float standardDeviationAxisZ = sqrt(sumDeviationAxisZ/(double)(counter));
		float standardDeviationFi = sqrt(sumDeviationFi/(double)(counter));

		CLOG(LINFO)<<"Odchylenie standardowe kąta: "<< standardDeviationFi << "\n";
		plik1<<"Odchylenie standardowe kąta: "<< standardDeviationFi << "\n";
		CLOG(LINFO)<<"Odchylenie standardowe osi w X: "<< standardDeviationAxisX << "\n";
		plik1<<"Odchylenie standardowe osi w X: "<< standardDeviationAxisX << "\n";
		CLOG(LINFO)<<"Odchylenie standardowe osi w Y: "<< standardDeviationAxisY << "\n";
		plik1<<"Odchylenie standardowe osi w Y: "<< standardDeviationAxisY << "\n";
		CLOG(LINFO)<<"Odchylenie standardowe osi w Z: "<< standardDeviationAxisZ<< "\n";
		plik1<<"Odchylenie standardowe osi w Z: "<< standardDeviationAxisZ<< "\n";

		Eigen::Matrix4f diff_trans = Eigen::Matrix4f::Identity();

		diff_trans = (avg_trans - proper_trans).array().abs().matrix();

		CLOG(LINFO) << "aktualny błąd: \n" << diff_trans;
		plik1 << "aktualny błąd: \n" << diff_trans <<endl;
		plik3 << "aktualny błąd: \n" << diff_trans <<";"<<endl;

		float blad_kat = (float)acos((1-diff_trans(0,0)+1-diff_trans(1,1)+1-diff_trans(2,2)-1)/(double)2);//*(double)180/(double)3.14;

		CLOG(LINFO) << "blad kat: " << blad_kat;
		plik1 << "blad kat: " << blad_kat <<endl;
		plik2 << blad_kat <<endl;


		float blad_odleglosc = sqrt(pow(diff_trans(0,3),2)+pow(diff_trans(1,3),2)+ pow(diff_trans(2,3),2));
		plik1 << "bledy w xyz: " << diff_trans(0,3) << " " << diff_trans(1,3) << " "<< diff_trans(2,3) << " " << endl;
		CLOG(LINFO) << "blad odleglosc: " << blad_odleglosc;
		plik1 << "blad odleglosc: " << blad_odleglosc <<endl;
		plik4  << blad_odleglosc  <<endl;

	}
	plik1.close();
	plik2.close();
	plik3.close();
	plik4.close();
}


} //: namespace CalcStatistics
} //: namespace Processors
