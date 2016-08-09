/*
 * Line.hpp
 *
 *  Created on: 09-12-2010
 *      Author: mateusz
 */

#ifndef LINE_HPP_
#define LINE_HPP_

#include "Drawable.hpp"
#include <cmath>
#include <opencv2/imgproc.hpp>
    #include <opencv2/highgui.hpp>
namespace Types {

class Line: public Types::Drawable
{
public:
	Line(cv::Point p1, cv::Point p2) :
		p1(p1), p2(p2)
	{
		computeLineEquation();
	}

	Line(const Line& o)
	{
		p1 = o.p1;
		p2 = o.p2;
		lineEquationComputed = o.lineEquationComputed;
		A = o.A;
		B = o.B;
		C = o.C;
	}

	virtual ~Line()
	{
	}

	virtual Line * clone()
	{	Line * ret = new Line(*this);
		ret->setCol(m_col);
		return ret;
	}

	virtual void draw(cv::Mat& image, cv::Scalar color, int offsetX = 0, int offsetY = 0)
	{
		cv::line(image, p1, p2, getCol(), 2);
	}

	double getDistanceFromPoint(cv::Point p)
	{
		//		if (!lineEquationComputed) {
		//			computeLineEquation();
		//		}
		return fabs(A * p.x + B * p.y + C) / sqrt(A * A + B * B);
	}

	// returns angle in range (-pi/2, pi/2]
	double getAngle() {
		if (B == 0)
			return M_PI_2;
		else
			return atan(A);
	}

	double length() {
		return cv::norm(p1 - p2);
	}

	cv::Point getP1(){
		return p1;
	}
	cv::Point getP2(){
		return p2;
	}
	
	cv::Point intersect(Line * other) {
		cv::Point x = other->p1 - p1;
    cv::Point d1 = p2 - p1;
    cv::Point d2 = other->p2 - other->p1;
    cv::Point r;

    float cross = d1.x*d2.y - d1.y*d2.x;
   
		/* parallel case */
    /*if (abs(cross) < 1e-8) return false; */
    
    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = p1 + d1 * t1;
    return r;
	}

private:

	void computeLineEquation()
	{
		/* y	- p2.y	=	((p2.y-p1.y)/(p2.x-p1.x))*(x - p1.x)*/

		/*Ax+By+C=0 - zle*/
		/*-Ax+By-C=0*/
		/*0=Ax-By+C*/
		if (p1.x == p2.x && p1.y != p2.y)//rownelegla do osi OY
		{
			A = -1.0;
			B = 0.0;
			C = p1.x;
		} else if (p1.y == p2.y && p1.x != p2.x) //rownolegla do osi OX
		{
			A = 0.0;
			B = -1.0;
			C = p1.y;

		} else {
			A = (p2.y - p1.y) / (double) (p2.x - p1.x);
			B = -1.0;
			C = A * (-p1.x) + p1.y;
		}
		lineEquationComputed = true;
	}

	cv::Point p1;
	cv::Point p2;

	bool lineEquationComputed;
	double A, B, C;
};

}

#endif /* LINE_HPP_ */
