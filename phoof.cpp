#include "opencv2/opencv.hpp"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <iterator>

#include "svm.h"
#include "svm.cpp"

#include <wiringPi.h>
#include <softPwm.h>

using namespace cv;
using namespace std;

//#define _USE_CVSVM

#define F_COLS 128
#define F_ROWS 128
#define OF_WIN_PI 10
#define OF_WIN_PI2 7
#define OF_WIN 4

vector<Mat> collection;
VideoCapture cap(0);

/*		IMPORTANT PIN-OUTS

RPI:
24 - BLACK
26 - GREEN
-----------
15 - BLACK
16 - WHITE
-----------
if you get this wrong, may the force be with you.
*/

void init()
{
	if( wiringPiSetup() == -1 )
		cout<<"wiring error";
	softPwmCreate(10, 0, 100);
	pinMode(11, OUTPUT);
	pinMode(3, OUTPUT);
	softPwmCreate(4, 0, 100);
}

void differentialDrive( int l, int r )
{
	softPwmWrite(10, l);
	digitalWrite(11, LOW);
	digitalWrite(3, LOW);
	softPwmWrite(4, r);
}

void un_init()
{
	digitalWrite(11, LOW);
	digitalWrite(3, LOW);
	softPwmWrite(10, 0);
	softPwmWrite(4, 0);
}

inline float slope_uni( Point2f &P1, Point2f &P2 )
{
	float x = std::atan( (P2.y-P1.y)/(P2.x-P1.x) )*180/CV_PI;
	if( isnan(x) ) x=0;
	x+=90;
	return x;
}
inline float slope( Point2f &P1, Point2f &P2 )
{
/*
	float m = std::atan2( P2.y-P1.y, P2.x-P1.x )*180/CV_PI;
	if( isnan(m) ) m=0;
	if( m<0 ) m+=360;
	return m;
*/
	return cv::fastAtan2( P2.y-P1.y, P2.x-P1.x );
}

inline float euclideanDist( Point2f &P1, Point2f &P2 )
{
	Point2f diff = P2-P1;
	return std::sqrt( diff.x*diff.x + diff.y*diff.y );
}
inline void LoG( Mat &f )
{
	GaussianBlur(f,f, Size(3,3), 5);
	Laplacian(f,f, CV_8U, 1);
}
void seehistogram( Mat &hist )
{
	Mat canvas = Mat::zeros(400, 500, CV_8UC1);
	for(int i=0; i<hist.rows; i++)
	{
		int x = (int)hist.at<ushort>(i,0);
		int y = 300-1*x;
		if( y<0 ) y=300;
		Point ul = Point(50+20*(i+1), y);
		Point lr = Point(50+20*(i+2), 300);
		rectangle(canvas, ul, lr, Scalar(255,255,255), -1);
	}
	imshow("histogram of flows", canvas);
	waitKey(5);
}
void displayFeatureVector( Mat &f )
{
	//cout<<f.rows<<endl;
	flip( f, f, 1 );
	Mat canvas = Mat::zeros(500,900, CV_8UC1);
	Mat canvas2 = Mat::zeros(500, 300, CV_8UC1);
	for( int i=0; i< f.cols; i++ )
	{
		float x = f.at<float>(0,i);
		Point base = Point( 50+2*i, 400 );
		Point top = Point( 50+2*i, 400-2*ceil(x) );
		cv::line( canvas, base, top, Scalar(255,255,255),2 );
	}
	imshow("phoof", canvas);
	for( int i=360; i< f.cols; i++ )
	{
		float x = f.at<float>(0,i);
		Point base = Point( 30+6*(i-360), 400 );
		Point top = Point( 30+6*(i-360), 400-1*ceil(x) );
		cv::line( canvas2, base, top, Scalar(255,255,255),2 );
	}
	imshow("hoof", canvas2);
	waitKey(5);
		
}
void createHist( vector<Point2f> &ptsA, vector<Point2f> &ptsB, vector<uchar> status )
{
	Mat hist = Mat::zeros(18, 1, CV_16UC1);
	
	for( unsigned int i=0; i< ptsA.size(); i++ )
	{
		float m=0, d=0;
		if( status[i] )
		{
			m = slope( ptsA[i], ptsB[i] );
			d = euclideanDist( ptsA[i], ptsB[i] );
		}						//commented the comment: //value returned is in -180 to +180. Scaling for histogram
		//cout<<(int)(m/20)<<" ";
		if( m<0 ) m+=360;				//scale [0..-180] to [180..360]
		if( isnan(m) ) m=0;
		int k = (int)(m/20);
		hist.at<ushort>(k,0) += (int) d;
	}
	//cout<<endl;
	normalize( hist, hist, 0, 100, NORM_MINMAX );
	//cout<<hist.t()<<endl;
	seehistogram( hist );
}

Mat roi_to_hist( Mat angles_roi, Mat dist_roi )
{
	// (hopefully) generic function that calculates 18-bin histogram from "angles_roi",
	// weighted by corresponding weights in "dist_roi". Note: both are floating point.
	// pass an equal sized matrix of ones to "dist_roi" to remove the weighing effect.
	
	Mat hist = Mat::zeros(18, 1, CV_32FC1);			//row form- c++ allows row concat. 32f for actual train/test. int sufficient to see.
	//cout<<"histogramming..\n";
	for( int i=0; i< angles_roi.rows; i++ )
	{
		//float *cur_Arow = angles_roi.ptr<float>(i);
		//float *cur_Drow = dist_roi.ptr<float>(i);
		for( int j=0; j< angles_roi.cols; j++ )
		{
			//float m = cur_Arow[j];
			//float d = cur_Drow[j];
			float m = angles_roi.at<float>(i,j);
			float d = dist_roi.at<float>(i,j);
			int k = (int)(m/20);
			if( k<18 )
				hist.at<float>(k,0) += d;
		}
		//delete [] cur_Arow;
		//delete [] cur_Drow;
	}
return hist.clone();
}
Mat createHist_Pi( Mat angles, Mat dist )
{
	Mat hist = Mat::zeros(18, 1, CV_16UC1);
	Mat Pifeature;
	
	//smallest: 2x3, 16 in number.
	for( int y=0; y<11; y+=3 )
		for( int x=0; x<12; x+=3 )
		{
			Mat roi_a = angles( Rect(Point(x,y), Point(x+2,y+1)) ).clone();
			Mat roi_d = dist( Rect(Point(x,y), Point(x+2,y+1)) ).clone();
			//cout<<roi_a.size()<<" "<<roi_d.size()<<endl;
			Pifeature.push_back( roi_to_hist( roi_a.clone(), roi_d.clone() ) );
			
		}
		
	for( int y=0; y<11; y+=6 )
		for( int x=0; x<12; x+=6 )
		{
			Mat roi_a = angles( Rect(Point(x,y), Point(x+5,y+4)) );
			Mat roi_d = dist( Rect(Point(x,y), Point(x+5,y+4)) );
			//cout<<roi_a.size()<<" "<<roi_d.size()<<endl;
			Pifeature.push_back( roi_to_hist( roi_a.clone(), roi_d.clone() ) );
		}
		
	Pifeature.push_back( roi_to_hist( angles.clone(), dist.clone() ) );
	//cout<<"pi: "<<Pifeature.size()<<endl;
	Mat temp = Pifeature.t();
	return temp.clone();
}
Mat createHist2( Mat angles, Mat dist )
{
	
	Mat hist = Mat::zeros(18, 1, CV_16UC1);
	int k=0;
	Mat feature;
	//smallest blocks. 16 in number.
	for( int i=0; i<31; i+=8 )		//31 instead of 12, 8 instead of 3
	{
		for( int j=0; j<31; j+=8 )
		{
			Mat roi_a = angles( Rect(Point(j,i), Point(j+7,i+7)) ).clone();		//+7
			Mat roi_d = dist( Rect(Point(j,i), Point(j+7,i+7)) ).clone();
			
			//cout<<roi_a.size()<<" "<<roi_d.size()<<endl;
			
			feature.push_back( roi_to_hist( roi_a.clone(), roi_d.clone() ) );
		}
	}
	//cout<<"---\n";
	
	//one level up. 4 in number.
	for( int i=0; i<31; i+=16 )
		for( int j=0; j<31; j+=16 )		//31 instead of 12, 16 instead of 6.
		{
			Mat roi_a = angles( Rect(Point(j,i), Point(j+15, i+15)) ).clone();		//+15 instead of 5
			Mat roi_d = dist( Rect(Point(j,i), Point(j+15, i+15)) ).clone();
			
			//cout<<roi_a.size()<<" "<<roi_d.size()<<endl;
			
			feature.push_back( roi_to_hist( roi_a.clone(), roi_d.clone() ) );
			
		}
	//cout<<"\n-----------\n";
	
	//finally, the whole thing.
	feature.push_back( roi_to_hist( angles.clone(), dist.clone() ) );
	//cout<<"usual: "<<feature.size()<<endl;
	Mat temp = feature.t();
	return temp.clone();
}

void calcAngles_Mags( vector<Point2f> &ptsA, vector<Point2f> &ptsB, vector<uchar> &status, vector<float> &m, vector<float> &d )
{
	#pragma SIMD
	
	for( unsigned int i=0; i< ptsA.size(); i++ )
	{
		if( status[i] )
		{
			m.push_back( slope( ptsA[i], ptsB[i] ) );
			d.push_back( euclideanDist( ptsA[i], ptsB[i] ) );
		}
		else
		{
			m.push_back( 0 );
			d.push_back( 0 );
		}
	}
}
struct Belief
{
	float obs, obs_svm, svm_obs;
	vector<int> decisions;
	vector<float> posteriors;	//this is for recording to file; don't bother.
	int idx, L;
	Belief()
	{	obs = obs_svm = svm_obs = 0.86; idx = 0;
		L = 8;									// learning rate.. kind of.
		decisions.reserve(L); decisions.assign(L, 0);
	}
	~Belief()
	{
		//ofstream fout("beliefs5");
		//copy( posteriors.begin(), posteriors.end(), ostream_iterator<float>(fout, " ") );
		//fout.close();
	}
	float confidence()
	{
		//obs_svm = svm_obs*obs;
		if( false )
			posteriors.push_back( obs );
		return obs_svm;
	}
	void update( int x )
	{
		if( idx == L )
			idx = 0;
		decisions[idx++] = x+1;
		obs = mean(decisions)[0]/2;
		obs_svm = svm_obs*obs;
		if( obs < 0.1 )
			obs = 0.1;
		cout<<"current prior: "<<obs<<"\t| current posterior: "<<obs_svm<<endl;
	}
} B;

void write_out()
{
	for( int i=0; i< collection.size(); i++ )
	{
		stringstream ss;
		ss<<"PHOOF/"<<i<<".jpg";
		imwrite( ss.str(), collection[i] );
	}
}
void getFrames( Mat &f1, Mat &f2 )
{
	Mat f;
	for( int i=0; i<4; i++ )
	{
		cap>> f;
		if( i==1 )
		{	f.copyTo( f1 ); delay(15); }
		if( i==2 )
			f.copyTo( f2 );
	}
}
int main()
{
	//VideoCapture cap(0);
	Mat obs_feature, free_feature, obs_feature_pi, free_feature_pi, free_feature_pi2, obs_feature_pi2;
	Mat frameA, frameB, temp, orig, fA, frameA_pi, frameB_pi;
	vector<Point2f> ptsA, ptsA_pi, ptsA_pi2;
	for( int i=OF_WIN_PI; i< F_ROWS-10; i+=OF_WIN_PI )
	{	for( int j=OF_WIN_PI; j< F_COLS; j+=OF_WIN_PI )
			ptsA_pi.push_back( Point(j,i) );
	cout<<ptsA_pi.size()<<endl;
	}

	init();

	cap.set(CV_CAP_PROP_FRAME_WIDTH, F_COLS);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, F_ROWS);
#ifdef _USE_CVSVM
	SVM rbf_svm;
	rbf_svm.load("SVMRBF.yml");
#else
	svm_model* linmodel = NULL;
	linmodel = svm_load_model("trainedmodel_linearC");
	assert( linmodel != NULL );
	
#endif

	bool start = true, useLOG = true, drawing = false, scaledrawing = true, paused = false;
	int filenum = 0, obs = 0, free = 0, folder = 1, keyDelay = 0, fn = 0, imcount = 0;
	char ch = 'p';
	while( fn++ < 100 )
	{
		differentialDrive( 15+fn%2, 15 );
		delay(30);
//		auto tstart = chrono::high_resolution_clock::now();

		//cap >> frameA;
		getFrames( frameA, frameB );
		resize( frameA, orig, Size(F_COLS*2, F_ROWS*2) );
		if( frameA.size() != Size(F_COLS, F_ROWS) )
		{
			resize( frameA, frameA, Size(F_COLS, F_ROWS) );
			resize( frameB, frameB, Size(F_COLS, F_ROWS) );
		}
		//GaussianBlur( frameA, fA, Size(3,3), 3 );
		//addWeighted( frameA, 2.5, fA, -2, 0, frameA );
		//frameA.copyTo(temp);
		
		cvtColor(frameA, frameA, CV_BGR2GRAY);
		cvtColor(frameB, frameB, CV_BGR2GRAY);
		if( useLOG )
		{	LoG(frameA); LoG(frameB); }
		
		if( false )
		{
			frameA.copyTo(frameB);
			start = false;
		}
		
		vector<Point2f> ptsB_pi;
		vector<float> err_pi;
		vector<uchar> status_pi;
		
		if( useLOG )
			calcOpticalFlowPyrLK(frameA*2, frameB*2, ptsA_pi, ptsB_pi, status_pi, err_pi, Size(31,31));
		else
			calcOpticalFlowPyrLK( frameA, frameB, ptsA_pi, ptsB_pi, status_pi, err_pi, Size(31,31) );
				
		//frameA.copyTo(frameB);
		
		vector<float> m_pi, d_pi;
		m_pi.reserve( ptsA_pi.size() );
		d_pi.reserve( ptsA_pi.size() );
		
		//from points, calculate slope (m_pi) and distance (d_pi)
		calcAngles_Mags( ptsA_pi, ptsB_pi, status_pi, m_pi, d_pi );
		
		//re-arrange into a matrix form (with 11 rows, in this case).
		Mat angles_pi(m_pi, true), mags_pi(d_pi, true);
		angles_pi = angles_pi.reshape(1, 11); mags_pi = mags_pi.reshape(1,11);
		
		//create the histogram.
		Mat feature_pi = createHist_Pi( angles_pi, mags_pi ).clone();
		
//		auto tend = chrono::high_resolution_clock::now();
//		float timetaken = chrono::duration_cast<chrono::milliseconds>( tend-tstart ).count();
		
		for( unsigned int i=0; i< ptsA_pi.size(); i++ )
			if( status_pi[i] )
				cv::line( orig, ptsA_pi[i]*2, ptsB_pi[i]*2, Scalar(0,0,0) );
		//imshow("orig", orig);
		//waitKey(5);
		//collection.push_back( orig.clone() );
		
	#ifdef _USE_CVSVM	
		float conf = rbf_svm.predict( feature_pi, true );
	#else
		svm_node* testdata = new svm_node[ feature_pi.cols +1 ];
		float *feat_ptr = feature_pi.ptr<float>(0);
		for( int i = 0; i<feature_pi.cols; i++ )
		{
			testdata[i].index = i;
			testdata[i].value = *feat_ptr++;
//			cout<<testdata[i].value<<" ";
		}
		cout<<endl;
		testdata[ feature_pi.cols ].index = -1;

		float conf = svm_predict( linmodel, testdata );
		delete [] testdata, feat_ptr;
	#endif
//		cout<<"prediction confidence: "<<conf<<"\ttime taken: "<<timetaken<<endl;
//		stringstream ss;
		if( conf < 0 )
		{
			int Lflow = (int) cv::sum( mags_pi( Rect(Point(0,0), Point(5,10)) ) )[0];
			int Rflow = (int) cv::sum( mags_pi( Rect(Point(6,0), Point(11,10)) ) )[0];

			if( Lflow > Rflow )
				differentialDrive( 25, 0 );
			else
				differentialDrive( 0, 19 );
			delay(80);
//			cout<<"obs";
//			ss<<Lflow<<"|"<<Rflow<<",";
		}
		else
			differentialDrive( 15+fn%2, 15 );

//		std::this_thread::sleep_for( chrono::milliseconds( 50 );
		//stringstream ss;
//		ss<<conf;
//		putText( orig, ss.str(), Point(20,250), 1, 1, Scalar(0,0,0) );
		collection.push_back( orig.clone() );

	}
	#ifndef _USE_CVSVM
		delete linmodel;
	#endif
	cap.release();
	un_init();
	write_out();
	system("sudo shutdown -h now");
return 0;
}
