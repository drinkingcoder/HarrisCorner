//
//  main.cpp
//  Lab2
//
//  Created by drinking on 12/13/15.
//  Copyright © 2015 drinking. All rights reserved.
//

#include <iostream>

//
//  main.cpp
//  Lab1
//
//  Created by drinking on 11/29/15.
//  Copyright © 2015 drinking. All rights reserved.
//

#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <string>
#include <math.h>

#define DEBUG 1

using namespace std;
using namespace::cv;
string inputFileName = "/Users/drinkingcoder/Documents/university/Vision/Labs/Lab2/Lab2/input.jpg";
string exportFileName = "/Users/drinkingcoder/Documents/university/Vision/Labs/Lab2/Lab2/output.png";
string grayScaleFileName = "/Users/drinkingcoder/Documents/university/Vision/Labs/Lab2/Lab2/gray.png";
string PartialDeriv = "/Users/drinkingcoder/Documents/university/Vision/Labs/Lab2/Lab2/PartialDeriv.png";
string rName = "/Users/drinkingcoder/Documents/university/Vision/Labs/Lab2/Lab2/r.png";
string featureName = "/Users/drinkingcoder/Documents/university/Vision/Labs/Lab2/Lab2/feature.png";
string temperatureName = "/Users/drinkingcoder/Documents/university/Vision/Labs/Lab2/Lab2/temperature.png";
string result = "/Users/drinkingcoder/Documents/university/Vision/Labs/Lab2/Lab2/result.png";

string fontContext = "3130000696 Copyright@2015,DrinkingCoder All Rights Reserved.";
CvSize size = cvSize(800,600);
double Threshold = 80894360;
int length = 11;

Mat inputImg,workImg,grayImg,showImg,xPDImg,yPDImg,rImg;
double** xpd,**ypd;
double k = 0.04;
int Aperture = 3;
int sobelx[10][10],sobely[10][10],sumSobel;

void PascalTriangle()
{
    int pascal1[21][21];
    int pascal2[21][21];
    memset(pascal1,0,21*21*sizeof(int));
    memset(pascal2,0,21*21*sizeof(int));
    memset(sobelx, 0, 10*10*sizeof(int));
    memset(sobely,0,10*10*sizeof(int));
    sumSobel = 0;
    pascal1[1][1] = 1;
    pascal1[2][1] = 1;
    pascal1[2][2] = 1;
    pascal2[1][1] = 1;
    pascal2[2][1] = 1;
    pascal2[2][2] = -1;
    for(int i=3;i<21;i++)
    {
        pascal1[i][1] = 1;
        pascal2[i][1] = 1;
        for(int j=2;j<=i;j++)
        {
            pascal1[i][j] = pascal1[i-1][j-1]+pascal1[i-1][j];
            pascal2[i][j] = pascal2[i-1][j-1]+pascal2[i-1][j];
        }
    }
    for(int i=1;i<=Aperture;i++)
    {
        for(int j=1;j<=Aperture;j++)
        {
            sobelx[i-1][j-1]+= pascal1[Aperture][i]*pascal2[Aperture][Aperture-j+1];
            sobely[i-1][j-1]+= pascal1[Aperture][j]*pascal2[Aperture][i];
            if(sobelx[i-1][j-1]>0) sumSobel+=sobelx[i-1][j-1];
        }
    }
}

void SobelOperator(Mat& grayImg)
{
    double max = -10000000,min = 10000000;
    xPDImg = Mat::zeros(grayImg.rows,grayImg.cols,CV_64F);
    yPDImg = Mat::zeros(grayImg.rows,grayImg.cols,CV_64F);
    Mat img = Mat::zeros(grayImg.rows, grayImg.cols, CV_8U);
    double d;
    xpd = (double**)malloc(grayImg.rows*sizeof(double*));
    ypd = (double**)malloc(grayImg.rows*sizeof(double*));
    uchar** gray = (uchar**)malloc(grayImg.rows*sizeof(uchar*));
    uchar** image = (uchar**)malloc(grayImg.rows*sizeof(uchar*));
    for(int i=0;i<grayImg.rows;i++)
    {
        xpd[i] = xPDImg.row(i).ptr<double>();
        ypd[i] = yPDImg.row(i).ptr<double>();
        gray[i] = grayImg.row(i).ptr<uchar>();
        image[i] = img.row(i).ptr<uchar>();
    }
    for(int i=0;i<grayImg.rows-Aperture+1;i++)
        for(int j=0;j<grayImg.cols-Aperture+1;j++)
        {
            d = 0;
            for(int r = 0; r<Aperture ; r++)
                for(int c = 0; c<Aperture ; c++)
                    d+=sobelx[r][c]*gray[i+r][j+c];
            d = fabs(d/sumSobel);
            xpd[i+1][j+1] = d;
            d = 0;
            for(int r = 0; r<Aperture ; r++)
                for(int c = 0; c<Aperture ; c++)
                    d+=sobely[r][c]*gray[i+r][j+c];
            d = fabs(d/sumSobel);
            ypd[i+1][j+1] = d;
            if(max < xpd[i+1][j+1]+ypd[i+1][j+1]) max = xpd[i+1][j+1]+ypd[i+1][j+1];
            if(min > xpd[i+1][j+1]+ypd[i+1][j+1]) min = xpd[i+1][j+1]+ypd[i+1][j+1];
        }
    double gap = max-min;
    for(int i=0;i<grayImg.rows-2;i++)
        for(int j=0;j<grayImg.cols-2;j++)
            image[i][j] = uchar((xpd[i][j]+ypd[i][j]-min)/gap*255);
    imshow("img", img);
    imwrite(PartialDeriv, img);
    waitKey();
}

void GetRValue()
{
    cout << "R start.."<<endl;
    int locatr,locatc;
    Mat l1 = Mat(grayImg.rows,grayImg.cols,grayImg.type()), l2 = Mat(grayImg.rows,grayImg.cols,grayImg.type());
    double lambda1,lambda2,d,a,b,c,min = 10000000000,max = -10000000000;
    rImg = Mat(grayImg.rows,grayImg.cols,CV_64F);
    Mat Img = Mat(grayImg.rows,grayImg.cols,CV_8U);
    Mat featurePointImg = Mat(grayImg.rows,grayImg.cols,CV_8U);
    Mat temperatureImg = Mat(grayImg.rows,grayImg.cols,CV_8UC3);
    double** r = (double**)malloc(grayImg.rows*sizeof(double*));
    uchar** image = (uchar**)malloc(grayImg.rows*sizeof(uchar*));
    uchar** feature = (uchar**)malloc(grayImg.rows*sizeof(uchar*));
    uchar** temperature = (uchar**)malloc(grayImg.rows*sizeof(uchar*));
    for(int i=0;i<grayImg.rows;i++)
    {
        r[i] = rImg.row(i).ptr<double>();
        image[i] = Img.row(i).ptr<uchar>();
        feature[i] = featurePointImg.row(i).ptr<uchar>();
        temperature[i] = temperatureImg.row(i).ptr<uchar>();
    }
    for(int i=0;i<grayImg.rows-length;i++)
        for (int j=0;j<grayImg.cols-length; j++) {
            locatr = i+length/2;
            locatc = j+length/2;
            a = 0;
            b = 0;
            c = 0;
            for(int rr=0; rr<length;rr++)
                for(int cc=0; cc<length ;cc++)
                {
                    a += xpd[i+rr][j+cc]*xpd[i+rr][j+cc];
                    b += xpd[i+rr][j+cc]*ypd[i+rr][j+cc];
                    c += ypd[i+rr][j+cc]*ypd[i+rr][j+cc];
                }
            lambda1 = ((a+c)+sqrt(4*b*b+(a-c)*(a-c)))/2;
            lambda2 = ((a+c)-sqrt(4*b*b+(a-c)*(a-c)))/2;
            d = lambda1*lambda2-k*(lambda1+lambda2)*(lambda1+lambda2);
            r[locatr][locatc] = d;
            if(min > d) min = d;
            if(max < d) max = d;
            if(d>Threshold)
                image[locatr][locatc] = 255;
            else
               image[locatr][locatc] = 0;
        }
    int rr,cc,x,y;
    for(int i=0;i<featurePointImg.rows;i++)
        for(int j=0;j<featurePointImg.cols;j++)
        {
            if( r[i][j] < Threshold )
            {
                feature[i][j] = 0;
                continue;
            }
            feature[i][j] = 255;
            for(x = -5;x<5;x++)
            {
                rr = i+x;
                if(rr<0||rr>=rImg.rows) continue;
                for (y = -5; y<5; y++) {
                    cc = j+y;
                    if(cc<0||cc>=rImg.cols) continue;
                    if(x == 0 && y == 0) continue;
                    if (r[rr][cc] >= r[i][j]) {
                        feature[i][j] = 0;
                        break;
                    }
                }
                if ( feature[i][j] == 0) break;
            }
            if(feature[i][j] == 255)
                circle(inputImg, Point(j,i), 1, CV_RGB(255, 0, 0),2);
        }
    
    double gap = max - min;
    int base = (1<<24)-1;
    for(int i=0;i<temperatureImg.rows;i++)
        for(int j=0;j<temperatureImg.cols;j++)
        {
            d = (r[i][j]-min)/gap*base;
            temperature[i][j*3] = uchar(int(d)%(256));
            temperature[i][j*3+1] = uchar(int(d)%(1<<16)/256);
            temperature[i][j*3+2] = uchar(int(d)/(1<<16));
        }
    
    imshow("img", temperatureImg);
    imwrite(temperatureName, temperatureImg);
    waitKey();
    
    imshow("img", Img);
    imwrite(rName, Img);
    waitKey();
    
    imshow("img", featurePointImg);
    imwrite(featureName, featurePointImg);
    cout << "R finished.." << endl;
    waitKey();
}

double transforToFloat(string s)
{
    double res = 0;
	double real = 1;
	int i;
    for(i=0;i<s.length();i++)
    {
        res*=10;
        if(s[i] == '.') break;
        else if(s[i]<='9' && s[i]>='0')
            res+=s[i]-'0';
    }
	if(i == s.length()) return res;
	i++;
	for(;i<s.length();i++)
	{
		real*=0.1;
        if(s[i]<='9' && s[i]>='0')
            res+=(s[i]-'0')*real;
	}
	return res;
}

int main(int argc, const char * argv[]) {
    if(argc == 1) return -1;
    inputFileName = string(argv[1]);
    if(argc >= 3) k = transforToFloat(argv[2]);
    if(argc >= 4) Aperture = (int)transforToFloat(argv[3]);
	cout << k << endl;
	cout << Aperture << endl;
    namedWindow("Feature Detection");
    
    inputImg = imread(inputFileName,IMREAD_COLOR);
    
    cvtColor(inputImg, grayImg,CV_RGB2GRAY);
    int r = inputImg.rows/10;
    int c = inputImg.cols/10;
    int flag = 0;

    imshow("img", grayImg);
    imwrite(grayScaleFileName, grayImg);
    waitKey();
    
    PascalTriangle();
    
    SobelOperator(grayImg);
    
    GetRValue();

    imshow("img", inputImg);
    imwrite(result, inputImg);
    waitKey();
    
    destroyAllWindows();
    return 0;
}
