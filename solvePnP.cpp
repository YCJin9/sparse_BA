#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "sophus/se3.h"

using namespace std;
using namespace Eigen;
using namespace cv;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;

void getdata (
        VecVector2d& p2d,
        VecVector3d& p3d
        );

int main ( int argc, char** argv )
{
    //生成3D点和像素坐标点
    VecVector2d p2d;
    VecVector3d p3d;
    getdata(p2d, p3d);

    //高斯牛顿迭代
    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();
    cout << "points: " << nPoints << endl;

    Sophus::SE3 T_esti; // estimated pose   相机位姿R,t李代数形式

    for (int iter = 0; iter < iterations; iter++) {
        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < nPoints; i++) {
            Vector2d p2 = p2d[i];
            Vector3d p3 = p3d[i];  //第i个数据

            Vector3d P = T_esti * p3;
            double x = P[0];
            double y = P[1];
            double z = P[2];

            Vector2d p2_ = { fx * ( x/z ) + cx, fy * ( y/z ) + cy };
            Vector2d e = p2 - p2_;    //误差error = 观测值 - 估计值
            cost += (e[0]*e[0]+e[1]*e[1]);

            // compute jacobian
            Matrix<double, 2, 6> J;
            J(0,0) = -(fx/z);
            J(0,1) = 0;
            J(0,2) = (fx*x/(z*z));
            J(0,3) = (fx*x*y/(z*z));
            J(0,4) = -(fx*x*x/(z*z)+fx);
            J(0,5) = (fx*y/z);
            J(1,0) = 0;
            J(1,1) = -(fy/z);
            J(1,2) = (fy*y/(z*z));
            J(1,3) = (fy*y*y/(z*z)+fy);
            J(1,4) = -(fy*x*y/(z*z));
            J(1,5) = -(fy*x/z);

            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

        // solve dx
        Vector6d dx;
        dx = H.ldlt().solve(b);

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // 误差增长了，说明近似的不够好
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // 更新估计位姿
        T_esti = Sophus::SE3::exp(dx)*T_esti;
        lastCost = cost;
        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    return 0;

}

void getdata (
        VecVector2d& p2d,
        VecVector3d& p3d
){
    //-- 读取图像
    Mat img_1 = imread ( "./data/1.png", CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( "./data/2.png", CV_LOAD_IMAGE_COLOR );

    //初始化
    vector<KeyPoint> keypoints_1;
    vector<KeyPoint> keypoints_2;
    vector< DMatch > matches;
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

    //检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;
    Mat d1 = imread ( "./data/1_depth.png", CV_LOAD_IMAGE_UNCHANGED );
    for ( DMatch m:matches )
    {
        ushort d = d1.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float z = d/5000.0;
        float x = ( keypoints_1[m.queryIdx].pt.x - cx )* z / fx;
        float y = ( keypoints_1[m.queryIdx].pt.y - cx ) * z / fy;
        p3d.push_back ( Vector3d( x, y, z) );
        p2d.push_back ( Vector2d( keypoints_2[m.trainIdx].pt.x, keypoints_2[m.trainIdx].pt.y) );
    }

}

