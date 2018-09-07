#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main ( int argc, char** argv )
{
    double a=1.0, b=2.0, c=1.0;         // 真实参数值
    int N=100;                          // 数据点
    double w_sigma=1.0;                 // 噪声Sigma值
    cv::RNG rng;                        // OpenCV随机数产生器
    double ae=2.0, be=-1.0, ce=5.0;     // abc参数的估计值

    vector<double> x_data, y_data;      // 数据

    cout<<"generating data…… "<<endl;
    for ( int i=0; i<N; i++ )
    {
        double x = i/100.0;
        x_data.push_back ( x );
        y_data.push_back (
                exp ( a*x*x + b*x + c ) + rng.gaussian ( w_sigma )
        );
        //cout<<x_data[i]<<" "<<y_data[i]<<endl;
    }

    // 开始Gauss-Newton迭代
    int iterations = 100;    // 迭代次数
    double cost = 0, lastCost = 0;  // 本次迭代的cost和上一次迭代的cost

    for (int iter = 0; iter < iterations; iter++) {

        Matrix3d H = Matrix3d::Zero();             // Hessian = J^T J
        Vector3d b = Vector3d::Zero();             // bias
        cost = 0;

        for (int i = 0; i < N; i++) {
            double xi = x_data[i], yi = y_data[i];  // 第i个数据点
            double error = 0;   // 第i个数据点的计算误差
            error = yi-exp(ae * xi * xi + be * xi + ce);
            Vector3d J; // 雅可比矩阵
            J[0] = -xi*xi*exp(ae*xi*xi+be*xi+ce);
            J[1] = -xi*exp(ae*xi*xi+be*xi+ce);
            J[2] = -exp(ae*xi*xi+be*xi+ce);

            H += J * J.transpose();
            b += -error * J;

            cost += error * error;

        }

        Vector3d dx;
        dx = H.inverse()*b;
        //dx = H.colPivHouseholderQr().solve(b);    //QR分解，可加快求解速度
        //dx = H.ldlt().solve(b);    //ldlt分解，可加快求解速度

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost > lastCost) {
            // 误差增长了，说明近似的不够好
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }
        // 更新abc估计值
        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;
        cout << "total cost: " << cost << endl;
    }
    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    return 0;
}