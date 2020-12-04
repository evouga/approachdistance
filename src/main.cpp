#include "ApproachDistance.h"
#include "igl/readOBJ.h"
#include <Eigen/Core>
#include <iostream>

int main()
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    if (!igl::readOBJ("bunny.obj", V, F))
    {
        if (!igl::readOBJ("../bunny.obj", V, F))
        {
            std::cerr << "Can't load bunny" << std::endl;
            return - 1;
        }
    }

    
    int nframes = 100;
    std::vector<Eigen::MatrixXd*> frames(nframes);
    for (int i = 0; i < nframes; i++)
    {
        frames[i] = new Eigen::MatrixXd;
        *frames[i] = V;
        Eigen::Vector3d xlate;
        xlate.setRandom();
        for (int j = 0; j < V.rows(); j++)
        {
            (*frames[i]).row(j) += xlate.transpose();
        }
        
    }

    for (int i = 0; i < 20; i++)
    {
        Eigen::Vector3d querypt;
        querypt.setRandom();
        querypt *= 10.0;
        std::cout << approachDistance(querypt, frames, F, 1e-6) << std::endl;
    }
    
}