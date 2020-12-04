#ifndef APPROACHDISTANCE_H
#define APPROACHDISTANCE_H

#include <Eigen/Core>
#include <vector>

/*
 * Computes the closest distance that a mesh approaches the point pt, where the mesh moves in a piecewise-linear trajectory
 * with keyframes specified in Vs.
 * The answer will be correct up to absolute error tol.
 *
 * Naive and slow; takes O([# faces]*[# frames]) space and time (with a large constant factor in the worst-case).
 */

double approachDistance(
    Eigen::Vector3d pt,
    std::vector<Eigen::MatrixXd*> Vs,
    const Eigen::MatrixXi& F,
    double tol);

#endif