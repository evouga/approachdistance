#include "ApproachDistance.h"
#include "CTCD.h"
#include "Distance.h"
#include <iostream>
#include <algorithm>

Eigen::Matrix<double, 13, 3> DOPs;

struct DOPNode
{
    DOPNode* left, * right;
    int frame;
    int face;
    double lo[13];
    double hi[13];
    double bestd;

    DOPNode() : left(NULL), right(NULL) {}
    ~DOPNode() { delete left; delete right; }
};

DOPNode* fitFace(const std::vector<Eigen::MatrixXd*>& Vs, const Eigen::MatrixXi& F, int frame, int face)
{
    DOPNode* result = new DOPNode;
    result->frame = frame;
    result->face = face;
    result->bestd = 0;
    for (int i = 0; i < 13; i++)
    {
        result->lo[i] = std::numeric_limits<double>::infinity();
        result->hi[i] = -std::numeric_limits<double>::infinity();
    }

    for (int i = 0; i < 3; i++)
    {
        Eigen::Vector3d pt1 = (*Vs[frame]).row(F(face, i));
        Eigen::Vector3d pt2 = (*Vs[frame+1]).row(F(face, i));
        for (int j = 0; j < 13; j++)
        {
            double coord = pt1.dot(DOPs.row(j));
            result->lo[j] = std::min(result->lo[j], coord);
            result->hi[j] = std::max(result->hi[j], coord);
            coord = pt2.dot(DOPs.row(j));
            result->lo[j] = std::min(result->lo[j], coord);
            result->hi[j] = std::max(result->hi[j], coord);
        }
    }
    return result;
}

DOPNode* buildDOP(std::vector<DOPNode*> &leaves, int first, int last)
{
    if (last - first == 1)
        return leaves[first];
    DOPNode* result = new DOPNode;
    result->frame = -1;
    result->face = -1;
    result->bestd = 0;
    for (int i = 0; i < 13; i++)
    {
        result->lo[i] = std::numeric_limits<double>::infinity();
        result->hi[i] = -std::numeric_limits<double>::infinity();
    }
    for (int i = first; i < last; i++)
    {
        for (int j = 0; j < 13; j++)
        {
            result->lo[j] = std::min(result->lo[j], leaves[i]->lo[j]);
            result->hi[j] = std::max(result->hi[j], leaves[i]->hi[j]);
        }
    }
    double bestlen = 0;
    int bestaxis = 0;
    for (int j = 0; j < 13; j++)
    {
        double len = result->hi[j] - result->lo[j];
        if (len > bestlen)
        {
            bestlen = len;
            bestaxis = j;
        }
    }
    std::sort(leaves.begin() + first, leaves.begin() + last, 
        [bestaxis](DOPNode* a, DOPNode* b) -> bool
    {
        return a->lo[bestaxis] + a->hi[bestaxis] < b->lo[bestaxis] + b->hi[bestaxis];
    }
    );
    int mid = first + (last - first) / 2;
    result->left = buildDOP(leaves, first, mid);
    result->right = buildDOP(leaves, mid, last);
    return result;
}

static double ptMeshDist(Eigen::Vector3d pt, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    double bestdist = std::numeric_limits<double>::infinity();
    int nfaces = F.rows();
    for (int i = 0; i < nfaces; i++)
    {
        double dummy;
        Eigen::Vector3d dvec = Distance::vertexFaceDistance(pt, V.row(F(i, 0)).transpose(), V.row(F(i, 1)).transpose(), V.row(F(i, 2)).transpose(), dummy, dummy, dummy);
        bestdist = std::min(bestdist, dvec.norm());
    }
    return bestdist;
}

static bool inProximity(
    Eigen::Vector3d pt,
    const std::vector<Eigen::MatrixXd*> &Vs,
    const Eigen::MatrixXi& F,    
    DOPNode *tree,
    double d)
{
    for (int i = 0; i < 13; i++)
    {
        double coord = pt.dot(DOPs.row(i));
        if (coord < tree->lo[i] - d)
            return false;
        if (coord > tree->hi[i] + d)
            return false;
    }

    if (tree->bestd >= d)
        return false;

    if (tree->left && inProximity(pt, Vs, F, tree->left, d))
        return true;
    if (tree->right && inProximity(pt, Vs, F, tree->right, d))
        return true;

    if (tree->frame != -1 && tree->face != -1)
    {
        double t;
        Eigen::Vector3d q1start = Vs[tree->frame]->row(F(tree->face, 0)).transpose();
        Eigen::Vector3d q2start = Vs[tree->frame]->row(F(tree->face, 1)).transpose();
        Eigen::Vector3d q3start = Vs[tree->frame]->row(F(tree->face, 2)).transpose();
        Eigen::Vector3d q1end = Vs[tree->frame + 1]->row(F(tree->face, 0)).transpose();
        Eigen::Vector3d q2end = Vs[tree->frame + 1]->row(F(tree->face, 1)).transpose();
        Eigen::Vector3d q3end = Vs[tree->frame + 1]->row(F(tree->face, 2)).transpose();
        if (CTCD::vertexFaceCTCD(pt, q1start, q2start, q3start, pt, q1end, q2end, q3end, d, t))
        {
            return true;
        }
        if (CTCD::vertexVertexCTCD(pt, q1start, pt, q1end, d, t))
        {
            return true;
        }
        if (CTCD::vertexVertexCTCD(pt, q2start, pt, q2end, d, t))
        {
            return true;
        }
        if (CTCD::vertexVertexCTCD(pt, q3start, pt, q3end, d, t))
        {
            return true;
        }
        if (CTCD::vertexEdgeCTCD(pt, q1start, q2start, pt, q1end, q2end, d, t))
        {
            return true;
        }
        if (CTCD::vertexEdgeCTCD(pt, q1start, q3start, pt, q1end, q3end, d, t))
        {
            return true;
        }
        if (CTCD::vertexEdgeCTCD(pt, q2start, q3start, pt, q2end, q3end, d, t))
        {
            return true;
        }        
    }
    tree->bestd = d;
    return false;
}

double approachDistance(
    Eigen::Vector3d pt,
    std::vector<Eigen::MatrixXd*> Vs,
    const Eigen::MatrixXi& F,
    double tol)
{
    int nfaces = F.rows();
    int nframes = Vs.size();
    if (nfaces == 0 || nframes == 0)
    {
        return std::numeric_limits<double>::infinity();
    }

    double low = 0.0;

    double high = ptMeshDist(pt, *Vs[0], F);
    if (nframes == 1)
        return high;

    DOPs << 1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 1, 0,
        1, -1, 0,
        1, 0, 1,
        1, 0, -1,
        0, 1, 1,
        0, 1, -1,
        1, 1, 1,
        1, 1, -1,
        1, -1, 1,
        1, -1, -1;

    for (int i = 0; i < 13; i++)
        DOPs.row(i) /= DOPs.row(i).norm();

    std::vector<DOPNode*> leaves((nframes - 1) * nfaces);
    for (int i = 0; i < nframes - 1; i++)
    {
        for (int j = 0; j < nfaces; j++)
        {
            leaves[i * nfaces + j] = fitFace(Vs, F, i, j);
        }
    }
    DOPNode* tree = buildDOP(leaves, 0, leaves.size());
    inProximity(pt, Vs, F, tree, high);
    
    while (high - low > tol)
    {
        double mid = (high + low) / 2.0;
        //std::cout << "Testing " << mid << std::endl;
        bool notok = inProximity(pt, Vs, F, tree, mid);
        if (notok)
            high = mid;
        else
            low = mid;
    }

    delete tree;
    return low;
}