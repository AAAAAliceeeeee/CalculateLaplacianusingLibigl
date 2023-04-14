#include <igl/boundary_facets.h>
#include <igl/colon.h>
#include <igl/cotmatrix.h>
#include <igl/jet.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/readOFF.h>
#include <igl/setdiff.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/unique.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Sparse>
#include <iostream>
int main(int argc, char *argv[])
{
    using namespace Eigen;
    using namespace std;
    MatrixXd V;
    MatrixXi F;
    
    igl::readOFF("models/camelhead.off",V,F);
    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    // Calculate the Laplacian 
    // 1. Set up matrix
    SparseMatrix<double> L (V.rows(), V.rows());
    // 2. Loop over the triangles and calculate them 
    for (int triangle = 0; triangle < F.rows(); triangle ++)
    {
        // 0,1; 0,2; 1,2
        for (int v1 = 0; v1 <= 1; v1++)
        {
            for (int v2 = 2; v2 > v1; v2 --)
            {
                int v3 = 3 - v2 - v1;
                int i = F.coeff(triangle, v1);
                int j = F.coeff(triangle, v2);
                int k = F.coeff(triangle, v3);

                Vector3d ki = V.row(i)-V.row(k);
                Vector3d kj = V.row(j) - V.row(k);
                double cosikj = (ki.dot(kj) )/ (ki.norm() + kj.norm());
                double sinikj = (ki.cross(kj)).norm() / (ki.norm() + kj.norm());
                double cot = cosikj / sinikj;

                L.coeffRef(i,j) += (cot/2); // if before = 0: need reallocation 
                L.coeffRef(j,i) += (cot/2); 
                L.coeffRef(i,i) -= (cot/2); 
                L.coeffRef(j,j) -= (cot/2); 
            }
        }
    }

    //cout << L << endl;
    Eigen::SparseMatrix<double> L2;
    igl::cotmatrix(V,F,L2);
    cout << (L.isApprox(L2)) << endl;
    //cout << "////////////////////////////////////////////////" << endl;
    //cout << L.col(0) << endl; 
    //cout << "////////////////////////////////////////////////" << endl;
    //cout << L2.col(0) << endl;
 ///////////////////////////////////////////////////////////////////////////////////
 // Solve a Laplacian Function with the given dirichlet boundary conditions
  // 1. break the L into block matrices:
  // Find boundary edges
  MatrixXi E;
  igl::boundary_facets(F,E); 
   // Find boundary vertices
  VectorXi b,IA,IC;
  igl::unique(E,b,IA,IC);
  // List of all vertex indices
  VectorXi all,in;
  igl::colon<int>(0,V.rows()-1,all);
  // List of interior indices
  igl::setdiff(all,b,in,IA);

  // Construct and slice up Laplacian
  SparseMatrix<double> L_in_in,L_in_b;

  igl::slice(L,in,in,L_in_in);
  igl::slice(L,in,b,L_in_b);
  
  // boundary condition ? how to input a custom boudary condition? 
  //L_in_in * uI == (bI) -L_in_b * g
  VectorXd uI = VectorXd::Zero(L_in_in.rows());
  VectorXd g = VectorXd::Zero(L_in_b.cols());
  // Input g here:
  // Default: set g = tutorial(zvalue)
  VectorXd Z = V.col(2);
  igl::slice(Z,b,g);
  // Solve PDE
  SimplicialLLT<SparseMatrix<double > > solver(-L_in_in);
  VectorXd Z_in = solver.solve(L_in_b*g);
  // slice into solution
  igl::slice_into(Z_in,in,Z);



  viewer.data().set_mesh(V, F);
  viewer.data().show_lines = false;
  viewer.data().set_data(Z);
  viewer.launch();
}