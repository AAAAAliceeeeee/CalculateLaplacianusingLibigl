#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <iostream>

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;
  MatrixXd V;
  MatrixXi F;
  // Input OFF file 
  string meshfilepath;
  cout << "Please enter a mesh file path (OFF): " << endl;
  cin >> meshfilepath;
  igl::readOFF(meshfilepath,V,F);
  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  
  // Calculate face normal 
  Eigen::MatrixXd FaceNormals = Eigen::MatrixXd::Zero(F.rows(), 3);
  Eigen::MatrixXd P1 = Eigen::MatrixXd::Zero(F.rows(), 3);
  Eigen::MatrixXd P2 = Eigen::MatrixXd::Zero(F.rows(), 3);

  for (size_t i = 0; i < F.rows(); i++)
  {
    Vector3d A = V.row(F.coeff(i,0));
    Vector3d B = V.row(F.coeff(i,1));
    Vector3d C = V.row(F.coeff(i,2));
    Vector3d AB = B - A; 
    Vector3d AC = C - A;
    Vector3d Normal = AB.cross(AC);  
    FaceNormals.row(i) = Normal; // it keeps the value of the area (not normalized)
    P1.row(i) = (A+B+C)/3; 
    Vector3d unitNormal = Normal.normalized(); // what happen for 0? they do the check for you
    P2.row(i) = (A + B + C) / 3 + 0.005 * unitNormal;
  }

  // Calculate vertex normal
  MatrixXd VertexNormal = MatrixXd::Zero(V.rows(), 3);

  for (size_t i = 0; i < F.rows(); i++)
  {
      for (size_t j = 0; j < 3; j++)
      {
          VertexNormal.row(F.coeff(i,j)) += FaceNormals.row(i);
      }
  }
  VertexNormal.rowwise().normalize();
  MatrixXd P3 = V + 0.005 * VertexNormal;

  viewer.data().add_edges(P1,P2,Eigen::RowVector3d(255,0,0));
  viewer.data().add_edges(V, P3, Eigen::RowVector3d(0, 0, 255));
  viewer.data().set_mesh(V, F);
  viewer.launch();

}