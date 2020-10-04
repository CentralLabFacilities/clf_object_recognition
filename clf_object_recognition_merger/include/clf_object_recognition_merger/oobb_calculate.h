#pragma once

#include <Eigen/Eigenvalues> // for SelfAdjointEigenSolver
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h> // for transformPointCloud

#include <vision_msgs/BoundingBox3D.h>
#include <sensor_msgs/PointCloud2.h>

namespace OOBB_Calculate
{
vision_msgs::BoundingBox3D calculate(sensor_msgs::PointCloud2 cloud)
{
  vision_msgs::BoundingBox3D ret;

  // http://codextechnicanum.blogspot.com/2015/04/find-minimum-oriented-bounding-box-of.html
  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(cloud, pcl_pc2);

  pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);

  if (temp_cloud->size() < 3)
  {
    ROS_ERROR_STREAM("cloud to small, could not create BB");
    return ret;
  }

  Eigen::Vector4f pcaCentroid;
  pcl::compute3DCentroid(*temp_cloud, pcaCentroid);

  Eigen::Matrix3f covariance;
  computeCovarianceMatrixNormalized(*temp_cloud, pcaCentroid, covariance);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
  Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
  eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

  // Note that getting the eigenvectors can also be obtained via the PCL PCA interface with something like:
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPCAprojection (new pcl::PointCloud<pcl::PointXYZ>);
  // pcl::PCA<pcl::PointXYZ> pca;
  // pca.setInputCloud(temp_cloud);
  // pca.project(*temp_cloud, *cloudPCAprojection);
  // std::cerr << std::endl << "PCLEigenVectors: " << pca.getEigenVectors() << std::endl;
  // std::cerr << std::endl << "EigenVectors: " << eigenVectorsPCA << std::endl;

  // Transform the original cloud to the origin where the principal components correspond to the axes.
  Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
  projectionTransform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
  projectionTransform.block<3, 1>(0, 3) = -1.f * (projectionTransform.block<3, 3>(0, 0) * pcaCentroid.head<3>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointsProjected(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*temp_cloud, *cloudPointsProjected, projectionTransform);

  // Get the minimum and maximum points of the transformed cloud.
  pcl::PointXYZ minPoint, maxPoint;
  pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
  const Eigen::Vector3f meanDiagonal = 0.5f * (maxPoint.getVector3fMap() + minPoint.getVector3fMap());

  // Final transform
  const Eigen::Quaternionf bboxQuaternion(eigenVectorsPCA);
  const Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();

  ret.size.x = maxPoint.x - minPoint.x;
  ret.size.y = maxPoint.y - minPoint.y;
  ret.size.z = maxPoint.z - minPoint.z;
  ret.center.position.x = bboxTransform[0];
  ret.center.position.y = bboxTransform[1];
  ret.center.position.z = bboxTransform[2];
  ret.center.orientation.w = (double)bboxQuaternion.w();
  ret.center.orientation.x = bboxQuaternion.x();
  ret.center.orientation.y = bboxQuaternion.y();
  ret.center.orientation.z = bboxQuaternion.z();

  return ret;
}
}
