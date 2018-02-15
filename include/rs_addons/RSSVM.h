//developed by: Rakib

#ifndef RSSVM_HEADER
#define RSSVM_HEADER

#include <iostream>
#include <string>
#include <ros/package.h>
#include <uima/api.hpp>
#include <boost/filesystem.hpp>

#if CV_MAJOR_VERSION == 2
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#elif CV_MAJOR_VERSION == 3
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#endif
#include <rs_addons/RSClassifier.h>
#include <rs/types/all_types.h>
#include <rs/scene_cas.h>
#include <rs/utils/time.h>
#include <rs/DrawingAnnotator.h>

#if CV_MAJOR_VERSION == 2
class RSSVM : public RSClassifier, public CvSVM
#elif CV_MAJOR_VERSION == 3
class RSSVM : public RSClassifier, public cv::ml::SVM
#endif
{

public:

  RSSVM();

  void trainModel(std::string train_matrix_name, std::string train_label_name, std::string trained_file_name);

  void classify(std::string trained_file_name,std::string test_matrix_name, std::string test_label_name, std::string obj_classInDouble);

  void classifyOnLiveData(std::string trained_file_name_saved, cv::Mat test_mat, double &det, double &confi);

  void RsAnnotation (uima::CAS &tcas, std::string class_name, std::string feature_name, std::string database_name, rs::Cluster &cluster, std::string set_mode, double &confi);

#if CV_MAJOR_VERSION == 3
  // StatModel interface
public:
  int getVarCount() const;
  bool isTrained() const;
  bool isClassifier() const;
  float predict(cv::InputArray samples, cv::OutputArray results, int flags) const;

  // SVM interface
public:
  int getType() const;
  void setType(int val);
  double getGamma() const;
  void setGamma(double val);
  double getCoef0() const;
  void setCoef0(double val);
  double getDegree() const;
  void setDegree(double val);
  double getC() const;
  void setC(double val);
  double getNu() const;
  void setNu(double val);
  double getP() const;
  void setP(double val);
  cv::Mat getClassWeights() const;
  void setClassWeights(const cv::Mat &val);
  cv::TermCriteria getTermCriteria() const;
  void setTermCriteria(const cv::TermCriteria &val);
  int getKernelType() const;
  void setKernel(int kernelType);
  void setCustomKernel(const cv::Ptr<Kernel> &_kernel);
  bool trainAuto(const cv::Ptr<cv::ml::TrainData> &data, int kFold, cv::ml::ParamGrid Cgrid, cv::ml::ParamGrid gammaGrid, cv::ml::ParamGrid pGrid, cv::ml::ParamGrid nuGrid, cv::ml::ParamGrid coeffGrid, cv::ml::ParamGrid degreeGrid, bool balanced);
  cv::Mat getSupportVectors() const;
  double getDecisionFunction(int i, cv::OutputArray alpha, cv::OutputArray svidx) const;
#endif

  ~ RSSVM();


};

#endif
