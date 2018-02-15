//developed by: Rakib

#ifndef RSRF_HEADER
#define RSRF_HEADER

#include <iostream>
#include <string>

#include <uima/api.hpp>

#include <ros/package.h>
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
class RSRF : public RSClassifier, public CvRTrees
#elif CV_MAJOR_VERSION == 3
class RSRF : public RSClassifier, public cv::ml::RTrees
#endif
{

public:

  RSRF();

  void trainModel(std::string train_matrix_name, std::string train_label_name, std::string trained_file_name);

  int predict_multi_class(cv::Mat sample, cv::AutoBuffer<int>& out_votes);

  void classify(std::string trained_file_name,std::string test_matrix_name, std::string test_label_name,std::string obj_classInDouble);

  void classifyOnLiveData(std::string trained_file_name_saved, cv::Mat test_mat, double &det, double &confi);

  void RsAnnotation (uima::CAS &tcas, std::string class_name, std::string feature_name, std::string database_name, rs::Cluster &cluster, std::string set_mode, double &confi);

#if CV_MAJOR_VERSION == 3
  // StatModel interface
public:
  int getVarCount() const;
  bool isTrained() const;
  bool isClassifier() const;
  float predict(cv::InputArray samples, cv::OutputArray results, int flags) const;

  // DTrees interface
public:
  int getMaxCategories() const;
  void setMaxCategories(int val);
  int getMaxDepth() const;
  void setMaxDepth(int val);
  int getMinSampleCount() const;
  void setMinSampleCount(int val);
  int getCVFolds() const;
  void setCVFolds(int val);
  bool getUseSurrogates() const;
  void setUseSurrogates(bool val);
  bool getUse1SERule() const;
  void setUse1SERule(bool val);
  bool getTruncatePrunedTree() const;
  void setTruncatePrunedTree(bool val);
  float getRegressionAccuracy() const;
  void setRegressionAccuracy(float val);
  cv::Mat getPriors() const;
  void setPriors(const cv::Mat &val);
  const std::vector<int> &getRoots() const;
  const std::vector<cv::ml::DTrees::Node> &getNodes() const;
  const std::vector<cv::ml::DTrees::Split> &getSplits() const;
  const std::vector<int> &getSubsets() const;

  // RTrees interface
public:
  bool getCalculateVarImportance() const;
  void setCalculateVarImportance(bool val);
  int getActiveVarCount() const;
  void setActiveVarCount(int val);
  cv::TermCriteria getTermCriteria() const;
  void setTermCriteria(const cv::TermCriteria &val);
  cv::Mat getVarImportance() const;
#endif

  ~ RSRF();


};

#endif
