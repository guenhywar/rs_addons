// Developed by: Rakib

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <map>

#include <yaml-cpp/yaml.h>
#include <ros/package.h>

#include <opencv2/opencv.hpp>
#if CV_MAJOR_VERSION == 2
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#elif CV_MAJOR_VERSION == 3
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#endif

#include <rs/scene_cas.h>
#include <rs/types/all_types.h>
#include <rs/scene_cas.h>
#include <rs/utils/time.h>
#include <rs/DrawingAnnotator.h>

#include <rs_addons/classifiers/RSKNN.h>
#include <uima/api.hpp>

using namespace cv;

//..............................k-Nearest Neighbor Classifier.........................................
RSKNN::RSKNN()
{
}

void RSKNN:: trainModel(std::string train_matrix_name, std::string train_label_name,std::string train_label_n)
{
}

void RSKNN:: classify(std::string trained_file_name_saved, std::string test_matrix_name, std::string test_label_name, std::string obj_classInDouble)
{
}

void RSKNN::classifyOnLiveData(std::string trained_file_name_saved, cv::Mat test_mat, double &det, double &confi)
{
}

void RSKNN:: classifyKNN(std::string train_matrix_name, std::string train_label_name,
                         std::string test_matrix_name, std::string test_label_name, std::string obj_classInDouble, int default_k)
{
  //To load the train data................
  cv::Mat train_matrix;
  cv::Mat train_label;
  readDescriptorAndLabel(train_matrix_name, train_label_name, train_matrix, train_label);
  std::cout << "size of train matrix:" << train_matrix.size() << std::endl;
  std::cout << "size of train label:" << train_label.size() << std::endl;


  //To load the test data.............................
  cv::Mat test_matrix;
  cv::Mat test_label;
  readDescriptorAndLabel(test_matrix_name, test_label_name, test_matrix, test_label);
  std::cout << "size of test matrix :" << test_matrix.size() << std::endl;
  std::cout << "size of test label" << test_label.size() << std::endl;

#if CV_MAJOR_VERSION == 2
  CvKNearest* knncalld = new CvKNearest;

  //Train the classifier...................................
  knncalld->train(train_matrix, train_label, cv::Mat(), false, default_k,false);

  //To get the value of k.............
  int k_max = knncalld->get_max_k();
  //cv::Mat neighborResponses, bestResponse, distances;

#elif CV_MAJOR_VERSION == 3
  cv::Ptr<cv::ml::KNearest> knncalld = cv::ml::KNearest::create();

  knncalld->setDefaultK(default_k);

  //Train the classifier
  //TODO: is it ROW_SAMPLE?
  knncalld->train(train_matrix, cv::ml::ROW_SAMPLE, train_label);

  int k_max = knncalld->getDefaultK();
#endif

  //convert test label matrix into a vector.......................
  std::vector<double> con_test_label;
  test_label.col(0).copyTo(con_test_label);

  //Container to hold the integer value of labels............................
  std::vector<int> actual_label;
  std::vector<int> predicted_label;

  for(int i = 0; i < test_label.rows; i++)
  {
#if CV_MAJOR_VERSION == 2
    // double res = knncls->find_nearest(test_matrix.row(i), k,bestResponse,neighborResponses,distances);
    double res = knncalld->find_nearest(test_matrix.row(i),k_max);
#elif CV_MAJOR_VERSION == 3

    double res = knncalld->findNearest(test_matrix.row(i), k_max, cv::noArray());
#endif

    int prediction = res;
    predicted_label.push_back(prediction);
    double lab = con_test_label[i];
    int actual_convert = lab;
    actual_label.push_back(actual_convert);
  }
  std::cout << "K-Nearest Neighbor Result :" << std::endl;
  evaluation(actual_label, predicted_label, obj_classInDouble);
}

void RSKNN::classifyOnLiveDataKNN(std::string train_matrix_name, std::string train_label_name, int default_k, cv::Mat test_mat, double &det)
{
  //To load the train data................
  cv::Mat train_matrix;
  cv::Mat train_label;
  readDescriptorAndLabel(train_matrix_name, train_label_name, train_matrix, train_label);
  std::cout << "size of train matrix:" << train_matrix.size() << std::endl;
  std::cout << "size of train label:" << train_label.size() << std::endl;

  //To load the test data and it's label.............................
  std::cout << "size of test matrix :" << test_mat.size() << std::endl;

#if CV_MAJOR_VERSION == 2
  CvKNearest* knncalldc = new CvKNearest;

  //Train the classifier...................................
  knncalldc->train(train_matrix, train_label, cv::Mat(), false, default_k, false);

  //To get the value of k.............
  int k_max = knncalldc->get_max_k();
  //cv::Mat neighborResponses, bestResponse, distances;

  //double res = knnclsa->find_nearest(test_mat, default_k, bestResponse, neighborResponses, distances);

  double res = knncalldc->find_nearest(test_mat,k_max);
  std::cout << "prediction class is :" << res << std::endl;
  det = res;
#elif CV_MAJOR_VERSION == 3
  cv::Ptr<cv::ml::KNearest> knncalldc = cv::ml::KNearest::create();

  knncalldc->setDefaultK(default_k);

  //Train the classifier
  //TODO: is it ROW_SAMPLE?
  knncalldc->train(train_matrix, cv::ml::ROW_SAMPLE, train_label);

  int k_max = knncalldc->getDefaultK();

  double res = knncalldc->findNearest(test_mat, k_max, cv::noArray());
  std::cout << "prediction class is :" << res << std::endl;
  det = res;
#endif
}

void  RSKNN::processPCLFeatureKNN(std::string train_matrix_name,std::string train_label_name,std::string set_mode, int default_k,std::string dataset_use,std::string feature_use,
                                  std::vector<rs::Cluster> clusters, RSKNN *obj_VFH, cv::Mat &color,std::vector<std::string> models_label, uima::CAS &tcas)
{
  outInfo("Number of cluster:" << clusters.size() << std::endl);

  for(size_t i = 0; i < clusters.size(); ++i)
  {
    rs::Cluster &cluster = clusters[i];
    std::vector<rs::PclFeature> features;
    cluster.annotations.filter(features);

    for(size_t j = 0; j < features.size(); ++j)
    {
      rs::PclFeature &feats = features[j];
      outInfo("type of feature:" << feats.feat_type() << std::endl);
      std::vector<float> featDescriptor = feats.feature();
      outInfo("Size after conversion:" << featDescriptor.size());
      cv::Mat test_mat(1, featDescriptor.size(), CV_32F);

      for(size_t k = 0; k < featDescriptor.size(); ++k)
      {
        test_mat.at<float>(0, k) = featDescriptor[k];
      }
      outInfo("number of elements in :" << i << std::endl);

      double classLabel;
      double confi;
      obj_VFH->classifyOnLiveDataKNN(train_matrix_name, train_label_name, default_k, test_mat, classLabel);
      int classLabelInInt = classLabel;
      std::string classLabelInString = models_label[classLabelInInt-1];

      //To annotate the clusters..................
      RsAnnotation (tcas,classLabelInString,feature_use, dataset_use, cluster,set_mode, confi);

      //set roi on image
      rs::ImageROI image_roi = cluster.rois.get();
      cv::Rect rect;
      rs::conversion::from(image_roi.roi_hires.get(), rect);

      //Draw result on image...........
      obj_VFH->drawCluster(color, rect, classLabelInString);

      outInfo("calculation is done" << std::endl);
    }
  }
}

void  RSKNN::processCaffeFeatureKNN(std::string train_matrix_name, std::string train_label_name,
                                    std::string set_mode, int default_k, std::string dataset_use, std::string feature_use, std::vector<rs::Cluster> clusters,
                                    RSKNN *obj_caffe, cv::Mat &color, std::vector<std::string> models_label, uima::CAS &tcas)
{
  //clusters comming from RS pipeline............................
  outInfo("Number of cluster:" << clusters.size() << std::endl);

  for(size_t i = 0; i < clusters.size(); ++i)
  {
    rs::Cluster &cluster = clusters[i];
    std::vector<rs::Features> features;
    cluster.annotations.filter(features);

    for(size_t j = 0; j < features.size(); ++j)
    {
      rs::Features &feats = features[j];
      outInfo("type of feature:" << feats.descriptorType() << std::endl);
      outInfo("size of feature:" << feats.descriptors << std::endl);
      outInfo("size of source:" << feats.source() << std::endl);

      //variable to store caffe feature..........
      cv::Mat featDescriptor;
      double classLabel;

      if(feats.source()=="Caffe")
      {
        rs::conversion::from(feats.descriptors(), featDescriptor);
        outInfo("Size after conversion:" << featDescriptor.size());

        //The function generate the prediction result................
        obj_caffe->classifyOnLiveDataKNN(train_matrix_name,train_label_name, default_k, featDescriptor, classLabel);

        //class label in integer, which is used as index of vector model_label.
        int classLabelInInt = classLabel;
        double confi;
        std::string classLabelInString = models_label[classLabelInInt-1];

        //To annotate the clusters..................
        RsAnnotation (tcas,classLabelInString,feature_use, dataset_use, cluster,set_mode, confi);

        //set roi on image
        rs::ImageROI image_roi = cluster.rois.get();
        cv::Rect rect;
        rs::conversion::from(image_roi.roi_hires.get(), rect);

        //Draw result on image...........................
        obj_caffe->drawCluster(color, rect, classLabelInString);
      }
      outInfo("calculation is done" << std::endl);
    }
  }
}

void RSKNN::RsAnnotation(uima::CAS &tcas, std::string class_name, std::string feature_name, std::string database_name, rs::Cluster &cluster, std::string set_mode, double &confi)
{
  rs::Classification classResult = rs::create<rs::Classification>(tcas);
  classResult.classname.set(class_name);
  classResult.classifier("k-Nearest Neighbor");
  classResult.featurename(feature_name);
  classResult.model(database_name);
  if(feature_name == "CNN")
  {
    classResult.classification_type("INSTANCE");
  } else if(feature_name == "VFH")
  {
    classResult.classification_type("SHAPE");
  }

  if(set_mode == "CL")
  {
    cluster.annotations.append(classResult);
  }
  else if(set_mode == "GT")
  {
    rs::GroundTruth setGT = rs::create<rs::GroundTruth>(tcas);
    setGT.classificationGT.set(classResult);
    cluster.annotations.append(setGT);
  }
  else
  {
    outError("You should set the parameter (set_mode) as CL or GT");
  }
}

RSKNN::~ RSKNN()
{
}
