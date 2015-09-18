//
// This code creates DataShard for MNIST dataset.
// It is adapted from the convert_mnist_data from Caffe
//
// Usage:
//    create_shard.bin input_image_file input_label_file output_folder
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <cstdint>
#include <iostream>

#include <fstream>
#include <string>

#include "utils/data_shard.h"
#include "utils/common.h"
#include "proto/common.pb.h"

using singa::DataShard;
using singa::WriteProtoToBinaryFile;
using std::string;


void create_shard(int num_items_argv, int rows_argv, int cols_argv, const char* feature_filename, const char* label_filename, const char* output) {
  // Open files
  std::ifstream ffile(feature_filename, std::ios::in | std::ios::binary); //feature file
  CHECK(ffile) << "Unable to open file " << feature_filename;

  std::ifstream lfile(label_filename, std::ios::in | std::ios::binary);   //label file
  CHECK(lfile) << "Unable to open file " << label_filename;

  string value;
  float n; //read in vector element
  int m; //read in label
  // Read the magic and the meta data
  int num_items;
  int rows;
  int cols;
  num_items = num_items_argv;
  std::cout << "num_items: " << num_items;
  rows = rows_argv;
  std::cout << "rows: " << rows;
  cols = cols_argv;
  std::cout << "cols: " << cols;
  std::cout << "\n";



  DataShard shard(output, DataShard::kCreate);
  char label;
  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];

  singa::Record record;
  record.set_type(singa::Record::kSingleLabelVector);
  singa::SingleLabelVectorRecord* vector=record.mutable_vector();
  vector->add_shape(rows);
  vector->add_shape(cols);
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (int item_id = 0; item_id < num_items; ++item_id) {
    for (int i = 0; i < (rows * cols -1); i++){
	    getline (ffile, value, ',');
	    n = atof(value.c_str());
      if (i < 6 && (item_id < 10 || item_id == (num_items-1)))
        std::cout << " data: " << n;
	    vector->add_data(n);
	/*if (item_id < 10)
		 LOG(INFO) << "zj: item_id" << item_id << "element " << (int)pixels[i];*/
	/*	LOG(INFO) << StringPrintf("zj: item_id %d element %d\n", item_id,(int)pixels[i]);*/
    }
    getline (ffile, value, '\n');
	  n = atof(value.c_str());
    if (item_id < 10 || item_id == (num_items-1))
      std::cout << "data: " << n;
	  vector->add_data(n); //remember to add here !!!!!!

    getline (lfile, value, '\n');
    m = atoi(value.c_str());
    label = (char)m;
    if (item_id < 10 || item_id == (num_items-1))
      std::cout << "label: " << m << std::endl;
  /*  if (item_id < 10)
                 LOG(INFO) << "zj: item_id" << item_id << "label " << (int)label;*/
    /*image->set_pixel(pixels, rows*cols);*/
    vector->set_label(label);
    snprintf(key, kMaxKeyLength, "%08d", item_id);
    shard.Insert(string(key), record);
    vector->clear_data();
  }
  std::cout << "finish" << std::endl;
  shard.Flush();
}

int main(int argc, char** argv) {
/*
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage("This program create a DataShard for a MNIST dataset\n"
        "Usage:\n"
        "    create_shard.bin  input_image_file input_label_file output_db_file\n"
        "The MNIST dataset could be downloaded at\n"
        "    http://yann.lecun.com/exdb/mnist/\n"
        "You should gunzip them after downloading.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
    gflags::ShowUsageWithFlagsRestrict(argv[0],
        "examples/mnist/create_shard.bin");
*/

  if (argc != 7) {
    std::cout<<"This program create a DataShard for a MNIST dataset\n"
        "Usage:\n"
        "    create_shard.bin  input_image_file input_label_file output_db_file\n"
        "The MNIST dataset could be downloaded at\n"
        "    http://yann.lecun.com/exdb/mnist/\n"
        "You should gunzip them after downloading.";
  } else {
    google::InitGoogleLogging(argv[0]);
    create_shard( atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), argv[4], argv[5], argv[6] );
  }
  return 0;
}

//./create_shard.bin 65000 1 1143 /data/zhaojing/SynPUF-regularization/SynPUF_2009_Carrier_Claims_Vector_Regulariz_train_data_norm.csv
///data/zhaojing/SynPUF-regularization/SynPUF_2009_refer_2010_Car_Cla_Vec_Regulariz_train_label.csv
///data/zhaojing/SynPUF-regularization/SynPUF_2009_refer_2010_Car_Cla_Vec_Regulariz_train_shard/
