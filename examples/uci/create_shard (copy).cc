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
#include "proto/model.pb.h"

using singa::DataShard;
using singa::WriteProtoToBinaryFile;
using std::string;


void create_shard(const char* image_filename, const char* output) {
  // Open files
  std::ifstream file(image_filename, std::ios::in | std::ios::binary);
  CHECK(file) << "Unable to open file " << image_filename;

  string value;
  int n;
  // Read the magic and the meta data
  int num_items;
  int rows;
  int cols;
  getline (file, value, ',');  // to check whether int will overflow if there are too many records
  num_items = atoi (value.c_str());
  getline (file, value, ',');
  rows = atoi (value.c_str());
  getline (file, value, '\n');
  cols = atoi (value.c_str());

  

  DataShard shard(output, DataShard::kCreate);
  char label;
  char* pixels = new char[rows * cols];
  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];

  singa::Record record;
  singa::SingleLabelImageRecord* image=record.mutable_image();
  image->add_shape(rows);
  image->add_shape(cols);
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (int item_id = 0; item_id < num_items; ++item_id) {
    for (int i = 0; i < rows * cols; i++){
	getline (file, value, ',');
	n = atoi(value.c_str());
	pixels[i] = (char)n;
    }
    getline (file, value, '\n');
    n = atoi(value.c_str());
    label = (char)n;
    image->set_pixel(pixels, rows*cols);
    image->set_label(label);
    snprintf(key, kMaxKeyLength, "%08d", item_id);
    shard.Insert(string(key), record);
  }
  delete pixels;
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

  if (argc != 3) {
    std::cout<<"This program create a DataShard for a MNIST dataset\n"
        "Usage:\n"
        "    create_shard.bin  input_image_file input_label_file output_db_file\n"
        "The MNIST dataset could be downloaded at\n"
        "    http://yann.lecun.com/exdb/mnist/\n"
        "You should gunzip them after downloading.";
  } else {
    google::InitGoogleLogging(argv[0]);
    create_shard(argv[1], argv[2]);
  }
  return 0;
}
