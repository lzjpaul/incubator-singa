/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

// can reuse str_buffer?


#include <glog/logging.h>
#include <fstream>
#include <string>
#include <cstdint>
#include <iostream>

#include "singa/io/store.h"
#include "singa/proto/common.pb.h"

using std::string;

const int kROWSize = 12; //modified here
const int kCOLSize = 631; //modified here
const int kNUHHFNBytes = kROWSize * kCOLSize;
const int kTrainBatchSize = 300;  //all the records, modified here
const int kTestBatchSize = 190;  //all the records, modified here
const int kValidBatchSize = 125;  //all the records, modified here

void create_data(const string& train_feature_filename, const string& train_label_filename,
                 const string& test_feature_filename, const string& test_label_filename,
                 const string& valid_feature_filename, const string& valid_label_filename,
                 const string& output_folder) {

  // Open files
  std::ifstream trainffile(train_feature_filename, std::ios::in | std::ios::binary); //feature file
  CHECK(trainffile) << "Unable to open file " << train_feature_filename;

  std::ifstream trainlfile(train_label_filename, std::ios::in | std::ios::binary);   //label file
  CHECK(trainlfile) << "Unable to open file " << train_label_filename;

  // Open files
  std::ifstream testffile(test_feature_filename, std::ios::in | std::ios::binary); //feature file
  CHECK(testffile) << "Unable to open file " << test_feature_filename;

  std::ifstream testlfile(test_label_filename, std::ios::in | std::ios::binary);   //label file
  CHECK(testlfile) << "Unable to open file " << test_label_filename;

  // Open files
  std::ifstream validffile(valid_feature_filename, std::ios::in | std::ios::binary); //feature file
  CHECK(validffile) << "Unable to open file " << valid_feature_filename;

  std::ifstream validlfile(valid_label_filename, std::ios::in | std::ios::binary);   //label file
  CHECK(validlfile) << "Unable to open file " << valid_label_filename;

  string value;
  int m; //read in label
  int label;
  int n; //read in NUHHF data vector
  char str_buffer[kNUHHFNBytes];
  string rec_buf;

  singa::RecordProto image;
  image.add_shape(1);
  image.add_shape(kROWSize);
  image.add_shape(kCOLSize);

  singa::RecordProto mean;
  mean.CopyFrom(image);
  for (int i = 0; i < kNUHHFNBytes; i++)
    mean.add_data(0.f);

  auto store = singa::io::CreateStore("kvfile");
  CHECK(store->Open(output_folder + "/train_data.bin", singa::io::kCreate));
  LOG(INFO) << "Preparing training data";
  int count = 0;

  for (int itemid = 0; itemid < kTrainBatchSize; ++itemid) {
    // read_image(&data_file, &label, str_buffer);
    getline (trainlfile, value, '\n');
    m = atoi(value.c_str());
    label = (char)m;
    if (itemid < 10 || itemid == (kTrainBatchSize-1))
      std::cout << "train label: " << m << std::endl;
    image.set_label(label);

    for (int i = 0; i < (kROWSize * kCOLSize -1); i++){
      getline (trainffile, value, ',');
      n = atoi(value.c_str());
      if (n > 256)
        std::cout << "number of counts greater than 256: " << n << std::endl;
      str_buffer[i] = (char)n;
      if (i < 6 && (itemid < 10 || itemid == (kTrainBatchSize-1)))
        std::cout << " data: " << n;
    }
    getline (trainffile, value, '\n');
    n = atoi(value.c_str());
    if (itemid < 10 || itemid == (kTrainBatchSize-1))
      std::cout << "data: " << n;
    str_buffer[kROWSize * kCOLSize -1] = (char)n;
    image.set_pixel(str_buffer, kNUHHFNBytes);
    image.SerializeToString(&rec_buf);
    int length = snprintf(str_buffer, kNUHHFNBytes, "%05d", count);
    CHECK(store->Write(string(str_buffer, length), rec_buf));

    const string& pixels = image.pixel();
    for (int i = 0; i < kNUHHFNBytes; i++)
      mean.set_data(i, mean.data(i) + static_cast<uint8_t>(pixels[i]));
    count += 1;
  }

  store->Flush();
  store->Close();

  LOG(INFO) << "Create image mean";
  store->Open(output_folder + "/image_mean.bin", singa::io::kCreate);
  for (int i = 0; i < kNUHHFNBytes; i++)
    mean.set_data(i, mean.data(i) / count);
  mean.SerializeToString(&rec_buf);
  store->Write("mean", rec_buf);
  store->Flush();
  store->Close();

  LOG(INFO) << "Create test data";
  store->Open(output_folder + "/test_data.bin", singa::io::kCreate);
  for (int itemid = 0; itemid < kTestBatchSize; ++itemid) {
    // read_image(&data_file, &label, str_buffer);
    getline (testlfile, value, '\n');
    m = atoi(value.c_str());
    label = (char)m;
    if (itemid < 10 || itemid == (kTestBatchSize-1))
      std::cout << "test label: " << m << std::endl;
    image.set_label(label);

    for (int i = 0; i < (kROWSize * kCOLSize -1); i++){
      getline (testffile, value, ',');
      n = atoi(value.c_str());
      if (n > 256)
        std::cout << "number of counts greater than 256: " << n << std::endl;
      str_buffer[i] = (char)n;
      if (i < 6 && (itemid < 10 || itemid == (kTestBatchSize-1)))
        std::cout << " data: " << n;
    }
    getline (testffile, value, '\n');
    n = atoi(value.c_str());
    if (itemid < 10 || itemid == (kTestBatchSize-1))
      std::cout << "data: " << n;
    str_buffer[kROWSize * kCOLSize -1] = (char)n;
    image.set_pixel(str_buffer, kNUHHFNBytes);
    image.SerializeToString(&rec_buf);
    int length = snprintf(str_buffer, kNUHHFNBytes, "%05d", itemid);
    CHECK(store->Write(string(str_buffer, length), rec_buf));
  }

  store->Flush();
  store->Close();

  LOG(INFO) << "Create valid data";
  store->Open(output_folder + "/valid_data.bin", singa::io::kCreate);
  for (int itemid = 0; itemid < kValidBatchSize; ++itemid) {
    // read_image(&data_file, &label, str_buffer);
    getline (validlfile, value, '\n');
    m = atoi(value.c_str());
    label = (char)m;
    if (itemid < 10 || itemid == (kValidBatchSize-1))
      std::cout << "valid label: " << m << std::endl;
    image.set_label(label);

    for (int i = 0; i < (kROWSize * kCOLSize -1); i++){
      getline (validffile, value, ',');
      n = atoi(value.c_str());
      if (n > 256)
        std::cout << "number of counts greater than 256: " << n << std::endl;
      str_buffer[i] = (char)n;
      if (i < 6 && (itemid < 10 || itemid == (kValidBatchSize-1)))
        std::cout << " data: " << n;
    }
    getline (validffile, value, '\n');
    n = atoi(value.c_str());
    if (itemid < 10 || itemid == (kValidBatchSize-1))
      std::cout << "data: " << n;
    str_buffer[kROWSize * kCOLSize -1] = (char)n;
    image.set_pixel(str_buffer, kNUHHFNBytes);
    image.SerializeToString(&rec_buf);
    int length = snprintf(str_buffer, kNUHHFNBytes, "%05d", itemid);
    CHECK(store->Write(string(str_buffer, length), rec_buf));
  }

  store->Flush();
  store->Close();


}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cout <<"Create train and test DataShard for Cifar dataset.\n"
      << "Usage:\n"
      << "    create_data.bin input_folder output_folder\n"
      << "Where the input folder should contain the binary batch files.\n";
  } else {
    google::InitGoogleLogging(argv[0]);
    create_data(string(argv[1]), string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]), string(argv[6]), string(argv[7]));
  }
  return 0;
}
