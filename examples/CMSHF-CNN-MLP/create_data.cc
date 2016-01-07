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
// atof + add_data() + no pixel any more..
// image.clear_data()!!!;

#include <glog/logging.h>
#include <fstream>
#include <string>
#include <cstdint>
#include <iostream>

#include "singa/io/store.h"
#include "singa/proto/common.pb.h"

using std::string;

int kROWSize = -1; //modified here
int kCOLSize = -1; //modified here
int kNUHHFNBytes = kROWSize * kCOLSize;
int kTrainBatchSize = -1;  //all the records, modified here
int kTestBatchSize = -1;  //all the records, modified here
int kValidBatchSize = -1;  //all the records, modified here

void create_data(int kROWSize_argv, int kCOLSize_argv, int kTrainBatchSize_argv,
                 int kTestBatchSize_argv, int kValidBatchSize_argv,
                 const string& train_feature_filename, const string& train_label_filename,
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

  /*meta data ...*/
  std::cout << "need to check the last element of each row + how many rows" << std::endl;
  kROWSize = kROWSize_argv; //modified here
  kCOLSize = kCOLSize_argv; //modified here
  kNUHHFNBytes = kROWSize * kCOLSize;
  kTrainBatchSize = kTrainBatchSize_argv;  //all the records, modified here
  kTestBatchSize = kTestBatchSize_argv;  //all the records, modified here
  kValidBatchSize = kValidBatchSize_argv;  //all the records, modified here


  std::cout << "kROWSize: " << kROWSize << std::endl;
  std::cout << "kCOLSize: " << kCOLSize << std::endl;
  std::cout << "kNUHHFNBytes: " << kNUHHFNBytes << std::endl;
  std::cout << "kTrainBatchSize: " << kTrainBatchSize << std::endl;
  std::cout << "kTestBatchSize: " << kTestBatchSize << std::endl;
  std::cout << "kValidBatchSize: " << kValidBatchSize << std::endl;
  std::cout << "need to check the last element of each row + how many rows" << std::endl << std::endl << std::endl;
  /*meta data ...*/
  string value;
  int m; //read in label
  int label;
  float n; //read in NUHHF data vector
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
      n = atof(value.c_str());
      if (n > 256)
        std::cout << "number of counts greater than 256: " << n << std::endl;
      image.add_data(n);
      if (i < 10 && (itemid < 10 || itemid == (kTrainBatchSize-1)))
        std::cout << " data: " << n;
    }
    getline (trainffile, value, '\n');
    n = atof(value.c_str());
    if (itemid < 10){
      std::cout << " read last value: " << value;
      std::cout << " n: " << n;
    }

    if (itemid < 10 || itemid == (kTrainBatchSize-1))
      std::cout << "data: " << n;
    image.add_data(n);
    
    image.SerializeToString(&rec_buf);
    store->Write("train", rec_buf);

    for (int i = 0; i < kNUHHFNBytes; i++)
      mean.set_data(i, mean.data(i) + image.data(i));
    count += 1;
    image.clear_data();
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
      n = atof(value.c_str());
      if (n > 256)
        std::cout << "number of counts greater than 256: " << n << std::endl;
      image.add_data(n);
      if (i < 10 && (itemid < 10 || itemid == (kTestBatchSize-1)))
        std::cout << " data: " << n;
    }
    getline (testffile, value, '\n');
    n = atof(value.c_str());
    if (itemid < 10){
      std::cout << " read last value: " << value;
      std::cout << " n: " << n;
    }
    if (itemid < 10 || itemid == (kTestBatchSize-1))
      std::cout << "data: " << n;
    image.add_data(n);

    image.SerializeToString(&rec_buf);
    store->Write("test", rec_buf);
    image.clear_data();
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
      n = atof(value.c_str());
      if (n > 256)
        std::cout << "number of counts greater than 256: " << n << std::endl;
      image.add_data(n);
      if (i < 10 && (itemid < 10 || itemid == (kValidBatchSize-1)))
        std::cout << " data: " << n;
    }
    getline (validffile, value, '\n');
    n = atof(value.c_str());
    if (itemid < 10){
      std::cout << " read last value: " << value;
      std::cout << " n: " << n;
    }
    if (itemid < 10 || itemid == (kValidBatchSize-1))
      std::cout << "data: " << n;
    image.add_data(n);

    image.SerializeToString(&rec_buf);
    store->Write("valid", rec_buf);
    image.clear_data(); 
  }

  store->Flush();
  store->Close();


}

int main(int argc, char** argv) {
  if (argc != 13) {
    std::cout <<"Create train and test DataShard for Cifar dataset.\n"
      << "Usage:\n"
      << "    create_data.bin input_folder output_folder\n"
      << "Where the input folder should contain the binary batch files.\n";
  } else {
    google::InitGoogleLogging(argv[0]);
    create_data( atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), string(argv[6]), string(argv[7]), string(argv[8]), string(argv[9]), string(argv[10]), string(argv[11]), string(argv[12]));
  }
  return 0;
}
//./create_data.bin 12 1277 5207 3000 2000 /data/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_INOUTPATIENT_CNN_SAMPLE_CASE_train_data_norm_1.csv /data/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_INOUTPATIENT_CNN_SAMPLE_CASE_train_data_label_1.csv /data/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_INOUTPATIENT_CNN_SAMPLE_CASE_test_data_norm_1.csv /data/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_INOUTPATIENT_CNN_SAMPLE_CASE_test_label_1.csv /data/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_INOUTPATIENT_CNN_SAMPLE_CASE_valid_data_norm_1.csv /data/zhaojing/cnn/NUHALLCOND/subsample1/NUH_DS_SOC_READMIT_DIAG_INOUTPATIENT_CNN_SAMPLE_CASE_valid_label_1.csv /data/zhaojing/cnn/NUHALLCOND/subsample1/
