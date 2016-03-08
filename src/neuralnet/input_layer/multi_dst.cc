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

/**
check:
group18dptr=group18_data_.mutable_cpu_data() + k * group18_dim_ * 12;
group1_data_.Reshape(vector<int>{batchsize, 1, 12, group1_dim_}); and previous k * group18_dim_ * 12 match?

**/
#include "singa/neuralnet/input_layer.h"
#include "mshadow/tensor.h"
namespace singa {

using std::string;
using std::vector;

void Multi_dstInputLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  SingleLabelRecordLayer::Setup(conf, srclayers);
  encoded_ = conf.store_conf().encoded();
  group1_dim_ = conf.store_conf().group1_dim();
  group2_dim_ = conf.store_conf().group2_dim();
  group3_dim_ = conf.store_conf().group3_dim();
  group4_dim_ = conf.store_conf().group4_dim();
  group5_dim_ = conf.store_conf().group5_dim();
  group6_dim_ = conf.store_conf().group6_dim();
  group7_dim_ = conf.store_conf().group7_dim();
  group8_dim_ = conf.store_conf().group8_dim();
  group9_dim_ = conf.store_conf().group9_dim();
  group10_dim_ = conf.store_conf().group10_dim();
  group11_dim_ = conf.store_conf().group11_dim();
  group12_dim_ = conf.store_conf().group12_dim();
  group13_dim_ = conf.store_conf().group13_dim();
  group14_dim_ = conf.store_conf().group14_dim();
  group15_dim_ = conf.store_conf().group15_dim();
  group16_dim_ = conf.store_conf().group16_dim();
  group17_dim_ = conf.store_conf().group17_dim();
  group18_dim_ = conf.store_conf().group18_dim();
  group19_dim_ = conf.store_conf().group19_dim();
  group20_dim_ = conf.store_conf().group20_dim();
  group21_dim_ = conf.store_conf().group21_dim();
  LOG(ERROR) << "batchsize: " << batchsize_;
  if (group1_dim_ != 0)
    group1_data_.Reshape(vector<int>{batchsize_, 1, 12, group1_dim_});
  if (group2_dim_ != 0)
    group2_data_.Reshape(vector<int>{batchsize_, 1, 12, group2_dim_});
  if (group3_dim_ != 0)
    group3_data_.Reshape(vector<int>{batchsize_, 1, 12, group3_dim_});
  if (group4_dim_ != 0)
    group4_data_.Reshape(vector<int>{batchsize_, 1, 12, group4_dim_});
  if (group5_dim_ != 0)
    group5_data_.Reshape(vector<int>{batchsize_, 1, 12, group5_dim_});
  if (group6_dim_ != 0)
    group6_data_.Reshape(vector<int>{batchsize_, 1, 12, group6_dim_});
  if (group7_dim_ != 0)
    group7_data_.Reshape(vector<int>{batchsize_, 1, 12, group7_dim_});
  if (group8_dim_ != 0)
    group8_data_.Reshape(vector<int>{batchsize_, 1, 12, group8_dim_});
  if (group9_dim_ != 0)
    group9_data_.Reshape(vector<int>{batchsize_, 1, 12, group9_dim_});
  if (group10_dim_ != 0)
    group10_data_.Reshape(vector<int>{batchsize_, 1, 12, group10_dim_});
  if (group11_dim_ != 0)
    group11_data_.Reshape(vector<int>{batchsize_, 1, 12, group11_dim_});
  if (group12_dim_ != 0)
    group12_data_.Reshape(vector<int>{batchsize_, 1, 12, group12_dim_});
  if (group13_dim_ != 0)
    group13_data_.Reshape(vector<int>{batchsize_, 1, 12, group13_dim_});
  if (group14_dim_ != 0)
    group14_data_.Reshape(vector<int>{batchsize_, 1, 12, group14_dim_});
  if (group15_dim_ != 0)
    group15_data_.Reshape(vector<int>{batchsize_, 1, 12, group15_dim_});
  if (group16_dim_ != 0)
    group16_data_.Reshape(vector<int>{batchsize_, 1, 12, group16_dim_});
  if (group17_dim_ != 0)
    group17_data_.Reshape(vector<int>{batchsize_, 1, 12, group17_dim_});
  if (group18_dim_ != 0)
    group18_data_.Reshape(vector<int>{batchsize_, 1, 12, group18_dim_});
  if (group19_dim_ != 0)
    group19_data_.Reshape(vector<int>{batchsize_, 1, 12, group19_dim_});
  if (group20_dim_ != 0)
    group20_data_.Reshape(vector<int>{batchsize_, 1, 12, group20_dim_});
  if (group21_dim_ != 0)
    group21_data_.Reshape(vector<int>{batchsize_, 1, 12, group21_dim_});
  test_sample_ = 0;
}

void Multi_dstInputLayer::LoadRecord(const string& backend,
    const string&path, Blob<float>* to) {
  io::Store* store = io::OpenStore(backend, path, io::kRead);
  string key, val;
  CHECK(store->Read(&key, &val));
  RecordProto image;
  image.ParseFromString(val);
  CHECK_EQ(to->count(), image.data_size());
  float* ptr = to->mutable_cpu_data();
  for (int i = 0; i< to->count(); i++)
    ptr[i] = image.data(i);
  delete store;
}

bool Multi_dstInputLayer::Parse(int k, int flag, const string& key,
    const string& value) {
  RecordProto image;
  image.ParseFromString(value);
  int size = data_.count() / batchsize_;
  // LOG(ERROR) << "size: " << size;
  // LOG(ERROR) << "key" << key;
  if (image.data_size() != 15036)
    LOG(ERROR) << "image.data_size(): " << image.data_size();
  if (image.data_size()) {
    if ( test_sample_ % 3000 == 0 )
      LOG(ERROR) << "begin step: " << (test_sample_)/batchsize_;
    CHECK_EQ(size, image.data_size());
    float* ptr = data_.mutable_cpu_data() + k * size;
    float* group1dptr;
    float* group2dptr;
    float* group3dptr;
    float* group4dptr;
    float* group5dptr;
    float* group6dptr;
    float* group7dptr;
    float* group8dptr;
    float* group9dptr;
    float* group10dptr;
    float* group11dptr;
    float* group12dptr;
    float* group13dptr;
    float* group14dptr;
    float* group15dptr;
    float* group16dptr;
    float* group17dptr;
    float* group18dptr;
    float* group19dptr;
    float* group20dptr;
    float* group21dptr;
    if (group1_dim_ != 0){
      group1dptr=group1_data_.mutable_cpu_data() + k * group1_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group2_dim_ != 0){
      group2dptr=group2_data_.mutable_cpu_data() + k * group2_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group3_dim_ != 0){
      group3dptr=group3_data_.mutable_cpu_data() + k * group3_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group4_dim_ != 0){
      group4dptr=group4_data_.mutable_cpu_data() + k * group4_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group5_dim_ != 0){
      group5dptr=group5_data_.mutable_cpu_data() + k * group5_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group6_dim_ != 0){
      group6dptr=group6_data_.mutable_cpu_data() + k * group6_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group7_dim_ != 0){
      group7dptr=group7_data_.mutable_cpu_data() + k * group7_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group8_dim_ != 0){
      group8dptr=group8_data_.mutable_cpu_data() + k * group8_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group9_dim_ != 0){
      group9dptr=group9_data_.mutable_cpu_data() + k * group9_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group10_dim_ != 0){
      group10dptr=group10_data_.mutable_cpu_data() + k * group10_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group11_dim_ != 0){
      group11dptr=group11_data_.mutable_cpu_data() + k * group11_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group12_dim_ != 0){
      group12dptr=group12_data_.mutable_cpu_data() + k * group12_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group13_dim_ != 0){
      group13dptr=group13_data_.mutable_cpu_data() + k * group13_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group14_dim_ != 0){
      group14dptr=group14_data_.mutable_cpu_data() + k * group14_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group15_dim_ != 0){
      group15dptr=group15_data_.mutable_cpu_data() + k * group15_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group16_dim_ != 0){
      group16dptr=group16_data_.mutable_cpu_data() + k * group16_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group17_dim_ != 0){
      group17dptr=group17_data_.mutable_cpu_data() + k * group17_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group18_dim_ != 0){
      group18dptr=group18_data_.mutable_cpu_data() + k * group18_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group19_dim_ != 0){
      group19dptr=group19_data_.mutable_cpu_data() + k * group19_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group20_dim_ != 0){
      group20dptr=group20_data_.mutable_cpu_data() + k * group20_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    if (group21_dim_ != 0){
      group21dptr=group21_data_.mutable_cpu_data() + k * group21_dim_ * 12;
    //LOG(ERROR)<<"diagdptr over";
    }
    int threshold_1 = group1_dim_;
    int threshold_2 = threshold_1 + group2_dim_;
    int threshold_3 = threshold_2 + group3_dim_;
    int threshold_4 = threshold_3 + group4_dim_;
    int threshold_5 = threshold_4 + group5_dim_;
    int threshold_6 = threshold_5 + group6_dim_;
    int threshold_7 = threshold_6 + group7_dim_;
    int threshold_8 = threshold_7 + group8_dim_;
    int threshold_9 = threshold_8 + group9_dim_;
    int threshold_10 = threshold_9 + group10_dim_;
    int threshold_11 = threshold_10 + group11_dim_;
    int threshold_12 = threshold_11 + group12_dim_;
    int threshold_13 = threshold_12 + group13_dim_;
    int threshold_14 = threshold_13 + group14_dim_;
    int threshold_15 = threshold_14 + group15_dim_;
    int threshold_16 = threshold_15 + group16_dim_;
    int threshold_17 = threshold_16 + group17_dim_;
    int threshold_18 = threshold_17 + group18_dim_;
    int threshold_19 = threshold_18 + group19_dim_;
    int threshold_20 = threshold_19 + group20_dim_;
    int threshold_21 = threshold_20 + group21_dim_;
    for (int i = 0; i < size; i++)
      ptr[i] = image.data(i);
    /*begin multi-dst*/
    int height = 12;
    int width = 1253;
    /*size = 15036 = 12 * 1253*/
    int j_dim = 0;
    for (int i = 0; i < size; i++){
      if (j_dim >= width)
        j_dim = 0;
      if (j_dim >= 0 && j_dim < threshold_1){
        *group1dptr = image.data(i);
        group1dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_1 && j_dim < threshold_2){
        *group2dptr = image.data(i);
        group2dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_2 && j_dim < threshold_3){
        *group3dptr = image.data(i);
        group3dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_3 && j_dim < threshold_4){
        *group4dptr = image.data(i);
        group4dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_4 && j_dim < threshold_5){
        *group5dptr = image.data(i);
        group5dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_5 && j_dim < threshold_6){
        *group6dptr = image.data(i);
        group6dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_6 && j_dim < threshold_7){
        *group7dptr = image.data(i);
        group7dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_7 && j_dim < threshold_8){
        *group8dptr = image.data(i);
        group8dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_8 && j_dim < threshold_9){
        *group9dptr = image.data(i);
        group9dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_9 && j_dim < threshold_10){
        *group10dptr = image.data(i);
        group10dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_10 && j_dim < threshold_11){
        *group11dptr = image.data(i);
        group11dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_11 && j_dim < threshold_12){
        *group12dptr = image.data(i);
        group12dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_12 && j_dim < threshold_13){
        *group13dptr = image.data(i);
        group13dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_13 && j_dim < threshold_14){
        *group14dptr = image.data(i);
        group14dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_14 && j_dim < threshold_15){
        *group15dptr = image.data(i);
        group15dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_15 && j_dim < threshold_16){
        *group16dptr = image.data(i);
        group16dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_16 && j_dim < threshold_17){
        *group17dptr = image.data(i);
        group17dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_17 && j_dim < threshold_18){
        *group18dptr = image.data(i);
        group18dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_18 && j_dim < threshold_19){
        *group19dptr = image.data(i);
        group19dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_19 && j_dim < threshold_20){
        *group20dptr = image.data(i);
        group20dptr++;
        j_dim ++;
      }
      else if (j_dim >= threshold_20 && j_dim < threshold_21){
        *group21dptr = image.data(i);
        group21dptr++;
        j_dim ++;
      }
    }
    /*end multi-dst*/
  } else if (image.pixel().size()) {
    CHECK_EQ(size, image.pixel().size());
    float* ptr = data_.mutable_cpu_data() + k * size;
    string pixel = image.pixel();
    for (int i = 0; i < size; i++)
      ptr[i] =  static_cast<float>(static_cast<uint8_t>(pixel[i]));
  } else {
    LOG(ERROR) << "not pixel nor pixel";
  }
  if ((flag & kDeploy) == 0) {  // deploy mode does not have label
    aux_data_.at(k) = image.label();
  }
  return true;
}

}  // namespace singa
