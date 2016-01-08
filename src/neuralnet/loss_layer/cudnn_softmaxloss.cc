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
/*whether flag&&flag can help to distinguishe data of different phase to push
  previously, push data into vector, is not correct?*/
#include "singa/neuralnet/loss_layer.h"
#include "singa/utils/blob.h"
#include "singa/utils/math_blob.h"
#include "singa/utils/math_kernel.h"
#include <time.h>
#include <fstream>
#include <iostream>

using namespace std;

namespace singa {
void CudnnSoftmaxLossLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  LossLayer::Setup(conf, srclayers);
  softmax_.Setup(conf, vector<Layer*> {srclayers.at(0)});
  data_.Reshape(softmax_.data(this).shape());
  data_.ShareData(softmax_.mutable_data(this), false);
  batchsize_ = data_.shape(0);
  dim_ = data_.count() / batchsize_;
  srand((unsigned)time(NULL));
  run_version_ = rand()%1000;
}
void CudnnSoftmaxLossLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  softmax_.ComputeFeature(flag, srclayers);
  Blob<int> label(batchsize_);
  int *labelptr = label.mutable_cpu_data();
  // aux_data: vector<int>, convert vector to int array.
  for (int i = 0; i < batchsize_; ++i) {
    labelptr[i] = srclayers[1]->aux_data(this)[i];
  }
  float *probptr = data_.mutable_cpu_data(); //release?
  Blob<float> loss(batchsize_);
  singa_gpu_softmaxloss_forward(batchsize_, dim_, data_.gpu_data(),
      label.gpu_data(), loss.mutable_gpu_data());
  loss_ += Asum(loss);
  counter_++;

  // print AUC and result
  ofstream probmatout;
  ofstream labelout;
  // LOG(ERROR) << "flag&flag: " << (flag&flag);
  if ((flag&flag) == 36){
    probmatout.open("/data/zhaojing/AUC/prob/version" + std::to_string(static_cast<int>(run_version_)) + ".csv", ios::app);
    labelout.open("/data/zhaojing/AUC/label/version" + std::to_string(static_cast<int>(run_version_)) + ".csv", ios::app);
    for (int i = 0; i < batchsize_; i++){
      test_prob.push_back(probptr[2*i+1]); //two dimension!!
      probmatout << probptr[2*i+1] << "\n";
      test_label.push_back(static_cast<int>(labelptr[i]));
      labelout << static_cast<int>(labelptr[i]) << "\n";
      /*if (i < 100)
        LOG(ERROR) << "prob: " << probptr[2*i+1];*/
    }
    probmatout.close();
    labelout.close();
  }
  else if ((flag&flag) == 34){
    probmatout.open("/data/zhaojing/AUC/prob/version" + std::to_string(static_cast<int>(run_version_)) + ".csv", ios::app);    
    labelout.open("/data/zhaojing/AUC/label/version" + std::to_string(static_cast<int>(run_version_)) + ".csv", ios::app);
    for (int i = 0; i < batchsize_; i++){
      valid_prob.push_back(probptr[2*i+1]); //two dimension!!
      probmatout << probptr[2*i+1] << "\n";
      valid_label.push_back(static_cast<int>(labelptr[i]));
      labelout << static_cast<int>(labelptr[i]) << "\n";
      /*if (i < 100)
        LOG(ERROR) << "prob: " << probptr[2*i+1];*/
    }
    probmatout.close();
    labelout.close();
  }

  /*if ((flag&flag) == 36)
    LOG(ERROR) << "test_label vector size: " << test_label.size();*/
  if ((flag&flag) == 36 && test_label.size() == 3000){
    int tol_sample = 0;
    int correct_sample = 0;
    float test_accuracy = 0.0;
    for (int i = 0; i < 3000; i++){
      if ( (( static_cast<float>(test_prob.at(i)) < (1.0f - static_cast<float>(test_prob.at(i)))) && test_label.at(i) == 0) 
            || ((static_cast<float>(test_prob.at(i)) >= (1.0f - static_cast<float>(test_prob.at(i)))) && test_label.at(i) == 1) )
        correct_sample ++;
      tol_sample ++;
    }
    test_accuracy = correct_sample / (1.0f * tol_sample);
    LOG(ERROR) << "tol test sample: " << tol_sample << " accuracy: " << test_accuracy;
    test_prob.clear();
    test_label.clear();
  }
  else if ((flag&flag) == 34 && valid_label.size() == 2000){
    int valid_tol_sample = 0;
    int valid_correct_sample = 0;
    float valid_accuracy = 0.0;
    for (int i = 0; i < 2000; i++){
      if ( (( static_cast<float>(valid_prob.at(i)) < (1.0f - static_cast<float>(valid_prob.at(i)))) && valid_label.at(i) == 0)
            || ((static_cast<float>(valid_prob.at(i)) >= (1.0f - static_cast<float>(valid_prob.at(i)))) && valid_label.at(i) == 1) )
        valid_correct_sample ++;
      valid_tol_sample ++;
    }
    valid_accuracy = valid_correct_sample / (1.0f * valid_tol_sample);
    LOG(ERROR) << "tol valid sample: " << valid_tol_sample << " valid accuracy: " << valid_accuracy;
    valid_prob.clear();
    valid_label.clear();
  }
  /*for (int i = 0; i < batchsize_; i++){
    test_prob.push_back(probptr[2*i+1]); //two dimension!!
    test_label.push_back(labelptr[i]);
  }*/
}

void CudnnSoftmaxLossLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  Blob<float>* gsrcblob = srclayers[0]->mutable_grad(this);
  Copy(data_, gsrcblob);
  // gsrcblob->CopyFrom(data_);
  float* gsrcptr = gsrcblob->mutable_gpu_data();

  Blob<int> label(batchsize_);
  int *labelptr = label.mutable_cpu_data();

  // aux_data: vector<int>, convert vector to int array.
  for (int i = 0; i < batchsize_; ++i) {
    labelptr[i] = srclayers[1]->aux_data(this)[i];
  }

  singa_gpu_softmaxloss_backward(batchsize_, dim_, 1.0f, label.gpu_data(),
      gsrcptr);
  Scale(1.0f / batchsize_, gsrcblob);
}

const std::string CudnnSoftmaxLossLayer::ToString(bool debug, int flag) {
  if (debug)
    return Layer::ToString(debug, flag);

  string disp = "Loss = " + std::to_string(loss_ / counter_);
  counter_ = 0;
  loss_ = 0;
  return disp;
}
}  // namespace singa
