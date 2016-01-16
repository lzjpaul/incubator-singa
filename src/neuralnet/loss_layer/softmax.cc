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


#include <glog/logging.h>
#include <algorithm>
#include "singa/neuralnet/loss_layer.h"
#include "mshadow/tensor.h"
#include "singa/utils/math_blob.h"
#include "singa/utils/cluster.h"
#include <time.h>
#include <fstream>
#include <iostream>

using namespace std;

namespace singa {

using namespace mshadow;
using mshadow::cpu;

using mshadow::Shape;
using mshadow::Shape1;
using mshadow::Shape2;
using mshadow::Tensor;

using std::vector;

void SoftmaxLossLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 2);
  LossLayer::Setup(proto, srclayers);
  data_.Reshape(srclayers[0]->data(this).shape());
  batchsize_ = data_.shape()[0];
  dim_ = data_.count() / batchsize_;
  topk_ = proto.softmaxloss_conf().topk();
  scale_ = proto.softmaxloss_conf().scale();
}

void SoftmaxLossLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  Shape<2> s = Shape2(batchsize_, dim_);
  Tensor<cpu, 2> prob(data_.mutable_cpu_data(), s);
  Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(), s);
  Softmax(prob, src);
  const auto& label = srclayers[1]->aux_data(this);
  const float* probptr = prob.dptr;
  float loss = 0, precision = 0;
  for (int n = 0; n < batchsize_; n++) {
    int ilabel = static_cast<int>(label[n]);
    //  CHECK_LT(ilabel,10);
    CHECK_GE(ilabel, 0);
    float prob_of_truth = probptr[ilabel];
    loss -= log(std::max(prob_of_truth, FLT_MIN));
    vector<std::pair<float, int> > probvec;
    for (int j = 0; j < dim_; ++j) {
      probvec.push_back(std::make_pair(probptr[j], j));
    }
    std::partial_sort(probvec.begin(), probvec.begin() + topk_, probvec.end(),
                      std::greater<std::pair<float, int> >());
    // check if true label is in top k predictions
    for (int k = 0; k < topk_; k++) {
      if (probvec[k].second == static_cast<int>(label[n])) {
        precision++;
        break;
      }
    }
    probptr += dim_;
  }
  CHECK_EQ(probptr, prob.dptr + prob.shape.Size());
  loss_ += loss * scale_ / (1.0f * batchsize_);
  accuracy_ += precision * scale_ / (1.0f * batchsize_);
  counter_++;
 
  const auto& labelptr = srclayers[1]->aux_data(this);
  probptr = prob.dptr; 
  // print AUC and result
  ofstream probmatout;
  ofstream labelout;
  // LOG(ERROR) << "flag&flag: " << (flag&flag);
  if ((flag&flag) == 36){
    auto cluster = Cluster::Get();
    probmatout.open(cluster->workspace()+"/prob.csv", ios::app);
    labelout.open(cluster->workspace()+"/label.csv", ios::app);
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
    // probmatout.open("/data/zhaojing/AUC/prob/version" + std::to_string(static_cast<int>(run_version_)) + ".csv", ios::app);    
    // labelout.open("/data/zhaojing/AUC/label/version" + std::to_string(static_cast<int>(run_version_)) + ".csv", ios::app);
    auto cluster = Cluster::Get();
    probmatout.open(cluster->workspace()+"/prob.csv", ios::app);
    labelout.open(cluster->workspace()+"/label.csv" + ".csv", ios::app);
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
}

void SoftmaxLossLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  const auto& label = srclayers[1]->aux_data();
  Blob<float>* gsrcblob = srclayers[0]->mutable_grad(this);
  Copy(data_, gsrcblob);
//  gsrcblob->CopyFrom(data_);
  float* gsrcptr = gsrcblob->mutable_cpu_data();
  for (int n = 0; n < batchsize_; n++) {
    gsrcptr[n*dim_ + static_cast<int>(label[n])] -= 1.0f;
  }
  Tensor<cpu, 1> gsrc(gsrcptr, Shape1(gsrcblob->count()));
  gsrc *= scale_ / (1.0f * batchsize_);
}
const std::string SoftmaxLossLayer::ToString(bool debug, int flag) {
  if (debug)
    return Layer::ToString(debug, flag);

  string disp = "Loss = " + std::to_string(loss_ / counter_)
    + ", accuracy = " + std::to_string(accuracy_ / counter_);
  counter_ = 0;
  loss_ = accuracy_ = 0;
  return disp;
}
}  // namespace singa
