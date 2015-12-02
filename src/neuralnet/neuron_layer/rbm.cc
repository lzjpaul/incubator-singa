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

#include "singa/neuralnet/neuron_layer/rbm.h"

#include <glog/logging.h>
#include "singa/utils/singleton.h"

namespace singa {

using std::vector;

/**************** Implementation for RBMLayer********************/
Blob<float>* RBMLayer::Sample(int flag) {
  Tensor<cpu, 2> sample, data;
  if ((flag & kPositive) == kPositive || first_gibbs_) {
    data = Tensor2(&data_vector_.at(0));
    sample = Tensor2(&data_vector_.at(3));
  } else {
    data = Tensor2(&data_vector_.at(1));
    sample = Tensor2(&data_vector_.at(2));
  }
  auto random = TSingleton<Random<cpu>>::Instance();
  if (gaussian_) {
    random->SampleGaussian(sample, 0.0f, 1.0f);
    sample += data;
  } else {
    random->SampleBinary(sample, data);
  }
  return (flag & kPositive) == kPositive || first_gibbs_ ?
    &data_vector_.at(3) : &data_vector_.at(2);
}
void RBMLayer::Setup(const LayerProto& conf, const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  hdim_ = conf.rbm_conf().hdim();
  gaussian_ = conf.rbm_conf().gaussian();
  first_gibbs_ = true;
  //data_, neg_data_, neg_sample_, sample_
  Blob<float> new_data_;
  Blob<float> neg_data_;
  Blob<float> neg_sample_;
  Blob<float> sample_;
  data_vector_.push_back(new_data_);
  data_vector_.push_back(neg_data_);
  data_vector_.push_back(neg_sample_);
  data_vector_.push_back(sample_);
}
/**************** Implementation for RBMVisLayer********************/
RBMVisLayer::~RBMVisLayer() {
  delete weight_;
  delete bias_;
}

void RBMVisLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 2);
  RBMLayer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 2);
  hid_layer_ = nullptr;
  for (auto src : srclayers) {
    if (typeid(*src) == typeid(RBMHidLayer)) {
      // note the hid layer has may not been set up.
      CHECK(hid_layer_ == nullptr);
      hid_layer_ = dynamic_cast<RBMHidLayer*>(src);
    }
  }
  input_layer_ = srclayers[0] != hid_layer_ ? srclayers[0]: srclayers[1];
  const auto& src = input_layer_->data(this);
  batchsize_ = src.shape()[0];
  data_vector_.at(0).ReshapeLike(src);
  data_vector_.at(1).ReshapeLike(data_vector_.at(0));
  data_vector_.at(2).ReshapeLike(data_vector_.at(0));
  vdim_ = src.count() / batchsize_;
  weight_ = Param::Create(conf.param(0));
  weight_ ->Setup(vector<int>{hdim_, vdim_});
  bias_ = Param::Create(conf.param(1));
  bias_->Setup(vector<int>{vdim_});
}

void RBMVisLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  if ((flag & kPositive) == kPositive) {
    data_vector_.at(0).CopyFrom(input_layer_->data(this), true);
    first_gibbs_ = true;
  } else if ((flag & kNegative) == kNegative) {
    // fetch sampling results from hidden layer
    auto hid_sample = Tensor2(hid_layer_->Sample(flag));
    auto data = Tensor2(&data_vector_.at(1));
    auto weight = Tensor2(weight_->mutable_data());
    auto bias = Tensor1(bias_->mutable_data());
    data = dot(hid_sample, weight);
    data += expr::repmat(bias, batchsize_);
    data = expr::F<op::sigmoid>(data);
    if ((flag & kTest) == kTest) {
      const float *dptr = data_vector_.at(0).cpu_data(), *rcns = data_vector_.at(1).cpu_data();
      float err = 0.f;
      for (int i = 0; i < data_vector_.at(0).count(); i++) {
        err += (dptr[i] - rcns[i]) * (dptr[i] - rcns[i]);
      }
      metric_.Add("Squared Error", err / batchsize_);
    }
    first_gibbs_ = false;
  }
}

void RBMVisLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  auto vis_pos = Tensor2(&data_vector_.at(0));
  auto vis_neg = Tensor2(&data_vector_.at(1));
  auto hid_pos = Tensor2(hid_layer_->mutable_data(0)); //modify here
  auto hid_neg = Tensor2(hid_layer_->mutable_data(1));

  auto gbias = Tensor1(bias_->mutable_grad());
  gbias = expr::sum_rows(vis_neg);
  gbias -= expr::sum_rows(vis_pos);
  gbias /= batchsize_;

  auto gweight = Tensor2(weight_->mutable_grad());
  gweight = dot(hid_neg.T(), vis_neg);
  gweight -= dot(hid_pos.T(), vis_pos);
  gweight /= batchsize_;
}
/**************** Implementation for RBMHidLayer********************/
RBMHidLayer::~RBMHidLayer() {
  delete weight_;
  delete bias_;
}

void RBMHidLayer::Setup(const LayerProto& conf,
      const vector<Layer*>& srclayers) {
  RBMLayer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  const auto& src_data = srclayers[0]->data(0); //modify here
  batchsize_ = src_data.shape()[0];
  vdim_ = src_data.count() / batchsize_;
  data_vector_.at(0).Reshape(vector<int>{batchsize_, hdim_});
  data_vector_.at(1).ReshapeLike(data_vector_.at(0));
  data_vector_.at(3).ReshapeLike(data_vector_.at(0));
  data_vector_.at(2).ReshapeLike(data_vector_.at(0));
  weight_ = Param::Create(conf.param(0));
  weight_->Setup(vector<int>{hdim_, vdim_});
  bias_ = Param::Create(conf.param(1));
  bias_->Setup(vector<int>{hdim_});
  vis_layer_ = dynamic_cast<RBMVisLayer*> (srclayers[0]);
}

void RBMHidLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());

  Tensor<cpu, 2> data, src;
  if ((flag & kPositive) == kPositive) {
    data = Tensor2(&data_vector_.at(0));
    src = Tensor2(vis_layer_->mutable_data(0));  //modify here
    first_gibbs_ = true;
  } else {
    data = Tensor2(&data_vector_.at(1));
    // hinton's science paper does not sample the vis layer
    src = Tensor2(vis_layer_->mutable_data(1));
    first_gibbs_ = false;
  }
  data = dot(src, weight.T());
  data += expr::repmat(bias, batchsize_);

  if (!gaussian_)
    data = expr::F<op::sigmoid>(data);
}

void RBMHidLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  auto hid_pos = Tensor2(&data_vector_.at(0));
  auto hid_neg = Tensor2(&data_vector_.at(1));
  auto gbias = Tensor1(bias_->mutable_grad());
  gbias = expr::sum_rows(hid_neg);
  gbias -= expr::sum_rows(hid_pos);
  gbias /= batchsize_;
}

}  // namespace singa
