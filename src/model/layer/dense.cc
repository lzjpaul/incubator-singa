/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "./dense.h"
#include "singa/model/layer.h"
#include <vector>

namespace singa {
using std::vector;

Dense::~Dense() {
  // delete weight_;
  // delete bias_;
}
void Dense::Setup(const LayerConf &conf) {
  Layer::Setup(conf);
  DenseConf dense_conf = conf.dense_conf();
  hdim_ = dense_conf.num_output();
  transpose_ = dense_conf.transpose();
  // convert MB to bytes
  // workspace_byte_limit_ = dense_conf.workspace_byte_limit()
  //                        << 20;
}

void Dense::SetupParam(const Tensor &input) {
  const auto& srcshape = input.shape();
  int dim = srcshape.size();
  CHECK_EQ(dim, 2);
  batchsize_ = srcshape[0];
  vdim_ = srcshape[1];
  Shape shape{(size_t)batchsize_, hdim_};
  if (transpose_)
    weight_.Reshape(Shape{vdim_, hdim_});
  else
    weight_.Reshape(Shape{hdim_, vdim_});
  bias_.Reshape(Shape{hdim_});
  param_values_.push_back(&weight_);
  param_values_.push_back(&bias_);
}  

/// \copydoc Layer::Forward(int flag, const Tensor&)
const Tensor Dense::Forward(int flag, const Tensor &input) {
  Tensor output;
  SetupParam(input);
  if (transpose_)
    output = Mult(input, weight_);
  else
    output = Mult(input, weight_.T());
  AddRow(bias_, &output);
  buf_.push(input);
  return output;
}

/// \copydoc Layer::Backward(int, const Tensor&, const Tensor&);
const std::pair<Tensor, vector<Tensor>>
Dense::Backward(int flag, const Tensor &grad) {
  vector<Tensor> param_grad;
  Tensor src_data = buf_.top();
  buf_.pop();
  Tensor dbias, dweight, dx;
  dbias.ResetLike(bias_);
  dweight.ResetLike(weight_);
  dx.ResetLike(src_data);
  SumRows(grad, &dbias);
  if (transpose_)
    dweight = Mult(src_data.T(), grad);
  else
    dweight = Mult(grad.T(), src_data);
  if (transpose_)
    dx = Mult(grad, weight_.T());
  else
    dx = Mult(grad, weight_);
  param_grad.push_back(dweight);
  param_grad.push_back(dbias);
  return std::make_pair(dx, param_grad);
}

void Dense::ToDevice(Device *device) { 
  Layer::ToDevice(device);  
}

} // namespace singa
