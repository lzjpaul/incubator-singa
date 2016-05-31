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
#include "../src/model/layer/dense.h"
#ifdef USE_CUDNN

#include "gtest/gtest.h"

using singa::Dense;
TEST(Dense, Setup) {
  Dense dense;
  EXPECT_EQ("Dense", dense.layer_type());

  singa::LayerConf conf;
  singa::DenseConf *denseconf = conf.mutable_dense_conf();
  denseconf->set_num_output(3);
  denseconf->set_transpose(false);
  dense.Setup(conf);

  EXPECT_EQ(3, dense.num_output());
  // EXPECT_EQ(false, dense.transpose());
}

TEST(Dense, Forward) {
  Dense dense;

  singa::LayerConf conf;
  singa::DenseConf *denseconf = conf.mutable_dense_conf();
  denseconf->set_num_output(3);
  denseconf->set_transpose(false);
  dense.Setup(conf);

  const size_t batchsize = 3, vdim = 2, hdim = 3;
  const float x[batchsize * vdim] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                      6.0f};
  singa::CudaGPU cuda(0, 1);
  singa::Tensor in(singa::Shape{batchsize, vdim}, &cuda);
  in.CopyDataFromHostPtr(x, batchsize * vdim);

  // set weight
  const float we[hdim * vdim] = {
      1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 1.0f};
  singa::Tensor weight(singa::Shape{hdim * vdim}, &cuda);
  weight.CopyDataFromHostPtr(we, hdim * vdim);

  const float bia[hdim] = {
      1.0f, 1.0f, 1.0f};
  singa::Tensor bias(singa::Shape{hdim}, &cuda);
  bias.CopyDataFromHostPtr(bia, hdim);
  
  dense.set_weight(weight);
  dense.set_bias(bias);

  singa::Tensor out1 = dense.Forward(singa::kTrain, in);
  singa::CppCPU host(0, 1);
  out1.ToDevice(&host);
  const float *outptr1 = out1.data<const float *>();
  EXPECT_EQ(9, out1.Size());
  EXPECT_EQ(4.0f, outptr1[0]);
  EXPECT_EQ(6.0f, outptr1[1]);
  EXPECT_EQ(3.0f, outptr1[2]);
  EXPECT_EQ(8.0f, outptr1[3]);
  EXPECT_EQ(12.0f, outptr1[4]);
  EXPECT_EQ(5.0f, outptr1[5]);
  EXPECT_EQ(12.0f, outptr1[6]);
  EXPECT_EQ(18.0f, outptr1[7]);
  EXPECT_EQ(7.0f, outptr1[8]);
}

TEST(Dense, Backward) {
  const size_t batchsize = 3, vdim = 2, hdim = 3;
  const float x[batchsize * vdim] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                      6.0f};
  singa::CudaGPU cuda(0, 1);
  singa::Tensor in(singa::Shape{batchsize, vdim}, &cuda);
  in.CopyDataFromHostPtr(x, batchsize * vdim);

  // set weight
  const float we[hdim * vdim] = {
      1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 1.0f};
  singa::Tensor weight(singa::Shape{hdim * vdim}, &cuda);
  weight.CopyDataFromHostPtr(we, hdim * vdim);
  
  const float bia[hdim] = {
      1.0f, 1.0f, 1.0f};
  singa::Tensor bias(singa::Shape{hdim}, &cuda);
  bias.CopyDataFromHostPtr(bia, hdim);

  Dense dense;
  dense.set_weight(weight);
  dense.set_bias(bias);

  singa::LayerConf conf;
  singa::DenseConf *denseconf = conf.mutable_dense_conf();
  denseconf->set_num_output(3);
  denseconf->set_transpose(false);
  dense.Setup(conf);

  singa::Tensor out1 = dense.Forward(singa::kTrain, in);

  // grad
  const float dy[batchsize * hdim] = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f};
  singa::Tensor grad(singa::Shape{batchsize, hdim}, &cuda);
  grad.CopyDataFromHostPtr(dy, batchsize * hdim);

  const auto ret = dense.Backward(singa::kTrain, grad);
  singa::CppCPU host(0, 1);
  singa::Tensor in_grad = ret.first;
  singa::Tensor dweight = ret.second.at(0);
  singa::Tensor dbias = ret.second.at(1);
  in_grad.ToDevice(&host);
  const float *dx = in_grad.data<const float *>();
  EXPECT_EQ(6, in_grad.Size());
  EXPECT_EQ(2.0f, dx[0]);
  EXPECT_EQ(4.0f, dx[1]);
  EXPECT_EQ(4.0f, dx[2]);
  EXPECT_EQ(8.0f, dx[3]);
  EXPECT_EQ(6.0f, dx[4]);
  EXPECT_EQ(12.0f, dx[5]);
  dweight.ToDevice(&host);
  const float *dweightx = dweight.data<const float *>();
  EXPECT_EQ(6, dweight.Size());
  EXPECT_EQ(22.0f, dweightx[0]);
  EXPECT_EQ(28.0f, dweightx[1]);
  EXPECT_EQ(22.0f, dweightx[2]);
  EXPECT_EQ(28.0f, dweightx[3]);
  EXPECT_EQ(22.0f, dweightx[4]);
  EXPECT_EQ(28.0f, dweightx[5]);
  dbias.ToDevice(&host);
  const float *dbiasx = dbias.data<const float *>();
  EXPECT_EQ(3, dbias.Size());
  EXPECT_EQ(6.0f, dbiasx[0]);
  EXPECT_EQ(6.0f, dbiasx[1]);
  EXPECT_EQ(6.0f, dbiasx[2]);
}
#endif  // USE_CUDNN
