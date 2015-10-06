#include <glog/logging.h>
#include <memory>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "mshadow/tensor.h"
#include "mshadow/cxxnet_op.h"
#include "neuralnet/layer.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include <time.h>
#include <fstream>
#include <iostream>

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

namespace singa {

/************ Implementation for ConvProductLayer*************************/
void ConvolutionLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  ConvolutionProto conv_conf=proto.convolution_conf();
  kernel_=conv_conf.kernel();
  CHECK_GT(kernel_, 0) << "Filter size cannot be zero.";
  pad_=conv_conf.pad();
  stride_=conv_conf.stride();
  num_filters_=conv_conf.num_filters();
  const vector<int>& srcshape=srclayers[0]->data(this).shape();
  int dim=srcshape.size();
  CHECK_GT(dim, 2);
  width_=srcshape[dim-1];
  height_=srcshape[dim-2];
  if(dim>3)
    channels_=srcshape[dim-3];
  else if(dim>2)
    channels_=1;
  batchsize_=srcshape[0];
  conv_height_=(height_ + 2 * pad_ - kernel_) / stride_ + 1;
  conv_width_= (width_ + 2 * pad_ - kernel_) / stride_ + 1;
  col_height_=channels_*kernel_*kernel_;
  col_width_=conv_height_*conv_width_;
  vector<int> shape{batchsize_, num_filters_, conv_height_, conv_width_};
  data_.Reshape(shape);
  grad_.Reshape(shape);
  col_data_.Reshape(vector<int>{col_height_, col_width_});
  col_grad_.Reshape(vector<int>{col_height_, col_width_});

  Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
  weight_=shared_ptr<Param>(factory->Create("Param"));
  weight_->Setup(proto.param(0), vector<int>{num_filters_, col_height_});
  bias_=shared_ptr<Param>(factory->Create("Param"));
  bias_->Setup(proto.param(1), vector<int>{num_filters_});
}

void ConvolutionLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  LayerProto newproto(proto);
  ConvolutionProto *conv_conf=newproto.mutable_convolution_conf();
  conv_conf->set_num_filters(shape[1]);
  Setup(newproto, srclayers);
}

void ConvolutionLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers){
  Tensor<cpu, 4> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape4(batchsize_, channels_, height_, width_));
  Tensor<cpu, 3> data(data_.mutable_cpu_data(),
      Shape3(batchsize_, num_filters_, conv_height_* conv_width_));
  Tensor<cpu, 2> col(col_data_.mutable_cpu_data(),
      Shape2(col_height_, col_width_));
  Tensor<cpu, 2> weight(weight_->mutable_cpu_data(),
      Shape2(num_filters_, col_height_));
  Tensor<cpu, 1> bias(bias_->mutable_cpu_data(),
      Shape1(num_filters_));

  for(int n=0;n<batchsize_;n++){
    if(pad_>0)
      col=unpack_patch2col(pad(src[n], pad_), kernel_, stride_);
    else
      col=unpack_patch2col(src[n], kernel_, stride_);
    data[n]=dot(weight, col);
  }
  data+=broadcast<1>(bias, data.shape);
}

void ConvolutionLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Tensor<cpu, 4> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape4(batchsize_, channels_, height_, width_));
  Tensor<cpu, 2> col(col_data_.mutable_cpu_data(),
      Shape2(col_height_, col_width_));
  Tensor<cpu, 2> weight(weight_->mutable_cpu_data(),
      Shape2(num_filters_, col_height_));

  Blob<float>* gsrcblob=srclayers[0]->mutable_grad(this);
  Tensor<cpu, 4> gsrc(nullptr, Shape4(batchsize_, channels_, height_, width_));
  if(gsrcblob!=nullptr)
    gsrc.dptr=gsrcblob->mutable_cpu_data();
  Tensor<cpu, 3> grad(grad_.mutable_cpu_data(),
      Shape3(batchsize_, num_filters_, conv_height_* conv_width_));
  Tensor<cpu, 2> gcol(col_grad_.mutable_cpu_data(),
      Shape2(col_height_, col_width_));
  Tensor<cpu, 2> gweight(weight_->mutable_cpu_grad(),
      Shape2(num_filters_, col_height_));
  Tensor<cpu, 1> gbias(bias_->mutable_cpu_grad(),
      Shape1(num_filters_));

  gweight=0.0f;
  gbias=sumall_except_dim<1>(grad);
  Shape<3> padshape(gsrc.shape.SubShape());
  padshape[0]+=2*pad_;padshape[1]+=2*pad_;
  Shape<2> imgshape=Shape2(height_, width_);
  for(int n=0;n<batchsize_;n++){
    if(pad_>0)
      col=unpack_patch2col(pad(src[n], pad_), kernel_, stride_);
    else
      col=unpack_patch2col(src[n], kernel_, stride_);
    gweight+=dot(grad[n], col.T());

    if(gsrcblob!=nullptr){
      gcol=dot(weight.T(), grad[n]);
      gsrc[n]=crop(pack_col2patch(gcol, padshape, kernel_, stride_), imgshape);
    }
  }
}

/****************** Implementation for DropoutLayer ***********************/
void DropoutLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  data_.ReshapeLike(srclayers[0]->data(this));
  grad_.ReshapeLike(*srclayers[0]->mutable_grad(this));
  mask_.Reshape(srclayers[0]->data(this).shape());
  pdrop_=proto.dropout_conf().dropout_ratio();
}

void DropoutLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}

void DropoutLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers) {
  // check training
  if(phase!= kTrain){//!training){
    data_.CopyFrom(srclayers[0]->data(this));
    return;
  }
  float pkeep=1-pdrop_;
  Tensor<cpu, 1> mask(mask_.mutable_cpu_data(), Shape1(mask_.count()));
  mask = F<op::threshold>(TSingleton<Random<cpu>>::Instance()\
      ->uniform(mask.shape), pkeep ) * (1.0f/pkeep);
  Tensor<cpu, 1> data(data_.mutable_cpu_data(), Shape1(data_.count()));
  Blob<float>* srcblob=srclayers[0]->mutable_data(this);
  Tensor<cpu, 1> src(srcblob->mutable_cpu_data(), Shape1(srcblob->count()));
  data=src*mask;
}

void DropoutLayer::ComputeGradient(const vector<SLayer>& srclayers)  {
  Tensor<cpu, 1> grad(grad_.mutable_cpu_data(), Shape1(data_.count()));
  Tensor<cpu, 1> mask(mask_.mutable_cpu_data(), Shape1(mask_.count()));
  Blob<float>* gsrcblob=srclayers[0]->mutable_grad(this);
  Tensor<cpu, 1> gsrc(gsrcblob->mutable_cpu_data(), Shape1(gsrcblob->count()));
  gsrc=grad*mask;
}

/**************** Implementation for MultiSrcSingleLayer********************/
//only assume there is single element from each src, output is single element
void MultiSrcSingleLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  //LOG(ERROR)<<"MultiSrcSingle setup begins";
  srclayer_num_ = srclayers.size();
  vdim_ = srclayer_num_; //each src one element
  const auto& src=srclayers[0]->data(this);
  batchsize_=src.shape()[0];
  hdim_=proto.multisrcsingle_conf().num_output();
  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
  weight_=shared_ptr<Param>(factory->Create("Param"));
  bias_=shared_ptr<Param>(factory->Create("Param"));
  weight_->Setup(proto.param(0), vector<int>{vdim_, hdim_});
  bias_->Setup(proto.param(1), vector<int>{hdim_});
  //LOG(ERROR)<<"MultiSrcSingle setup ends";
  /*Tensor<cpu, 2> weight(weight_->mutable_cpu_data(), Shape2(vdim_,hdim_));
  float* weighttest = weight.dptr;
  weighttest[0] = 0.33;
  weighttest[1] = 0.33;
  weighttest[2] = 0.33;*/
}
void MultiSrcSingleLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  LOG(ERROR)<<"No partition after setup ";
}

void MultiSrcSingleLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers) {
  float* srcdptr;
  float* concat_srcdptr;
  Tensor<cpu, 2> data(data_.mutable_cpu_data(), Shape2(batchsize_,hdim_));
  //CHECK_EQ(srclayers[0]->data(this).count(), batchsize_*vdim_);
  /*Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape2(batchsize_,vdim_));*/
  Tensor<cpu, 2> concat_src(Shape2(batchsize_, vdim_));
  AllocSpace(concat_src);
  concat_srcdptr = concat_src.dptr;
  for (int i = 0; i < vdim_; i++){  //traverse all sources
    srcdptr = srclayers[i]->mutable_data(this)->mutable_cpu_data(); //can pass data up?
    for (int j = 0; j < batchsize_; j++){
      concat_srcdptr[i + vdim_ * j] = srcdptr[1 + 2*j];
      /*if (j < 10)
        LOG(INFO)<<"srcno: "<<i<<" src data: "<<srcdptr[1 + 2*j];*/
    }
      //only fetch the y1 from srclayer!!!!!!!!!!!!!!!!!!!!!
  }


  Tensor<cpu, 2> weight(weight_->mutable_cpu_data(), Shape2(vdim_,hdim_));
  Tensor<cpu, 1> bias(bias_->mutable_cpu_data(), Shape1(hdim_));
  float* weighttest = weight.dptr;
  float* biastest = bias.dptr;
  LOG(INFO)<<"weight: "<<weighttest[0]<<" "<<weighttest[1]<<" "<<weighttest[2];
  LOG(INFO)<<"hdim_: "<<hdim_<<" vdim_: "<<vdim_;
  LOG(INFO)<<"bias: "<<biastest[0];
  //float* datatest = data.dptr;
  //float* concattest = concat_src.dptr;
  /*for (int i = 0; i < 10; i++){
    LOG(INFO)<<"concattest: "<<concattest[0]<<" "<<concattest[1]<<" "<<concattest[2];
    concattest+=vdim_;
  }*/
  //LOG(INFO)<<"batchsize: "<<batchsize_;
  concat_srcdptr = concat_src.dptr;
  float* datadptr = data.dptr;
  float* weightdptr = weight.dptr;
  for (int i = 0; i < batchsize_; i++){
    datadptr[i] = 0.0f;
    for (int j = 0; j < vdim_; j++)
      datadptr[i] += concat_srcdptr[i * vdim_ + j] * weightdptr[j];
  }
  //data=dot(concat_src, weight);
  // repmat: repeat bias vector into batchsize rows
  /*for (int i = 0; i < 10; i++){
    LOG(INFO)<<"before repmat datatest: "<<data[i][0];
    //datatest+=hdim_;
  }*/
  data+=repmat(bias, batchsize_);
  //LOG(INFO)<<"weight: "<<weighttest[0];
  //concattest = concat_src.dptr;
  //datatest = data.dptr;
  /*for (int i = 0; i < 30; i++){
    LOG(INFO)<<"concattest: "<<concattest[0]<<" "<<concattest[1]<<" "<<concattest[2];
    concattest+=vdim_;
    LOG(INFO)<<"after repmat datatest: "<<datatest[i];
    //datatest+=hdim_;
  }*/
  FreeSpace(concat_src);
}

void MultiSrcSingleLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  /*Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape2(batchsize_,vdim_));*/
  Tensor<cpu, 2> grad(grad_.mutable_cpu_data(),Shape2(batchsize_,hdim_));
  Tensor<cpu, 2> weight(weight_->mutable_cpu_data(), Shape2(vdim_,hdim_));
  Tensor<cpu, 2> gweight(weight_->mutable_cpu_grad(), Shape2(vdim_,hdim_));//3*1
  Tensor<cpu, 1> gbias(bias_->mutable_cpu_grad(), Shape1(hdim_));
  float* graddptr = grad.dptr;
  float* weightdptr = weight.dptr;
  float* gweightdptr = gweight.dptr;
  float* srcdptr;
  float* gsrcdptr;
  gbias=sum_rows(grad);
  //gweight=dot(src.T(), grad);
  for (int i = 0; i < vdim_; i++){  //traverse all sources
    srcdptr = srclayers[i]->mutable_data(this)->mutable_cpu_data();
    gweightdptr[i] = 0.0f;
    for (int j = 0; j < batchsize_; j++){
      gweightdptr[i] += srcdptr[1 + 2*j] * graddptr[j];
      /*if (j < 20){
        LOG(INFO)<<"MultiSrcSingle gradient srcno"<<i<<"src data: "<<srcdptr[1 + 2*j]<<" grad: "<<graddptr[j]<<" gweight: "<<gweightdptr[i];
      }*/
    }
  }

  // gsrc=dot(grad, weight.T()); can refer to tanh layer gsrc=F<op::stanh_grad>(data)*grad;
  for (int i = 0; i < vdim_; i++){  //traverse all sources
    Tensor<cpu, 2> gsrc(srclayers[i]->mutable_grad(this)->mutable_cpu_data(),//no need for the tensor?
        Shape2(batchsize_,1));
    gsrcdptr = gsrc.dptr; //each src grad one by one
    for (int j = 0; j < batchsize_; j++){
      gsrcdptr[j] = graddptr[j] * weightdptr[i];
      /*if (j < 10)
        LOG(INFO)<<"srcno: "<<i<<" gsrc: "<<gsrcdptr[j];*/
    }
  }
}

/**************** Implementation for InnerRegularizLayer********************/
void InnerRegularizLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){

  std::ifstream file(proto.innerregulariz_conf().path(), std::ios::in | std::ios::binary);
  CHECK(file) << "Unable to open file " << proto.innerregulariz_conf().path();
  LOG(ERROR)<<"similarity matrix path: "<< proto.innerregulariz_conf().path();

  LOG(ERROR)<<"innereg setup begin ";
  CHECK_EQ(srclayers.size(),1);
  LOG(ERROR)<<"layer name from inner reg layer"<<(this->name());
  const auto& src=srclayers[0]->data(this);
  LOG(ERROR)<<"fetch data ends ";
  batchsize_=src.shape()[0];
  LOG(ERROR)<<"batchsize "<<batchsize_;
  vdim_=src.count()/batchsize_;
  LOG(ERROR)<<"vdim_ "<<vdim_;
  hdim_=proto.innerregulariz_conf().num_output();
  LOG(ERROR)<<"hdim "<<hdim_;
  regdim_ = proto.innerregulariz_conf().regdim();
  regcoefficient_ = proto.innerregulariz_conf().regcoefficient();
  LOG(ERROR)<< "reg coefficient: "<<regcoefficient_;
  ////////////read in sim matrix
  similarity_matrix_.Reshape(vector<int>{regdim_, regdim_});
  float* simmatrix = similarity_matrix_.mutable_cpu_data();
  float sim_value;
  string value;
  int sim_index;
  for (sim_index = 0; sim_index < ((regdim_ * regdim_) - 1); sim_index++){
    getline (file, value, ',');
    if (strcmp((value).c_str(), "") == 0)
      LOG(ERROR)<<"empty string!!";
    sim_value = atof(value.c_str());
    simmatrix[sim_index] = sim_value;
    if (sim_index < 10)
      LOG(ERROR)<<"sim value: "<<sim_value;
  }
  LOG(ERROR)<<"sim_index: "<<sim_index;
  getline (file, value, '\n');
  sim_value = atof(value.c_str());
  simmatrix[sim_index] = sim_value;
  LOG(ERROR)<<"last sim value: "<<sim_value;
  //////////////read in sim matrix ends
  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  LOG(ERROR)<<"data_ and grad_ ";
  Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
  weight_=shared_ptr<Param>(factory->Create("Param"));
  bias_=shared_ptr<Param>(factory->Create("Param"));
  weight_->Setup(proto.param(0), vector<int>{vdim_, hdim_});
  bias_->Setup(proto.param(1), vector<int>{hdim_});
  LOG(ERROR)<<"innerreg setup end ";
}
void InnerRegularizLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  LOG(ERROR)<<"No partition after setup in innerregulariz";
}

void InnerRegularizLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers) {
  Tensor<cpu, 2> data(data_.mutable_cpu_data(), Shape2(batchsize_,hdim_));
  CHECK_EQ(srclayers[0]->data(this).count(), batchsize_*vdim_);
  Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape2(batchsize_,vdim_));
  /*float* srcdptr = src.dptr;

  LOG(ERROR) << "regularize vdim_: "<< vdim_;
  LOG(INFO) << "begin print diag_data";
  if (strcmp((this->name()).c_str(), "Diagnosis") == 0){
    for (int i = 0; i < 131; i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<" src: "<<srcdptr[i] << "i: "<<i;
    LOG(INFO)<<"finish 1st round";
    for (int i = vdim_; i < (vdim_+131); i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<" src: "<<srcdptr[i];
    LOG(INFO)<<"finish 2nd round";
    for (int i = 2 * vdim_; i < (2 * vdim_+131); i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<" src: "<<srcdptr[i];
    LOG(INFO)<<"finish 3rd round";
    for (int i = 3 * vdim_; i < (3 * vdim_+131); i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<" src: "<<srcdptr[i];
    LOG(INFO)<<"finish 4th round";
    for (int i = 4 * vdim_; i < (4 * vdim_+131); i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<" src: "<<srcdptr[i];
    LOG(INFO)<<"finish 5th round";
  }*/

  Tensor<cpu, 2> weight(weight_->mutable_cpu_data(), Shape2(vdim_,hdim_));
  Tensor<cpu, 1> bias(bias_->mutable_cpu_data(), Shape1(hdim_));
  data=dot(src, weight);
  // repmat: repeat bias vector into batchsize rows
  data+=repmat(bias, batchsize_);
}

void InnerRegularizLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape2(batchsize_,vdim_));
  Tensor<cpu, 2> grad(grad_.mutable_cpu_data(),Shape2(batchsize_,hdim_));
  Tensor<cpu, 2> weight(weight_->mutable_cpu_data(), Shape2(vdim_,hdim_));
  Tensor<cpu, 2> gweight(weight_->mutable_cpu_grad(), Shape2(vdim_,hdim_));
  Tensor<cpu, 2> similarity_matrix(similarity_matrix_.mutable_cpu_data(),Shape2(regdim_,regdim_));
  Tensor<cpu, 1> gbias(bias_->mutable_cpu_grad(), Shape1(hdim_));
  // LOG(ERROR)<< "regdim_: "<< regdim_;

  gbias=sum_rows(grad);
  gweight = dot(similarity_matrix, weight);
  // print weight norm and regularizaed norm
  LOG(INFO)<<"weight norm: "<<weight_->mutable_data()->asum_data();
  Tensor<cpu, 2> regularized_middle(Shape2(hdim_, regdim_));
  Tensor<cpu, 2> regularized_final(Shape2(hdim_, hdim_));
  AllocSpace(regularized_middle);
  AllocSpace(regularized_final);
  regularized_middle = dot(weight.T(), similarity_matrix);
  regularized_final = dot(regularized_middle, weight);
  float trace_value = 0;
  for (int i = 0; i < hdim_; i++)
    trace_value += regularized_final[i][i];
  LOG(INFO)<<"trace before coefficient: "<<trace_value;
  regularized_final *= regcoefficient_/(1.0f);
  trace_value = 0;
  for (int i = 0; i < hdim_; i++)
    trace_value += regularized_final[i][i];
  LOG(INFO)<<"trace after coefficient: "<<trace_value;
  FreeSpace(regularized_middle);
  FreeSpace(regularized_final);
  // LOG(ERROR)<<"simmatrix norm: "<<similarity_matrix_.asum_data();
  // LOG(ERROR)<<"coefficient: "<<regcoefficient_/(1.0f);
  // LOG(ERROR)<<"gweight norm before coefficient: "<<weight_->mutable_grad()->asum_data();

  gweight *= regcoefficient_/(1.0f);
  // LOG(ERROR)<<"gweight norm after coefficient: "<<weight_->mutable_grad()->asum_data();
  gweight += dot(src.T(), grad);
  // will affect backpropagation ? minus or add this regularization?
  if(srclayers[0]->mutable_grad(this)!=nullptr){
    Tensor<cpu, 2> gsrc(srclayers[0]->mutable_grad(this)->mutable_cpu_data(),
        Shape2(batchsize_,vdim_));
    gsrc=dot(grad, weight.T());
  }
}

/**************** Implementation for InnerProductLayer********************/
void InnerProductLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  LOG(ERROR)<<"inner setup begin ";
  CHECK_EQ(srclayers.size(),1);
  LOG(ERROR)<<"layer name from inner product layer"<<(this->name());
  const auto& src=srclayers[0]->data(this);
  LOG(ERROR)<<"fetch data ends ";
  batchsize_=src.shape()[0];
  LOG(ERROR)<<"batchsize "<<batchsize_;
  vdim_=src.count()/batchsize_;
  LOG(ERROR)<<"vdim_ "<<vdim_;
  hdim_=proto.innerproduct_conf().num_output();
  LOG(ERROR)<<"hdim "<<hdim_;
  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  LOG(ERROR)<<"data_ and grad_ ";
  Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
  weight_=shared_ptr<Param>(factory->Create("Param"));
  bias_=shared_ptr<Param>(factory->Create("Param"));
  weight_->Setup(proto.param(0), vector<int>{vdim_, hdim_});
  bias_->Setup(proto.param(1), vector<int>{hdim_});
  srand((unsigned)time(NULL));
  weight_file_version_ = rand()%1000;
  activation_file_version_ = rand()%1000;
  LOG(ERROR)<<"inner setup end ";
}
void InnerProductLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  LayerProto newproto(proto);
  InnerProductProto * innerproto=newproto.mutable_innerproduct_conf();
  innerproto->set_num_output(shape[1]);
  Setup(newproto, srclayers);
}

void InnerProductLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers) {
  Tensor<cpu, 2> data(data_.mutable_cpu_data(), Shape2(batchsize_,hdim_));
  CHECK_EQ(srclayers[0]->data(this).count(), batchsize_*vdim_);
  //LOG(ERROR)<<" layer name: "<< (this->name());
  Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape2(batchsize_,vdim_));
  float* srcdptr = src.dptr;
  /*if (strcmp((this->name()).c_str(), "Demographics") == 0 || strcmp((this->name()).c_str(), "Procedure") == 0){
    for (int i = 0; i < 10; i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<" src: "<<srcdptr[i];
  }*/
  /*if (strcmp((this->name()).c_str(), "Diagnosis") == 0){
    for (int i = 0; i < 50; i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<" src: "<<srcdptr[i];
    LOG(INFO)<<"finish 1st round";
    for (int i = 1018; i < (1018+50); i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<" src: "<<srcdptr[i];
    LOG(INFO)<<"finish 2nd round";
    for (int i = 2036; i < (2036+50); i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<" src: "<<srcdptr[i];
    LOG(INFO)<<"finish 3rd round";
    for (int i = 3054; i < (3054+50); i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<" src: "<<srcdptr[i];
    LOG(INFO)<<"finish 4th round";
    for (int i = 4072; i < (4072+50); i++)
      LOG(INFO)<<" layer name: "<<(this->name())<<" src: "<<srcdptr[i];
    LOG(INFO)<<"finish 5th round";
  }*/
  // LOG(INFO)<<" layer name: "<<(this->name())<<" src finish";
  /*if (strcmp((this->name()).c_str(), "AllSources") == 0){
    for (int i = 0; i < 13; i++)
      LOG(ERROR)<<" layer name: "<<(this->name())<<" first round data: "<<srcdptr[i];
    for (int j = 338; j < (338 + 13); j++)
      LOG(ERROR)<<" layer name: "<<(this->name())<<" second round data: "<<srcdptr[j];
    for (int j = 676; j < (676 + 13); j++)
      LOG(ERROR)<<" layer name: "<<(this->name())<<" third round data: "<<srcdptr[j];
    for (int j = 1014; j < (1014 + 13); j++)
      LOG(ERROR)<<" layer name: "<<(this->name())<<" third round data: "<<srcdptr[j];

  }*/

  Tensor<cpu, 2> weight(weight_->mutable_cpu_data(), Shape2(vdim_,hdim_));
  Tensor<cpu, 1> bias(bias_->mutable_cpu_data(), Shape1(hdim_));
  data=dot(src, weight);
  // repmat: repeat bias vector into batchsize rows
  data+=repmat(bias, batchsize_);


  // print activation and weight
  if (strcmp((this->name()).c_str(), "Diagfc2") == 0 && phase == kTest){
    ofstream weightout("/data/zhaojing/SynPUF-regularization/visualization/weight" + std::to_string(static_cast<int>(weight_file_version_)) + ".txt");
    ofstream activationout("/data/zhaojing/SynPUF-regularization/visualization/activation" + std::to_string(static_cast<int>(activation_file_version_)) + ".txt");
    weightout << vdim_ << "," << hdim_ << "\n";
    for (int i = 0; i < (vdim_-1); i++)
      for (int j = 0; j < hdim_; j++)
        weightout << weight[i][j] << ",";
    weightout << weight[vdim_-1][hdim_-1] << "\n";

    activationout << batchsize_ << "," << vdim_ << "\n";
    for (int i = 0; i < batchsize_; i++){
      for (int j = 0; j < (vdim_ - 1); j++)
        activationout << src[i][j] << ",";
      activationout << src[i][vdim_-1] << "\n";
    }
  }
}

void InnerProductLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape2(batchsize_,vdim_));
  Tensor<cpu, 2> grad(grad_.mutable_cpu_data(),Shape2(batchsize_,hdim_));
  Tensor<cpu, 2> weight(weight_->mutable_cpu_data(), Shape2(vdim_,hdim_));
  Tensor<cpu, 2> gweight(weight_->mutable_cpu_grad(), Shape2(vdim_,hdim_));
  Tensor<cpu, 1> gbias(bias_->mutable_cpu_grad(), Shape1(hdim_));

  gbias=sum_rows(grad);
  gweight=dot(src.T(), grad);
  if(srclayers[0]->mutable_grad(this)!=nullptr){
    Tensor<cpu, 2> gsrc(srclayers[0]->mutable_grad(this)->mutable_cpu_data(),
        Shape2(batchsize_,vdim_));
    gsrc=dot(grad, weight.T());
  }
}
/**************** Implementation for MultiSrcInnerProductLayer********************/
void MultiSrcInnerProductLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  srclayer_num_ = srclayers.size();
  batchsize_=(srclayers[0]->data(this)).shape()[0];
  LOG(ERROR)<<"(srclayers[0]->data(this)).shape()[0] batchsize_"<<batchsize_;
  vdim_ = 0;
  int srcvdim;
  LOG(ERROR)<<"layer name from multi src inner product layer"<<(this->name());
  for (int i = 0; i < srclayer_num_; i++){
    srcvdim = ((srclayers[i]->data(this)).count())/batchsize_;
    vdim_ += srcvdim;
    LOG(ERROR)<<"layer num: "<<i<<" srcvdim: "<<srcvdim;
  //assume batchsize the same for different srclayers
  }
  LOG(ERROR)<<"vdim all: "<<vdim_;
  hdim_=proto.multisrcinnerproduct_conf().num_output();
  LOG(ERROR)<<"hdim: "<<hdim_;
  data_.Reshape(vector<int>{batchsize_, hdim_});
  LOG(ERROR)<<"data_";
  grad_.ReshapeLike(data_);
  LOG(ERROR)<<"grad_";
  LOG(ERROR)<<"Param begins";
  Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
  for (int i = 0; i < srclayer_num_; i++){
    srcvdim = (((srclayers[i]->data(this)).count())/batchsize_);
    LOG(ERROR)<<"num of vdim_"<<srcvdim;
    weight_.push_back(shared_ptr<Param>(factory->Create("Param")));
    weight_.at(i)->Setup(proto.param(i), vector<int>{srcvdim, hdim_});
  }
  LOG(ERROR)<<"Weight ends";
  bias_=shared_ptr<Param>(factory->Create("Param"));
  bias_->Setup(proto.param(srclayer_num_), vector<int>{hdim_});
  LOG(ERROR)<<"MultiSrcInnerProduct setup done ";
}
void MultiSrcInnerProductLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  LOG(ERROR)<<"No partition after setup";
}

void MultiSrcInnerProductLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers) {
  Tensor<cpu, 2> data(data_.mutable_cpu_data(), Shape2(batchsize_,hdim_));
  //CHECK_EQ(srclayers[0]->data(this).count(), batchsize_*vdim_);
  int srcvdim = (((srclayers[0]->data(this)).count())/batchsize_);
  Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape2(batchsize_,srcvdim));
  Tensor<cpu, 2> weight(weight_.at(0)->mutable_cpu_data(), Shape2(srcvdim, hdim_));
  data=dot(src, weight);
  for (int i = 1; i < srclayer_num_; i++){
    srcvdim = (((srclayers[i]->data(this)).count())/batchsize_);
    Tensor<cpu, 2> src1(srclayers[i]->mutable_data(this)->mutable_cpu_data(),
      Shape2(batchsize_,srcvdim));
    Tensor<cpu, 2> weight1(weight_.at(i)->mutable_cpu_data(), Shape2(srcvdim, hdim_));
    data += dot(src1, weight1);
  }
  // repmat: repeat bias vector into batchsize rows
  Tensor<cpu, 1> bias(bias_->mutable_cpu_data(), Shape1(hdim_));
  data += repmat(bias, batchsize_);
}

void MultiSrcInnerProductLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Tensor<cpu, 2> grad(grad_.mutable_cpu_data(),Shape2(batchsize_,hdim_));
  Tensor<cpu, 1> gbias(bias_->mutable_cpu_grad(), Shape1(hdim_));
  gbias=sum_rows(grad);
  for (int i = 0; i < srclayer_num_; i++){
    int srcvdim = ((srclayers[i]->data(this)).count())/batchsize_;
    Tensor<cpu, 2> src(srclayers[i]->mutable_data(this)->mutable_cpu_data(),
        Shape2(batchsize_,srcvdim));
    Tensor<cpu, 2> weight(weight_.at(i)->mutable_cpu_data(), Shape2(srcvdim, hdim_));
    Tensor<cpu, 2> gweight(weight_.at(i)->mutable_cpu_grad(), Shape2(srcvdim, hdim_));
    gweight=dot(src.T(), grad);
    if(srclayers[i]->mutable_grad(this)!=nullptr){
      Tensor<cpu, 2> gsrc(srclayers[i]->mutable_grad(this)->mutable_cpu_data(),
          Shape2(batchsize_, srcvdim));
      gsrc=dot(grad, weight.T());
    }
  }
}
/*****************************************************************************
 * Implementation for LabelLayer
 *****************************************************************************/
void LabelLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers[0].get())->batchsize();
  data_.Reshape(vector<int>{batchsize});
}

void LabelLayer::ParseRecords(Phase phase, const vector<Record>& records,
    Blob<float>* blob){
  int rid=0;
  float *label= blob->mutable_cpu_data() ;
  for(const Record& record: records){
    label[rid++]=record.image().label();
    CHECK_LT(record.image().label(),10);
  }
  CHECK_EQ(rid, blob->shape()[0]);
}

/*****************************************************************************
 * Implementation for FLabelLayer
 *****************************************************************************/
void FLabelLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers[0].get())->batchsize();
  data_.Reshape(vector<int>{batchsize});
}

void FLabelLayer::ParseRecords(Phase phase, const vector<Record>& records,
    Blob<float>* blob){
  int rid=0;
  float *label= blob->mutable_cpu_data() ;
  for(const Record& record: records){
    label[rid++]=record.vector().label();
    CHECK_LT(record.image().label(),10);
  }
  CHECK_EQ(rid, blob->shape()[0]);
}

/*********************LMDBDataLayer**********************************/
void LMDBDataLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers){
  if(random_skip_){
    int nskip=rand()%random_skip_;
    int n=0;
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
          &mdb_value_, MDB_FIRST), MDB_SUCCESS);
    while (mdb_cursor_get(mdb_cursor_, &mdb_key_,
          &mdb_value_, MDB_NEXT) == MDB_SUCCESS)
      n++;
    LOG(INFO)<<"Random Skip "<<nskip<<" records of total "<<n<<"records";
    // We have reached the end. Restart from the first.
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
          &mdb_value_, MDB_FIRST), MDB_SUCCESS);
    for(int i=0;i<nskip;i++){
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
            &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
    }
    random_skip_=0;
  }
  Datum datum;
  for(auto& record: records_){
    SingleLabelImageRecord* image=record.mutable_image();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
          &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    ConvertDatumToSingleLableImageRecord(datum, image);
    if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
          &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
            &mdb_value_, MDB_FIRST), MDB_SUCCESS);
    }
  }
}

void LMDBDataLayer::ConvertDatumToSingleLableImageRecord(const Datum& datum,
    SingleLabelImageRecord* record){
  record->set_label(datum.label());
  record->clear_shape();
  if(datum.has_channels())
    record->add_shape(datum.channels());
  if(datum.has_height())
    record->add_shape(datum.height());
  if(datum.has_width())
    record->add_shape(datum.width());
  if(datum.has_data())
    record->set_pixel(datum.data());
  if(datum.float_data_size()){
    record->clear_data();
    for(float x: datum.float_data())
      record->add_data(x);
  }
}

void LMDBDataLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS); // 1TB
  CHECK_EQ(mdb_env_open(mdb_env_,
        proto.lmdbdata_conf().path().c_str(),
        MDB_RDONLY, 0664), MDB_SUCCESS) << "cannot open lmdb "
    << proto.lmdbdata_conf().path();
  CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
    << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
    << "mdb_open failed";
  CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
    << "mdb_cursor_open failed";
  LOG(INFO) << "Opening lmdb " << proto.lmdbdata_conf().path();
  CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
      MDB_SUCCESS) << "mdb_cursor_get failed";

  if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
      != MDB_SUCCESS) {
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
          MDB_FIRST), MDB_SUCCESS);
  }
  Datum datum;
  datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
  SingleLabelImageRecord* record=sample_.mutable_image();
  ConvertDatumToSingleLableImageRecord(datum, record);

  batchsize_=batchsize();
  records_.resize(batchsize_);
  random_skip_=proto.lmdbdata_conf().random_skip();
}

/***************** Implementation for LRNLayer *************************/
void LRNLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  lsize_ = proto.lrn_conf().local_size();
  CHECK_EQ(lsize_ % 2, 1) << "LRN only supports odd values for Localvol";
  knorm_=proto.lrn_conf().knorm();
  alpha_ = proto.lrn_conf().alpha();
  beta_ = proto.lrn_conf().beta();

  const vector<int>& s=srclayers[0]->data(this).shape();
  data_.Reshape(s);
  grad_.Reshape(s);
  norm_.Reshape(s);
  batchsize_=s[0];
  channels_=s[1];
  height_=s[2];
  width_=s[3];
}

void LRNLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}

void LRNLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers){
  const float salpha = alpha_ / lsize_;
  Shape<4> s=Shape4(batchsize_,channels_, height_, width_);
  Tensor<cpu, 4> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(), s);
  Tensor<cpu, 4> data(data_.mutable_cpu_data(), s);
  Tensor<cpu, 4> norm(norm_.mutable_cpu_data(), s);
  // stores normalizer without power
  norm= chpool<red::sum>( F<op::square>(src) , lsize_ ) * salpha + knorm_;
  data = src * F<op::power>(norm, -beta_ );
}

void LRNLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  const float salpha = alpha_ / lsize_;
  Shape<4> s=Shape4(batchsize_,channels_, height_, width_);
  Tensor<cpu, 4> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(), s);
  Tensor<cpu, 4> norm(norm_.mutable_cpu_data(), s);
  Tensor<cpu, 4> grad(grad_.mutable_cpu_data(), s);
  Tensor<cpu, 4> gsrc(srclayers[0]->mutable_grad(this)->mutable_cpu_data(), s);

  gsrc = grad * F<op::power>( norm, -beta_ );
  gsrc += ( - 2.0f * beta_ * salpha ) * chpool<red::sum>(
      grad * src * F<op::power>( norm, -beta_-1.0f ), lsize_ )  * src;
}

/**************** Implementation for MnistImageLayer******************/

void MnistLayer::ParseRecords(Phase phase,
    const vector<Record>& records, Blob<float>* blob){
  LOG_IF(ERROR, records.size()==0)<<"Empty records to parse";
  int ndim=records.at(0).image().shape_size();
  int inputsize =records.at(0).image().shape(ndim-1);

  float* dptr=blob->mutable_cpu_data();
  for(const Record& record: records){
    // copy from record to cv::Mat
    cv::Mat input(inputsize, inputsize, CV_32FC1);
    const SingleLabelImageRecord& imagerecord=record.image();
    if(imagerecord.pixel().size()){
      string pixel=imagerecord.pixel();
      for(int i=0,k=0;i<inputsize;i++)
        for(int j=0;j<inputsize;j++)
          // NOTE!!! must cast pixel to uint8_t then to float!!! waste a lot of
          // time to debug this
          input.at<float>(i,j)=static_cast<float>(static_cast<uint8_t>(pixel[k++]));
    }else{
      for(int i=0,k=0;i<inputsize;i++)
        for(int j=0;j<inputsize;j++)
          input.at<float>(i,j)=imagerecord.data(k++);
    }
    int size=blob->shape()[1];
    /*
    cv::Mat resizeMat=input;
    // affine transform, scaling, rotation and shearing
    if(gamma_){
      float r1=rand_real()*2-1;
      float r2=rand_real()*2-1;
      int h=static_cast<int>(inputsize*(1.+r1*gamma_/100.0));
      int w=static_cast<int>(inputsize*(1.+r2*gamma_/100.0));
      cv::resize(input, resizeMat, cv::Size(h,w));
    }
    cv::Mat betaMat=resizeMat;
    cv::Mat warpmat(2,3, CV_32FC1);
    warpmat.at<float>(0,0)=1.0;
    warpmat.at<float>(0,1)=0.0;
    warpmat.at<float>(0,2)=0.0;
    warpmat.at<float>(1,0)=0.0;
    warpmat.at<float>(1,1)=1.0;
    warpmat.at<float>(1,2)=0.0;

    if(beta_){
      float r=rand_real()*2-1;
      if(rand() % 2){ // rotation
        cv::Point center(resizeMat.rows/2, resizeMat.cols/2);
        warpmat=cv::getRotationMatrix2D(center, r*beta_, 1.0);
      }else{
        //shearing
        warpmat.at<float>(0,1)=r*beta_/90;
        if(imagerecord.label()==1 ||imagerecord.label()==7)
          warpmat.at<float>(0,1)/=2.0;
      }
    }
    cv::warpAffine(resizeMat, betaMat, warpmat, cv::Size(size, size));
    */

    for(int i=0;i<size;i++){
      for(int j=0;j<size;j++){
        *dptr=input.at<float>(i,j)/norm_a_-norm_b_;
        dptr++;
      }
    }
  }
  CHECK_EQ(dptr, blob->mutable_cpu_data()+blob->count());
}
void MnistLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers[0].get())->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers[0].get())->sample();
  kernel_=proto.mnist_conf().kernel();
  sigma_=proto.mnist_conf().sigma();
  alpha_=proto.mnist_conf().alpha();
  beta_=proto.mnist_conf().beta();
  gamma_=proto.mnist_conf().gamma();
  resize_=proto.mnist_conf().resize();
  norm_a_=proto.mnist_conf().norm_a();
  norm_b_=proto.mnist_conf().norm_b();
  elastic_freq_=proto.mnist_conf().elastic_freq();

  int ndim=sample.image().shape_size();
  CHECK_GE(ndim,2);
  if(resize_)
    data_.Reshape(vector<int>{batchsize, resize_, resize_});
  else{
    int s=sample.image().shape(ndim-1);
    CHECK_EQ(s,sample.image().shape(ndim-2));
    data_.Reshape(vector<int>{batchsize, s, s });
  }
}

/**************** Implementation for MultisrcDataLayer******************/

void MultiSrcDataLayer::ParseRecords(Phase phase,
    const vector<Record>& records, Blob<float>* blob){
  LOG_IF(ERROR, records.size()==0)<<"Empty records to parse";
  //LOG(ERROR)<<"Multisrcdata parse records begins ";
  int ndim=records.at(0).image().shape_size();
  int rows =records.at(0).image().shape(ndim-2);  //rows and columns
  int cols =records.at(0).image().shape(ndim-1);
  //LOG(ERROR)<<"ndim "<<ndim<<"rows "<<rows<<"cols "<<cols;
  float* diagdptr;
  float* labdptr;
  float* raddptr;
  float* meddptr;
  float* procdptr;
  float* demodptr;

  float* dptr=blob->mutable_cpu_data();
  //LOG(ERROR)<<"dptr over";
  if (diag_dim_ != 0){
    diagdptr=diag_data_.mutable_cpu_data();
    //LOG(ERROR)<<"diagdptr over";
  }
  if (lab_dim_ != 0){
    labdptr=lab_data_.mutable_cpu_data();
    LOG(ERROR)<<"labdptr over";
  }
  if (rad_dim_ != 0){
    raddptr=rad_data_.mutable_cpu_data();
    LOG(ERROR)<<"raddptr over";
  }
  if (med_dim_ != 0){
    meddptr=med_data_.mutable_cpu_data();
    //LOG(ERROR)<<"meddptr over";
  }
  if (proc_dim_ != 0){
    procdptr=proc_data_.mutable_cpu_data();
    //LOG(ERROR)<<"procdptr over";
  }
  if (demo_dim_ != 0){
    demodptr=demo_data_.mutable_cpu_data();
    //LOG(ERROR)<<"demodptr over";
  }
  int num_records = 0;
  //LOG(ERROR)<<"Multisrcdata declaration over";
  for(const Record& record: records){
    num_records ++;
    // copy from record to cv::Mat
  //  cv::Mat input(rows, cols, CV_32FC1);      //not perfect rectangular, so rows and columns
    const SingleLabelImageRecord& imagerecord=record.image();
    if(imagerecord.pixel().size()){
      string pixel=imagerecord.pixel();
      for(int i=0,k=0,l=0;i<rows;i++)
        for(int j=0;j<cols;j++){
          // time to debug this
          *dptr=static_cast<float>(static_cast<uint8_t>(pixel[l++]));
	  //LOG(INFO)<<StringPrintf("zj if pixel().size(): data %f\n", *dptr);
	  dptr++;
          // NOTE!!! must cast pixel to uint8_t then to float!!! waste a lot of
          if (j >= 0 && j < diag_dim_){
            *diagdptr=static_cast<float>(static_cast<uint8_t>(pixel[k++]));
	    //LOG(INFO)<<StringPrintf("zj if pixel().size(): data %f\n", *diagdptr);
	    diagdptr++;
          }
          else if (j >= diag_dim_ && j < (diag_dim_ + lab_dim_)){
            *labdptr=static_cast<float>(static_cast<uint8_t>(pixel[k++]));
            //LOG(INFO)<<StringPrintf("zj if pixel().size(): data %f\n", *labdptr);
            labdptr++;
          }
          else if (j >= (diag_dim_ + lab_dim_) && j < (diag_dim_ + lab_dim_ + rad_dim_)){
            *raddptr=static_cast<float>(static_cast<uint8_t>(pixel[k++]));
            //LOG(INFO)<<StringPrintf("zj if pixel().size(): data %f\n", *raddptr);
            raddptr++;
          }
          else if (j >= (diag_dim_ + lab_dim_ + rad_dim_) && j < (diag_dim_ + lab_dim_ + rad_dim_ + med_dim_)){
            *meddptr=static_cast<float>(static_cast<uint8_t>(pixel[k++]));
            //LOG(INFO)<<StringPrintf("zj if pixel().size(): med data %f\n", *meddptr);
            meddptr++;
          }
          else if (j >= (diag_dim_ + lab_dim_ + rad_dim_ + med_dim_) && j < (diag_dim_ + lab_dim_ + rad_dim_ + med_dim_ + proc_dim_)){
            *procdptr=static_cast<float>(static_cast<uint8_t>(pixel[k++]));
            //LOG(INFO)<<StringPrintf("zj if pixel().size(): data %f\n", *procdptr);
            procdptr++;
          }
          else if (j >= (diag_dim_ + lab_dim_ + rad_dim_ + med_dim_ + proc_dim_) && j < (diag_dim_ + lab_dim_ + rad_dim_ + med_dim_ + proc_dim_ + demo_dim_)){
            *demodptr=static_cast<float>(static_cast<uint8_t>(pixel[k++]));
            //LOG(INFO)<<StringPrintf("zj if pixel().size(): data %f\n", *demodptr);
            demodptr++;
          }
        }
	 //LOG(INFO)<<StringPrintf("One Record Done! label: %d\n", static_cast<uint8_t>(record.image().label()));
    }else{
       LOG(ERROR)<<"MultiSrcDataLayer: not fetch image record ";
    }
  //LOG(INFO)<<StringPrintf("num_records: %d\n", num_records);
  }
  //LOG(INFO)<<StringPrintf("sum: num_records: %d\n", num_records);
  //CHECK_EQ(demodptr, blob->mutable_cpu_data()+blob->count());
  //LOG(ERROR)<<"parse records ends ";
  /*float* medtest=med_data_.mutable_cpu_data();
  for (int n = 0; n < num_records; n++)
    for (int j = 0; j < med_dim_; j++){
      LOG(INFO)<<"data after parser: "<<medtest[0];
      medtest++;
    }*/

}
void MultiSrcDataLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers[0].get())->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers[0].get())->sample();

  LOG(ERROR)<<"MultiSrcData batchsize: "<<batchsize;
  int ndim=sample.image().shape_size();
  LOG(ERROR)<<"ndim: "<<ndim;
  CHECK_GE(ndim,2);
  //LOG(ERROR)<<"parse set up";
  //int s=sample.image().shape(ndim-1);
  //CHECK_EQ(s,sample.image().shape(ndim-2));
  int rows=sample.image().shape(ndim-2);
  int cols=sample.image().shape(ndim-1);
  diag_dim_ = proto.multisrcdata_conf().diag_dim();
  lab_dim_ = proto.multisrcdata_conf().lab_dim();
  rad_dim_ = proto.multisrcdata_conf().rad_dim();
  med_dim_ = proto.multisrcdata_conf().med_dim();
  proc_dim_ = proto.multisrcdata_conf().proc_dim();
  demo_dim_ = proto.multisrcdata_conf().demo_dim();
  //LOG(ERROR)<<"rows:"<<rows<<"cols: "<<cols<<"diag_dim"<<diag_dim_<<" lab_dim"<<lab_dim_<<" rad_dim_"<<rad_dim_<<" med_dim_"<<med_dim_<<" proc_dim_"<<proc_dim_<<" demo_dim_"<<demo_dim_;
  //LOG(ERROR)<<"data_ reshape begins";
  data_.Reshape(vector<int>{batchsize, rows, cols});
  //LOG(ERROR)<<"data_ size: "<<data_.shape().size();
  //LOG(ERROR)<<"data_ reshape ends";
  if (diag_dim_ != 0)
    diag_data_.Reshape(vector<int>{batchsize, rows, diag_dim_});
  if (lab_dim_ != 0)
    lab_data_.Reshape(vector<int>{batchsize, rows, lab_dim_});
  if (rad_dim_ != 0)
    rad_data_.Reshape(vector<int>{batchsize, rows, rad_dim_});
  if (med_dim_ != 0)
    med_data_.Reshape(vector<int>{batchsize, rows, med_dim_});
  if (proc_dim_ != 0)
    proc_data_.Reshape(vector<int>{batchsize, rows, proc_dim_});
  if (demo_dim_ != 0)
    demo_data_.Reshape(vector<int>{batchsize, rows, demo_dim_});
  /*LOG(INFO)<<StringPrintf("zj: rows %d cols %d\n", rows,cols);*/
}

/**************** Implementation for MultisrcFDataLayer******************/

void MultiSrcFDataLayer::ParseRecords(Phase phase,
    const vector<Record>& records, Blob<float>* blob){
  LOG_IF(ERROR, records.size()==0)<<"Empty records to parse";
  //LOG(ERROR)<<"Multisrcfdata parse records begins ";
  int ndim=records.at(0).vector().shape_size();
  int rows =records.at(0).vector().shape(ndim-2);  //rows and columns
  int cols =records.at(0).vector().shape(ndim-1);
  //LOG(ERROR)<<"ndim "<<ndim<<"rows "<<rows<<"cols "<<cols;
  float* diagdptr;
  float* labdptr;
  float* raddptr;
  float* meddptr;
  float* procdptr;
  float* demodptr;

  float* dptr=blob->mutable_cpu_data();
  //LOG(ERROR)<<"dptr over";
  if (diag_dim_ != 0){
    diagdptr=diag_data_.mutable_cpu_data();
    //LOG(ERROR)<<"diagdptr over";
  }
  if (lab_dim_ != 0){
    labdptr=lab_data_.mutable_cpu_data();
    LOG(ERROR)<<"labdptr over";
  }
  if (rad_dim_ != 0){
    raddptr=rad_data_.mutable_cpu_data();
    LOG(ERROR)<<"raddptr over";
  }
  if (med_dim_ != 0){
    meddptr=med_data_.mutable_cpu_data();
    //LOG(ERROR)<<"meddptr over";
  }
  if (proc_dim_ != 0){
    procdptr=proc_data_.mutable_cpu_data();
    //LOG(ERROR)<<"procdptr over";
  }
  if (demo_dim_ != 0){
    demodptr=demo_data_.mutable_cpu_data();
    //LOG(ERROR)<<"demodptr over";
  }
  int num_records = 0;
  //LOG(ERROR)<<"Multisrcfdata declaration over";
  for(const Record& record: records){
    num_records ++;
    // copy from record to cv::Mat
  //  cv::Mat input(rows, cols, CV_32FC1);      //not perfect rectangular, so rows and columns
    const SingleLabelVectorRecord& vectorrecord=record.vector();
      for(int i=0,k=0,l=0;i<rows;i++)
        for(int j=0;j<cols;j++){
          // time to debug this
          *dptr=static_cast<float>(vectorrecord.data(l++));
	  //LOG(INFO)<<StringPrintf("zj if pixel().size(): data %f\n", *dptr);
	  dptr++;
          // NOTE!!! must cast pixel to uint8_t then to float!!! waste a lot of
          if (j >= 0 && j < diag_dim_){
            *diagdptr=static_cast<float>(vectorrecord.data(k++));
	          // LOG(INFO)<<StringPrintf("zj float: num_records: %d  data %f\n", num_records, *diagdptr);
	          diagdptr++;
          }
          else if (j >= diag_dim_ && j < (diag_dim_ + lab_dim_)){
            *labdptr=static_cast<float>(vectorrecord.data(k++));
            //LOG(INFO)<<StringPrintf("zj if pixel().size(): data %f\n", *labdptr);
            labdptr++;
          }
          else if (j >= (diag_dim_ + lab_dim_) && j < (diag_dim_ + lab_dim_ + rad_dim_)){
            *raddptr=static_cast<float>(vectorrecord.data(k++));
            //LOG(INFO)<<StringPrintf("zj if pixel().size(): data %f\n", *raddptr);
            raddptr++;
          }
          else if (j >= (diag_dim_ + lab_dim_ + rad_dim_) && j < (diag_dim_ + lab_dim_ + rad_dim_ + med_dim_)){
            *meddptr=static_cast<float>(vectorrecord.data(k++));
            //LOG(INFO)<<StringPrintf("zj if pixel().size(): med data %f\n", *meddptr);
            meddptr++;
          }
          else if (j >= (diag_dim_ + lab_dim_ + rad_dim_ + med_dim_) && j < (diag_dim_ + lab_dim_ + rad_dim_ + med_dim_ + proc_dim_)){
            *procdptr=static_cast<float>(vectorrecord.data(k++));
            //LOG(INFO)<<StringPrintf("zj if pixel().size(): data %f\n", *procdptr);
            procdptr++;
          }
          else if (j >= (diag_dim_ + lab_dim_ + rad_dim_ + med_dim_ + proc_dim_) && j < (diag_dim_ + lab_dim_ + rad_dim_ + med_dim_ + proc_dim_ + demo_dim_)){
            *demodptr=static_cast<float>(vectorrecord.data(k++));
            //LOG(INFO)<<StringPrintf("zj if pixel().size(): data %f\n", *demodptr);
            demodptr++;
          }
        }
	 //LOG(INFO)<<StringPrintf("One Record Done! label: %d\n", static_cast<uint8_t>(record.image().label()));
  //LOG(INFO)<<StringPrintf("num_records: %d\n", num_records);
  }
  //LOG(INFO)<<StringPrintf("sum: num_records: %d\n", num_records);
  //CHECK_EQ(demodptr, blob->mutable_cpu_data()+blob->count());
  //LOG(ERROR)<<"MultiSrcFloat parse records ends ";
  /*float* medtest=med_data_.mutable_cpu_data();
  for (int n = 0; n < num_records; n++)
    for (int j = 0; j < med_dim_; j++){
      LOG(INFO)<<"data after parser: "<<medtest[0];
      medtest++;
    }*/

}
void MultiSrcFDataLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers[0].get())->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers[0].get())->sample();

  LOG(ERROR)<<"MultiSrcFData batchsize: "<<batchsize;
  int ndim=sample.vector().shape_size();
  LOG(ERROR)<<"ndim: "<<ndim;
  CHECK_GE(ndim,2);
  LOG(ERROR)<<"parse set up";
  //int s=sample.image().shape(ndim-1);
  //CHECK_EQ(s,sample.image().shape(ndim-2));
  int rows=sample.vector().shape(ndim-2);
  int cols=sample.vector().shape(ndim-1);
  LOG(ERROR)<<"rows: "<<rows<<" columns: "<<cols;
  diag_dim_ = proto.multisrcfdata_conf().diag_dim();
  LOG(ERROR)<<"diag_dim: "<<diag_dim_;
  lab_dim_ = proto.multisrcfdata_conf().lab_dim();
  rad_dim_ = proto.multisrcfdata_conf().rad_dim();
  med_dim_ = proto.multisrcfdata_conf().med_dim();
  proc_dim_ = proto.multisrcfdata_conf().proc_dim();
  demo_dim_ = proto.multisrcfdata_conf().demo_dim();
  //LOG(ERROR)<<"rows:"<<rows<<"cols: "<<cols<<"diag_dim"<<diag_dim_<<" lab_dim"<<lab_dim_<<" rad_dim_"<<rad_dim_<<" med_dim_"<<med_dim_<<" proc_dim_"<<proc_dim_<<" demo_dim_"<<demo_dim_;
  //LOG(ERROR)<<"data_ reshape begins";
  data_.Reshape(vector<int>{batchsize, rows, cols});
  //LOG(ERROR)<<"data_ size: "<<data_.shape().size();
  //LOG(ERROR)<<"data_ reshape ends";
  if (diag_dim_ != 0){
    diag_data_.Reshape(vector<int>{batchsize, rows, diag_dim_});
    LOG(ERROR)<<"diag_data: "<<"batchsize: "<<batchsize;
  }
  if (lab_dim_ != 0)
    lab_data_.Reshape(vector<int>{batchsize, rows, lab_dim_});
  if (rad_dim_ != 0)
    rad_data_.Reshape(vector<int>{batchsize, rows, rad_dim_});
  if (med_dim_ != 0)
    med_data_.Reshape(vector<int>{batchsize, rows, med_dim_});
  if (proc_dim_ != 0)
    proc_data_.Reshape(vector<int>{batchsize, rows, proc_dim_});
  if (demo_dim_ != 0)
    demo_data_.Reshape(vector<int>{batchsize, rows, demo_dim_});
  /*LOG(INFO)<<StringPrintf("zj: rows %d cols %d\n", rows,cols);*/
  LOG(ERROR)<<"MultiSrcFData Setup ends";
}


/******************** Implementation for PoolingLayer******************/
void PoolingLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  PoolingProto pool_conf = proto.pooling_conf();
  kernel_=pool_conf.kernel();
  stride_=pool_conf.stride();
  CHECK_LT(pad_, kernel_);
  pool_=proto.pooling_conf().pool();
  CHECK(pool_ == PoolingProto_PoolMethod_AVE
        || pool_ == PoolingProto_PoolMethod_MAX)
      << "Padding implemented only for average and max pooling.";

  const auto& srcshape=srclayers[0]->data(this).shape();
  int dim=srcshape.size();
  CHECK_GT(dim,2);
  width_ = srcshape[dim-1];
  height_ = srcshape[dim-2];
  if(dim>3)
    channels_ = srcshape[dim-3];
  else
    channels_=1;
  batchsize_=srcshape[0];
  pooled_height_ = static_cast<int>((height_ - kernel_) / stride_) + 1;
  pooled_width_ = static_cast<int>(( width_ - kernel_) / stride_) + 1;
  data_.Reshape(vector<int>{batchsize_, channels_, pooled_height_, pooled_width_});
  grad_.ReshapeLike(data_);
}

void PoolingLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}

void PoolingLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers){
  Tensor<cpu, 4> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape4(batchsize_, channels_, height_, width_));
  Tensor<cpu, 4> data(data_.mutable_cpu_data(),
      Shape4(batchsize_, channels_, pooled_height_, pooled_width_));
  if(pool_ == PoolingProto_PoolMethod_MAX)
    data=pool<red::maximum>(src, kernel_, stride_);
  else if(pool_ == PoolingProto_PoolMethod_AVE)
    data=pool<red::sum>(src, kernel_, stride_)
      *(1.0f/(kernel_*kernel_));
}

/*
 * partition only on num/channel dim
 * assume grad and data have the same paritition
 */
void PoolingLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Shape<4> s1= Shape4(batchsize_, channels_, height_, width_);
  Tensor<cpu, 4> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),s1);
  Tensor<cpu, 4> gsrc(srclayers[0]->mutable_grad(this)->mutable_cpu_data(),s1);
  Shape<4> s2= Shape4(batchsize_, channels_, pooled_height_, pooled_width_);
  Tensor<cpu, 4> data(data_.mutable_cpu_data(), s2);
  Tensor<cpu, 4> grad(grad_.mutable_cpu_data(), s2);
  if(pool_ == PoolingProto_PoolMethod_MAX)
      gsrc = unpool<red::maximum>(src, data, grad, kernel_, stride_);
  else if(pool_ == PoolingProto_PoolMethod_AVE)
      gsrc = unpool<red::sum>(src, data, grad, kernel_, stride_)
        *(1.0f/(kernel_*kernel_));
}

/***************** Implementation for ReLULayer *****************************/

void ReLULayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  data_.ReshapeLike(srclayers[0]->data(this));
  grad_.ReshapeLike(*(srclayers[0]->mutable_grad(this)));
}

void ReLULayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}

void ReLULayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers){
  Tensor<cpu, 1> data(data_.mutable_cpu_data(), Shape1(data_.count()));
  Tensor<cpu, 1> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape1(data_.count()));
  data=F<op::relu>(src);
}

void ReLULayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Tensor<cpu, 1> grad(grad_.mutable_cpu_data(), Shape1(grad_.count()));
  Tensor<cpu, 1> data(data_.mutable_cpu_data(), Shape1(data_.count()));
  Tensor<cpu, 1> gsrc(srclayers[0]->mutable_grad(this)->mutable_cpu_data(),
      Shape1(data_.count()));
  gsrc=F<op::relu_grad>(data)*grad;
}

/*************** Implementation for RGBImageLayer *************************/

void RGBImageLayer::ParseRecords(Phase phase,
    const vector<Record>& records, Blob<float>* blob){
  const vector<int>& s=blob->shape();
  Tensor<cpu, 4> images(data_.mutable_cpu_data(), Shape4(s[0],s[1],s[2],s[3]));
  const SingleLabelImageRecord& r=records.at(0).image();
  Tensor<cpu, 3> raw_image(Shape3(r.shape(0),r.shape(1),r.shape(2)));
  AllocSpace(raw_image);
  Tensor<cpu, 3> croped_image(nullptr, Shape3(s[1],s[2],s[3]));
  if(cropsize_)
    AllocSpace(croped_image);
    //CHECK(std::equal(croped_image.shape(), raw_image.shape());
  int rid=0;
  const float* meandptr=mean_.cpu_data();
  for(const Record& record: records){
    auto image=images[rid];
    bool do_crop=cropsize_>0&&(phase == kTrain);
    bool do_mirror=mirror_&&rand()%2&&(phase == kTrain);
    float* dptr=nullptr;
    if(do_crop||do_mirror)
      dptr=raw_image.dptr;
    else
      dptr=image.dptr;
    if(record.image().pixel().size()){
      string pixel=record.image().pixel();
      for(size_t i=0;i<pixel.size();i++)
        dptr[i]=static_cast<float>(static_cast<uint8_t>(pixel[i]));
    }else {
      memcpy(dptr, record.image().data().data(),
          sizeof(float)*record.image().data_size());
    }
    for(int i=0;i<mean_.count();i++)
      dptr[i]-=meandptr[i];

    if(do_crop){
      int hoff=rand()%(r.shape(1)-cropsize_);
      int woff=rand()%(r.shape(2)-cropsize_);
      Shape<2> cropshape=Shape2(cropsize_, cropsize_);
      if(do_mirror){
        croped_image=crop(raw_image, cropshape, hoff, woff);
        image=mirror(croped_image);
      }else{
        image=crop(raw_image, cropshape, hoff, woff);
      }
    }else if(do_mirror){
      image=mirror(raw_image);
    }
    rid++;
  }
  if(scale_)
    images=images*scale_;

  FreeSpace(raw_image);
  if(cropsize_)
    FreeSpace(croped_image);
}
void RGBImageLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  scale_=proto.rgbimage_conf().scale();
  cropsize_=proto.rgbimage_conf().cropsize();
  mirror_=proto.rgbimage_conf().mirror();
  int batchsize=static_cast<DataLayer*>(srclayers[0].get())->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers[0].get())->sample();
  vector<int> shape;
  shape.push_back(batchsize);
  for(int x: sample.image().shape()){
    shape.push_back(x);
  }
  CHECK_EQ(shape.size(),4);
  if(cropsize_){
    shape[2]=cropsize_;
    shape[3]=cropsize_;
  }
  data_.Reshape(shape);
  mean_.Reshape({shape[1],shape[2],shape[3]});
  if(proto.rgbimage_conf().has_meanfile()){
    if(proto.rgbimage_conf().meanfile().find("binaryproto")!=string::npos){
      BlobProto tmp;
      ReadProtoFromBinaryFile(proto.rgbimage_conf().meanfile().c_str(), &tmp);
      CHECK_EQ(mean_.count(), tmp.data_size());
      memcpy(mean_.mutable_cpu_data(), tmp.data().data(), sizeof(float)*tmp.data_size());
    }else{
      SingleLabelImageRecord tmp;
      ReadProtoFromBinaryFile(proto.rgbimage_conf().meanfile().c_str(), &tmp);
      CHECK_EQ(mean_.count(), tmp.data_size());
      memcpy(mean_.mutable_cpu_data(), tmp.data().data(), sizeof(float)*tmp.data_size());
    }
  }else{
    memset(mean_.mutable_cpu_data(),0,sizeof(float)*mean_.count());
  }
}

/***************Implementation for ShardDataLayer**************************/
void ShardDataLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers){
  //LOG(ERROR)<<"shard data begins ";
  if(random_skip_){
    int nskip=rand()%random_skip_;
    LOG(INFO)<<"Random Skip "<<nskip<<" records, there are "<<shard_->Count()
      <<" records in total";
    string key;
    for(int i=0;i<nskip;i++){
      shard_->Next(&key, &sample_);
    }
    random_skip_=0;
  }
  for(auto& record: records_){
    string key;
    if(!shard_->Next(&key, &record)){
      shard_->SeekToFirst();
      CHECK(shard_->Next(&key, &record));
    }
  }
  //LOG(ERROR)<<"shard_records num"<<records_.size();
}

void ShardDataLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  shard_= std::make_shared<DataShard>(proto.sharddata_conf().path(),
      DataShard::kRead);
  LOG(ERROR)<<"data path "<<proto.sharddata_conf().path();
  string key;
  shard_->Next(&key, &sample_);
  batchsize_=proto.sharddata_conf().batchsize();

  records_.resize(batchsize_);
  random_skip_=proto.sharddata_conf().random_skip();
  //LOG(ERROR)<<"shard data set up ends ";
}
/*******************Implementation of TanLayer***************************/
void TanhLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  data_.ReshapeLike(srclayers[0]->data(this));
  grad_.ReshapeLike(srclayers[0]->grad(this));
}

void TanhLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}


void TanhLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers){
  Tensor<cpu, 1> data(data_.mutable_cpu_data(), Shape1(data_.count()));
  Tensor<cpu, 1> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape1(data_.count()));
  data=F<op::stanh>(src);
}

void TanhLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Tensor<cpu, 1> data(data_.mutable_cpu_data(), Shape1(data_.count()));
  Tensor<cpu, 1> grad(grad_.mutable_cpu_data(), Shape1(grad_.count()));
  Tensor<cpu, 1> gsrc(srclayers[0]->mutable_grad(this)->mutable_cpu_data(),
      Shape1(data_.count()));
  gsrc=F<op::stanh_grad>(data)*grad;
}
/********** * Implementation for SoftmaxProbLayer*************************/
void SoftmaxProbLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1); //no label
  //LOG(ERROR)<<"layer name: "<<(this->name())<<"softmax Prob setup begins";
  data_.Reshape(srclayers[0]->data(this).shape());
  batchsize_=data_.shape()[0];
  //LOG(ERROR)<<"batchsize_ "<<batchsize_;
  grad_.Reshape(vector<int>{batchsize_, 1}); /*grad is 1, only the y1*/
  dim_=data_.count()/batchsize_; /*dim_ is 2, becuase prob remains 2*/
  //LOG(INFO)<<"layer name: "<<(this->name())<<"dim_: "<<dim_;
}
void SoftmaxProbLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  LOG(ERROR)<<"No partition after set up";
}
void SoftmaxProbLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers) {
  Shape<2> s=Shape2(batchsize_, dim_);
  LOG(INFO)<<"softmaxprob dimension: "<<dim_;
  Tensor<cpu, 2> prob(data_.mutable_cpu_data(), s);
  Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(), s);
  Softmax(prob, src); //compute probability
  float* probdptr = prob.dptr;
  float* srcdptr = src.dptr;
  for (int i = 0; i < 10; i++){
    LOG(INFO)<<this->name()<<" "<<i<<": prob_1: "<<probdptr[1]<<" prob_0: "<<probdptr[0];
    //LOG(INFO)<<this->name()<<" "<<i<<": src_1: "<<srcdptr[1]<<" src_0: "<<srcdptr[0];
    probdptr+=dim_;
    srcdptr+=dim_;
  }
}

void SoftmaxProbLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Blob<float>* gsrcblob=srclayers[0]->mutable_grad(this);
  float* gsrcptr = gsrcblob->mutable_cpu_data();
  float* dsrcptr = srclayers[0]->mutable_data(this)->mutable_cpu_data();
  float* gradptr = grad_.mutable_cpu_data();
  //innerproduct result of srclayer
  for(int n=0;n<batchsize_;n++){
    float a0 = dsrcptr[n*dim_+0]; //inner_product_0
    float a1 = dsrcptr[n*dim_+1]; //inner_product_1
    float gradval = gradptr[n];
    gsrcptr[n*dim_+0] = -gradval*exp(a0-a1)/((exp(a0-a1)+1) * (exp(a0-a1)+1));
    gsrcptr[n*dim_+1] = gradval*exp(a0-a1)/((exp(a0-a1)+1) * (exp(a0-a1)+1));
    /*if (n < 10)
      LOG(INFO)<<"a0: "<<a0<<" a1: "<<a1<<" gradval: "<<gradval<<" gsrc0: "<<gsrcptr[n*dim_+0]<<" gsrc1: "<<gsrcptr[n*dim_+1];*/
  }
  //remember to scale batchsize in logistic loss layer!!!!!!!!!!!
}
/********** * Implementation for LogisticLossLayer*************************/
void LogisticLossLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  //LOG(ERROR)<<"Logistic layer set up begins ";
  CHECK_EQ(srclayers.size(),2);
  data_.Reshape(srclayers[0]->data(this).shape());
  batchsize_=data_.shape()[0];
  dim_=data_.count()/batchsize_;
  metric_.Reshape(vector<int>{8});
  scale_=proto.logisticloss_conf().scale();
  print_step_ = 0;
  srand((unsigned)time(NULL));
  run_version_ = rand()%1000;
  LOG(ERROR)<<"logistic scale_: "<<scale_;
  //LOG(ERROR)<<"Logistic layer set up ends ";
}
void LogisticLossLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  LOG(ERROR)<<"No partition after set up";
}
void LogisticLossLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers) {
  Shape<2> s=Shape2(batchsize_, dim_);
  // LOG(ERROR)<<"logistic dimension "<<dim_; //dimension should be 1
  Tensor<cpu, 2> prob(data_.mutable_cpu_data(), s);
  Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(), s);
  prob=F<op::sigmoid>(src);
  const float* label=srclayers[1]->data(this).cpu_data();
  const float* probptr=prob.dptr;
  const float* srcdptr=src.dptr;
  float loss=0, precision=0;
  float predict_0 = 0, true_0 = 0, correct_0 = 0;
  float predict_1 = 0, true_1 = 0, correct_1 = 0;

  for(int n=0;n<batchsize_;n++){
    int ilabel=static_cast<int>(label[n]);
    CHECK_LT(ilabel,10);
    CHECK_GE(ilabel,0);
    loss += -ilabel*log(std::max(probptr[0], FLT_MIN))-(1-ilabel)*log(std::max(1-probptr[0], FLT_MIN));//is this correct?
    // LOG(INFO)<< "ilabel*log(probptr[0]): "<<ilabel*log(probptr[0]);
    // LOG(INFO)<< "(1-ilabel)*log(1-probptr[0]): "<<(1-ilabel)*log(1-probptr[0]);
    if (phase == kTest)
      LOG(INFO)<<"ilabel "<<ilabel<<" predict prob-1 "<<probptr[0]<<" src pre-sigmoid"<<srcdptr[0]<<" loss "<<loss;
    if (static_cast<float>(probptr[0]) < (1.0f - static_cast<float>(probptr[0])))
      predict_0++;
    else
      predict_1++;

    if (ilabel == 0){
      true_0 ++;
      if (static_cast<float>(probptr[0]) < (1.0f - static_cast<float>(probptr[0]))){
        correct_0++;
        precision++;
      }
      //LOG(INFO)<<"ilabel 0"<<" prob_1: "<<static_cast<float>(probptr[0])<<" prob_0: "<<(1.0f - static_cast<float>(probptr[0]));
    }
    else if (ilabel ==1){
      true_1 ++;
      if (static_cast<float>(probptr[0]) >= (1.0f - static_cast<float>(probptr[0]))){
        correct_1++;
        precision++;
      }
      //LOG(INFO)<<"ilabel 1"<<" prob_1: "<<static_cast<float>(probptr[0])<<" prob_0: "<<(1.0f - static_cast<float>(probptr[0]));
    }
    //LOG(INFO)<<"precision "<<precision;
    probptr+=dim_; //!!!!!!!!!!!!add 1 to next sample!!!
    srcdptr+=dim_;
  }

  // write out probability matrix
  if (phase == kTest){
    const float* probptr_writeout=prob.dptr;
    ofstream probmatout;
    probmatout.open("/data/zhaojing/marble/AUC/version" + std::to_string(static_cast<int>(run_version_)) + "step" + std::to_string(static_cast<int>(print_step_)) + ".csv");
    print_step_ ++;
    for(int n=0;n<batchsize_;n++){
      int label_writeout=static_cast<int>(label[n]);
      probmatout << label_writeout << "," << probptr_writeout[0] << "\n";
      probptr_writeout+=dim_;
    }
    probmatout.close();
  }


  CHECK_EQ(probptr, prob.dptr+prob.shape.Size());
  float *metric=metric_.mutable_cpu_data();
  metric[0]=loss*scale_/(1.0f*batchsize_);
  metric[1]=precision*scale_/(1.0f*batchsize_);
  // LOG(ERROR)<<"loss: "<<metric[0];
  // LOG(ERROR)<<"precision: "<<metric[1];
  metric[2]=correct_0;
  metric[3]=predict_0;
  metric[4]=true_0;
  metric[5]=correct_1;
  metric[6]=predict_1;
  metric[7]=true_1;
  // LOG(INFO) << " loss: "<<metric[0]<<" precision: "<<metric[1];
  // LOG(INFO)<<" correct_0: "<<metric[2]<<" predict_0: "<<metric[3]<<" true_0: "<<metric[4];
  // LOG(INFO)<<" correct_1: "<<metric[5]<<" predict_1: "<<metric[6]<<" true_1: "<<metric[7];
}

void LogisticLossLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  const float* label=srclayers[1]->data(this).cpu_data();
  Blob<float>* gsrcblob=srclayers[0]->mutable_grad(this); //dim is 1
  float* gsrcptr=gsrcblob->mutable_cpu_data();
  Tensor<cpu, 1> gsrc(gsrcptr, Shape1(gsrcblob->count()));
  Tensor<cpu, 1> prob(data_.mutable_cpu_data(), Shape1(batchsize_));//have problem? or shape is just shape, data is data
  float* probdptr = prob.dptr; //after sigmoid
  for (int n = 0; n < batchsize_; n++){
    gsrcptr[n] = -label[n]*(1-probdptr[n]) + (1-label[n])*probdptr[n];
    /*if (n < 20)
      LOG(INFO)<<"gsrc: "<<gsrcptr[n];*/
  }
  gsrc*=scale_/(1.0f*batchsize_);
}
/********** * Implementation for SoftmaxLossLayer*************************/
void SoftmaxLossLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),2);
  data_.Reshape(srclayers[0]->data(this).shape());
  batchsize_=data_.shape()[0];
  dim_=data_.count()/batchsize_;
  topk_=proto.softmaxloss_conf().topk();
  metric_.Reshape(vector<int>{2});
  scale_=proto.softmaxloss_conf().scale();
}
void SoftmaxLossLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}
void SoftmaxLossLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers) {
  Shape<2> s=Shape2(batchsize_, dim_);
  Tensor<cpu, 2> prob(data_.mutable_cpu_data(), s);
  Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(), s);
  Softmax(prob, src);
  const float* label=srclayers[1]->data(this).cpu_data();
  const float* probptr=prob.dptr;
  float loss=0, precision=0;
  for(int n=0;n<batchsize_;n++){
    int ilabel=static_cast<int>(label[n]);
    CHECK_LT(ilabel,10);
    CHECK_GE(ilabel,0);
    float prob_of_truth=probptr[ilabel];

    /*if (n < 50)
      LOG(INFO)<<"n: "<<n<<" dim_: "<<dim_<<" prob of 1: "<<probptr[1];*/
    /*if (n < 30)
      LOG(INFO)<<"ilabel "<<ilabel<<" prob of 1: "<<probptr[1];*/
    if (phase == kTest)
      LOG(INFO)<<"ilabel "<<ilabel<<" predict prob-1 "<<probptr[1];

    loss-=log(std::max(prob_of_truth, FLT_MIN));
    vector<std::pair<float, int> > probvec;
    for (int j = 0; j < dim_; ++j) {
      probvec.push_back(std::make_pair(probptr[j], j));
    }
    std::partial_sort(
        probvec.begin(), probvec.begin() + topk_,
        probvec.end(), std::greater<std::pair<float, int> >());
    // check if true label is in top k predictions
    for (int k = 0; k < topk_; k++) {
      if (probvec[k].second == static_cast<int>(label[n])) {
        precision++;
        break;
      }
    }
    probptr+=dim_;
  }
  CHECK_EQ(probptr, prob.dptr+prob.shape.Size());
  float *metric=metric_.mutable_cpu_data();
  metric[0]=loss*scale_/(1.0f*batchsize_);
  metric[1]=precision*scale_/(1.0f*batchsize_);
}

void SoftmaxLossLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  const float* label=srclayers[1]->data(this).cpu_data();
  Blob<float>* gsrcblob=srclayers[0]->mutable_grad(this);
  gsrcblob->CopyFrom(data_);
  float* gsrcptr=gsrcblob->mutable_cpu_data();
  for(int n=0;n<batchsize_;n++){
    gsrcptr[n*dim_+static_cast<int>(label[n])]-=1.0f;
  }
  Tensor<cpu, 1> gsrc(gsrcptr, Shape1(gsrcblob->count()));
  gsrc*=scale_/(1.0f*batchsize_);
}

}  // namespace singa
