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

#ifndef SINGA_NEURALNET_INPUT_LAYER_H_
#define SINGA_NEURALNET_INPUT_LAYER_H_

#include <string>
#include <vector>
#include <thread>
#include "singa/io/store.h"
#include "singa/io/kvfile.h"
#include "singa/neuralnet/layer.h"

namespace singa {

/**
 * Base class for loading data from Store.
 */
class StoreInputLayer : virtual public InputLayer {
 public:
  ~StoreInputLayer();
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;


 protected:
  /**
   * Parsing the (key, val) tuple to get feature (and label).
   * Subclasses must implment this function.
   * @param[in] k parse this tuple as the k-th instance of one mini-batch.
   * @param[in] flag used to guide the parsing, e.g., kDeploy phase should not
   * parse labels from the tuple.
   * @param[in] key
   * @param[in] val
   */
  virtual bool Parse(int k, int flag, const string& key, const string& val) = 0;

 protected:
  int batchsize_ = 1;
  int random_skip_ = 0;
  io::Store* store_ = nullptr;
};

/**
 * Base layer for parsing a key-value tuple as a feature vector with fixed
 * length. The feature shape is indicated by users in the configuration.
 * Each tuple may has a label.
 */
class SingleLabelRecordLayer : public StoreInputLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;

 protected:
  /**
   * Load a single record (tuple), e.g., the mean or standard variance vector.
   */
  virtual void LoadRecord(const string& backend, const string& path,
      Blob<float>* to) = 0;

 protected:
  /**
   * Feature standardization by processing each feature dimension via
   * @f$ y = (x - mu)/ std @f$
   * <a href= "http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing">
   * UFLDL</a>
   */
  Blob<float> mean_, std_;
};
/**
 * Specific layer that parses the value string loaded by Store as a line from
 * a CSV file.
 *
 * It assumes the first column is the label except that has_label_ is configured
 * to false. Or the data is used in deploy mode.
 */
class CSVInputLayer : public SingleLabelRecordLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;

 protected:
  bool Parse(int k, int flag, const string& key, const string& val) override;
  void LoadRecord(const string& backend,
                  const string& path,
                  Blob<float>* to) override;

 private:
  std::string sep_;
  bool has_label_;
};


/**
 * Specific layer that parses the value string loaded by Store into a
 * RecordProto.
 */
class RecordInputLayer : public SingleLabelRecordLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;

 protected:
  /**
   * Parse key as instance ID and val into RecordProto.
   * @copydetails StoreInputLayer::Parse()
   */
  bool Parse(int k, int flag, const string& key, const string& val) override;
  void LoadRecord(const string& backend,
                  const string& path,
                  Blob<float>* to) override;

 private:
  // TODO(wangwei) decode the image
  bool encoded_;
};

/**
 * Occlude layer that parses the value string loaded by Store into a
 * RecordProto.
 */
class OccludeInputLayer : public SingleLabelRecordLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;

 protected:
  /**
   * Parse key as instance ID and val into RecordProto.
   * @copydetails StoreInputLayer::Parse()
   */
  bool Parse(int k, int flag, const string& key, const string& val) override;
  void LoadRecord(const string& backend,
                  const string& path,
                  Blob<float>* to) override;

 private:
  // TODO(wangwei) decode the image
  bool encoded_;
  int test_sample_;
};

/**
 * Multi destination layer that parses the value string 
 * to several destination layers.
 */
class Multi_dstInputLayer : public SingleLabelRecordLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;

 protected:
  /**
   * Parse key as instance ID and val into RecordProto.
   * @copydetails StoreInputLayer::Parse()
   */
  bool Parse(int k, int flag, const string& key, const string& val) override;
  void LoadRecord(const string& backend,
                  const string& path,
                  Blob<float>* to) override;

 const Blob<float>& data(const Layer* from) override{
    if (from != nullptr){
      //LOG(ERROR)<<"not nullptr";
      if ( strcmp((from->name()).c_str(), "Group1") == 0 ) //any other better solutions?
        return group1_data_;
      else if ( strcmp((from->name()).c_str(), "Group2") == 0 )
        return group2_data_;
      else if ( strcmp((from->name()).c_str(), "Group3") == 0 )
        return group3_data_;
      else if ( strcmp((from->name()).c_str(), "Group4") == 0 )
        return group4_data_;
      else if ( strcmp((from->name()).c_str(), "Group5") == 0 )
        return group5_data_;
      else if ( strcmp((from->name()).c_str(), "Group6") == 0 )
        return group6_data_;
      else if ( strcmp((from->name()).c_str(), "Group7") == 0 )
        return group7_data_;
      else if ( strcmp((from->name()).c_str(), "Group8") == 0 )
        return group8_data_;
      else if ( strcmp((from->name()).c_str(), "Group9") == 0 )
        return group9_data_;
      else if ( strcmp((from->name()).c_str(), "Group10") == 0 )
        return group10_data_;
      else if ( strcmp((from->name()).c_str(), "Group11") == 0 )
        return group11_data_;
      else if ( strcmp((from->name()).c_str(), "Group12") == 0 )
        return group12_data_;
      else if ( strcmp((from->name()).c_str(), "Group13") == 0 )
        return group13_data_;
      else if ( strcmp((from->name()).c_str(), "Group14") == 0 )
        return group14_data_;
      else if ( strcmp((from->name()).c_str(), "Group15") == 0 )
        return group15_data_;
      else if ( strcmp((from->name()).c_str(), "Group16") == 0 )
        return group16_data_;
      else if ( strcmp((from->name()).c_str(), "Group17") == 0 )
        return group17_data_;
      else if ( strcmp((from->name()).c_str(), "Group18") == 0 )
        return group18_data_;
      else if ( strcmp((from->name()).c_str(), "Group19") == 0 )
        return group19_data_;
      else if ( strcmp((from->name()).c_str(), "Group20") == 0 )
        return group20_data_;
      else if ( strcmp((from->name()).c_str(), "Group21") == 0 )
        return group21_data_;
      else{
        LOG(ERROR)<<"no data returned in the Multi_distInputLayer return data_";
        return data_;
      }
    }
    else{
      LOG(ERROR)<<"nullptr";
      return data_;
    }
  }
 Blob<float>* mutable_data(const Layer* from) override{
    if (from != nullptr){
      //LOG(ERROR)<<"not nullptr";
      if ( strcmp((from->name()).c_str(), "Group1") == 0 ) //any other better solutions?
        return &group1_data_;
      else if ( strcmp((from->name()).c_str(), "Group2") == 0 )
        return &group2_data_;
      else if ( strcmp((from->name()).c_str(), "Group3") == 0 )
        return &group3_data_;
      else if ( strcmp((from->name()).c_str(), "Group4") == 0 )
        return &group4_data_;
      else if ( strcmp((from->name()).c_str(), "Group5") == 0 )
        return &group5_data_;
      else if ( strcmp((from->name()).c_str(), "Group6") == 0 )
        return &group6_data_;
      else if ( strcmp((from->name()).c_str(), "Group7") == 0 )
        return &group7_data_;
      else if ( strcmp((from->name()).c_str(), "Group8") == 0 )
        return &group8_data_;
      else if ( strcmp((from->name()).c_str(), "Group9") == 0 )
        return &group9_data_;
      else if ( strcmp((from->name()).c_str(), "Group10") == 0 )
        return &group10_data_;
      else if ( strcmp((from->name()).c_str(), "Group11") == 0 )
        return &group11_data_;
      else if ( strcmp((from->name()).c_str(), "Group12") == 0 )
        return &group12_data_;
      else if ( strcmp((from->name()).c_str(), "Group13") == 0 )
        return &group13_data_;
      else if ( strcmp((from->name()).c_str(), "Group14") == 0 )
        return &group14_data_;
      else if ( strcmp((from->name()).c_str(), "Group15") == 0 )
        return &group15_data_;
      else if ( strcmp((from->name()).c_str(), "Group16") == 0 )
        return &group16_data_;
      else if ( strcmp((from->name()).c_str(), "Group17") == 0 )
        return &group17_data_;
      else if ( strcmp((from->name()).c_str(), "Group18") == 0 )
        return &group18_data_;
      else if ( strcmp((from->name()).c_str(), "Group19") == 0 )
        return &group19_data_;
      else if ( strcmp((from->name()).c_str(), "Group20") == 0 )
        return &group20_data_;
      else if ( strcmp((from->name()).c_str(), "Group21") == 0 )
        return &group21_data_;
      else{
        LOG(ERROR)<<"no mutable_data returned in the Multi_distInputLayer return &data_";
        return &data_;
      }
    }
    else{
      LOG(ERROR)<<"nullptr";
      return &data_;
    }
  }
 private:
  // TODO(wangwei) decode the image
  bool encoded_;
  int test_sample_;
  int group1_dim_, group2_dim_, group3_dim_, group4_dim_, group5_dim_, group6_dim_, group7_dim_, group8_dim_, group9_dim_, group10_dim_, group11_dim_, group12_dim_, group13_dim_, 
  group14_dim_, group15_dim_, group16_dim_, group17_dim_, group18_dim_, group19_dim_, group20_dim_, group21_dim_;
  Blob<float> group1_data_, group2_data_, group3_data_, group4_data_, group5_data_, group6_data_, group7_data_, group8_data_, group10_data_, group11_data_, group12_data_,
  group13_data_, group14_data_, group15_data_, group16_data_, group17_data_, group18_data_, group19_data_, group20_data_, group21_data_;
  Blob<float> group9_data_;
};

/**
 * Do preprocessing for images, including cropping, mirroring, resizing.
 */
class ImagePreprocessLayer : public InputLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers);

 private:
  bool mirror_ = false;
  int cropsize_ = 0;
  int resize_ = 0;
  float scale_ = 1;
};

/**
 * TODO(wangwei) Layer for prefetching data records and parsing them.
 *
 * This layer controls the prefetching thread, i.e.,
 * creating and joining the prefetching thread.
 */
class PrefetchLayer : public Layer {
 public:
  ~PrefetchLayer();
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override {}

 protected:
  std::thread thread_;
};

/****************Deprecated layers******************/
/**
 * @deprecated please use the StoreInputLayer.
 *
 * Base layer for reading ::Record  from local Shard, HDFS, lmdb, etc.
 */
class DataLayer: virtual public InputLayer {
 public:
  Blob<float>* mutable_data(const Layer* layer) override { return nullptr; }
  ConnectionType dst_layer_connection() const override {
    return kOneToMany;
  }

  inline int batchsize() const { return batchsize_; }
  virtual const Record& sample() const {
    return sample_;
  }
  /**
   * @return the loaded records
   */
  virtual const std::vector<Record>& records() const {
    return records_;
  }

 protected:
  int random_skip_;
  int batchsize_;
  Record sample_;
  std::vector<Record> records_;
};
/**
 * @deprecated Please use the subclasses of StoreInputLayer.
 *
 * Layer for loading Record from DataShard.
 */
class ShardDataLayer : public DataLayer {
 public:
  ~ShardDataLayer();

  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;

 private:
  DataShard* shard_;
};
/**
 * @deprecated please use the subclasses of StoreInputLayer.
 *
 * Layer for loading Record from LMDB.
 */
#ifdef USE_LMDB
#include <lmdb.h>
class LMDBDataLayer : public DataLayer {
 public:
  ~LMDBDataLayer();

  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void OpenLMDB(const std::string& path);
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ConvertCaffeDatumToRecord(const CaffeDatum& datum,
                                 SingleLabelImageRecord* record);

 private:
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};
#endif

/******************Parser layers***************/
/**
 * @deprecated Please use the subclasses of StoreInputLayer which load and parse
 * data in a single layer.
 *
 * Base layer for parsing the input records into Blobs.
 */
class ParserLayer : public InputLayer {
 public:
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override {}
  ConnectionType dst_layer_connection() const override {
    return kOneToMany;
  }
  /**
   * Parse records from DataLayer into blob.
   */
  virtual void ParseRecords(int flag, const std::vector<Record>& records,
      Blob<float>* blob) = 0;
};
/**
 *
 * @deprecated Please use the SingleLabelRecordLayer which parses both feature
 * and label for each record. Its aux_data() function returns the parsed labels.
 *
 * Derived from ParserLayer to parse label in SingaleLabelImageRecord loaded by
 * ShardDataLayer.
 */
class LabelLayer : public ParserLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ParseRecords(int flag, const std::vector<Record>& records,
                    Blob<float>* blob) override;
};

/**
 * @deprecated Please use the subclasses of StoreInputLayer.
 *
 * Derived from ParserLayer to parse MNIST feature from SingaleLabelImageRecord.
 */
class MnistLayer : public ParserLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ParseRecords(int flag, const std::vector<Record>& records,
                    Blob<float>* blob) override;

 protected:
  float norm_a_, norm_b_;
};
/**
 * @deprecated please use the ImagePreprocessLayer which preprocess image
 * feature from data Blob of source layers.
 *
 * Derived from ParserLayer to parse RGB image feature from
 * SingaleLabelImageRecord.
 */
class RGBImageLayer : public ParserLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ParseRecords(int flag, const std::vector<Record>& records,
                    Blob<float>* blob) override;

 private:
  float scale_;
  int cropsize_;
  bool mirror_;
  Blob<float> mean_;
};
}  // namespace singa

#endif  // SINGA_NEURALNET_INPUT_LAYER_H_
