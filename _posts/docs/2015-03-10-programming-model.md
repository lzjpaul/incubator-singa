---
layout: post
title: Programming Model
category : docs
tagline:
tags : [programming model, API]
---
{% include JB/setup %}

We describe the programming model of SINGA in this article.
Base data structures are introduced firstly, and then we show examples for
users with different levels of deep learning background.

### Base Data Structures

#### Layer

Layer is the first class citizen in SINGA. Users construct their deep learning
models by creating layer objects and combining them. SINGA
takes care of running BackPropagation (or Contrastive Divergence) algorithms
to calculate the gradients for parameters and calling [Updaters](#updater) to
update them.

    class Layer{
      /**
       * Setup layer properties.
       * Setup the shapes for data and parameters, also setup some properties
       * based on the layer configuration and connected src layers.
       * @param conf user defined layer configuration of type [LayerProto](#netproto)
       * @param srclayers layers connecting to this layer
       */
      Setup(conf, srclayers);
      /**
       * Setup the layer properties.
       * This function is called if the model is partitioned due to distributed
       * training. Shape of the layer is already set by the partition algorithm,
       * and is passed in to set other properties.
       * @param conf user defined layer configuration of type [LayerProto](#netproto)
       * @param shape shape set by partition algorithm (for distributed training).
       * @param srclayers layers connecting to this layer
       */
      SetupAfterPartition(conf, shape, srclayers);
      /**
       * Compute features of this layer based on connected layers.
       * BP and CD will call this to calculate gradients
       * @param training boolean phase indicator for training or test
       * @param srclayers layers connecting to this layer
       */
      ComputeFeature(training, srclayers);
      /**
       * Compute gradients for parameters and connected layers.
       * BP and CD will call this to calculate gradients
       * @param srclayers layers connecting to this layer.
       */
      ComputeGradient(srclayers)=0;
    }

The above pseudo code shows the base Layer class. Users override these
methods to implement their own layer classes. For example, we have implemented
popular layers like ConvolutionLayer, InnerProductLayer. We also provide a
DataLayer which is a base layer for loading (and prefetching) data from disk or HDFS. A base ParserLayer
is created for parsing the raw data and convert it into records that are recognizable by SINGA.

#### NetProto

Since deep learning models consist of multiple layers. The model structure includes
the properties of each layer and the connections between layers. SINGA uses
google protocol buffer for users to configure the model structure. The protocol
buffer message for the model structure is defined as:

    NetProto{
      repeated LayerProto layer;
    }

    LayerProto{
      string name; // user defined layer name for displaying
      string type; // One layer class has a unique type.
      repeated string srclayer_name; // connected layer names;
      repeated ParamProto param; // parameter configurations
      ...
    }

Users can create a plain text file and fill it with the configurations. SINGA
parses it according to user provided path.

#### Param

The Param class is shown below. Users do not need to extend the Param class for
most cases. We make it a base class just for future extension. For example,
if a new initialization trick is proposed in the future, we can override the `Init`
method to implement it.

    Param{
      /**
       * Set properties of the parameter.
       * @param conf user defined parameter configuration of type ParamProto
       * @param shape shape of the parameter
      Setup(conf, shape);
      /**
       * Initialize the data of the parameter.
       /
      Init();
      ...// methods to handle synchronizations with parameter servers and other workers
    }

#### Updater

There are many SGD extensions for updating parameters,
like [AdaDelta](http://arxiv.org/pdf/1212.5701v1.pdf),
[AdaGrad](http://www.magicbroom.info/Papers/DuchiHaSi10.pdf),
[RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf),
[Nesterov](http://scholar.google.com/citations?view_op=view_citation&hl=en&user=DJ8Ep8YAAAAJ&citation_for_view=DJ8Ep8YAAAAJ:hkOj_22Ku90C)
and SGD with momentum. We provide a base Updater to deal with these algorithms.
New parameter updating algorithms can be added by extending the base Updater.

    Updater{
      /**
      * @param proto user configuration for the updater.
      Init(conf);
      /**
      * Update parameter based on its gradient
      * @param step training step
      * @param param the Param object
      */
      Update(step, param);
    }

### Examples

The [MLP example]({{ BASE_PATH }})
shows how to configure the model through google protocol buffer.

