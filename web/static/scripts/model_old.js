/**
 * Created by Aaron on 6/20/2015.
 */
"use strict";

(function(){
  window.Model={};

  Model.Config={
    notShowId:60,  //if field id is bigger than this, then not show
    host:"http://dbpcm.d1.comp.nus.edu.sg:8153/",
    apiUrl:"http://dbpcm.d1.comp.nus.edu.sg:8153/api/"
  }

  var Package =function(name,comment){

    this.name=name;
    this.comment=comment;
    this.messages=[];
    this.enums=[];
    this.rootMessages=[];
  }

  Package.prototype={
    addMessage:function(message){
      this.messages.push(message);
    },
    addRootMessage:function(message){
      this.rootMessages.push(message);
    },
    addEnum:function(_enum){
      this.enums.push(_enum);
    },
    getEnumByName:function(name){
      for(var index in this.enums){
        if(this.enums[index]["name"]==name){
          return this.enums[index];
        }
      }
      return;
    },
    getMessageByName:function(name){
      for(var index in this.messages){
        if(this.messages[index]["name"]==name){
          return this.messages[index];
        }
      }
      return;
    },
    toString:function(){

      var str="";
      str+=printComment(this.comment);

      str+="package "+this.name+";"+"\n";
      for(var index in this.enums){
        str+="\n"+this.enums[index].toString()+"\n";
      }
      for(var index in this.messages){
        str+="\n"+this.messages[index].toString()+"\n";
      }
      return str;
    },
    renderForm:function(container){
      var me=this;
      $.each(this.rootMessages,function(index,message){
        message.setConfName();

        var source = $("#formTemplate").html();
        var template = Handlebars.compile(source);
        var html= template(message);
        container.append(html);

        var messageFormContainer = container.find("#"+message.name+"Form");
        message.renderForm(messageFormContainer);

        $("#value").append("<div id='value_"+message.name+"'><h3>"+message.confName+"</h3> <textarea></textarea></div>")
      });

      container.find("h2>.toggle-optional-btn").click(function(){

        var text =$(this).text();
        $(this).text(text=="show optional"?"hide optional":"show optional");

        var $message = $(this).parent().next();
        $message.find(">.optional").not(".field_group").toggle();

        return false;
      });

      var currentMessageName="";

      $("#uploadModal .submit").on("click",function(){
        var file=$("#uploadModal input").prop("files")[0];

        readBlobAsText(file, function (err, result) {
          if (err) {
            alert(err.name + '\n' + err.message);
            return;
          }
          $("#value_"+currentMessageName+" textarea").val(result);
          var confObj={}
          parseMessage(confObj,result);
          var message = Model.package.getMessageByName(currentMessageName);

          message.renderForm($("#"+currentMessageName+"Form"),confObj);

          $("#uploadModal").modal("hide");
        });

        return false;
      });
      $("#ClusterProtoLoadBtn").click(function(){
        $("#uploadModal").modal("show");
        currentMessageName="ClusterProto";
      });

      $("#ModelProtoLoadBtn").click(function(){
        $("#uploadModal").modal("show");
        currentMessageName="ModelProto";
      });

      function readBlobAsText(blob, callback) {
        var fr = new FileReader();
        fr.onload = function (e) {
          var fr = e.currentTarget;
          if (typeof callback === 'function') {
            callback.call(fr, null, fr.result);
          }
        };
        fr.onerror = function (e) {
          var fr = e.currentTarget;
          if (typeof callback === 'function') {
            callback.call(fr, fr.error, null);
          }
        };
        fr.readAsText(blob);
      }


    },
    showValue:function(){
      $.each(this.rootMessages,function(index,message){
        var messageFormContainer = $("#"+message.name+"Form");
        var value=message.getValue(messageFormContainer);
        //console.log(value);
       // value=value.replace(new RegExp("\n",'g'),"<br/>");
        value="<h3>"+message.confName()+"</h3> <textarea>"+value+"</textarea>";
        $("#value_"+message.name).html(value);
      });

    }
  }

  var Message =function(name,comment){

    this.name=name;
    this.comment=comment;
    this.fields=[];
  }

  Message.prototype={
    setConfName:function(){
      this.confName=this.name.substr(0,this.name.length-5)+".conf";
    },
    addField:function(field){
      this.fields.push(field);
    },
    toString:function(){
      var str="";
      str+=printComment(this.comment);
      str+="message "+this.name+"{"+"\n";
      for(var index in this.fields){
        str+="\n"+this.fields[index].toString()+"\n";
      }
      str+="}";
      return str;
    },
    renderForm:function(container,valueObj){
      $.each(this.fields,function(index,field){
        field.renderForm(container,valueObj?valueObj[field.name]:undefined);
      });

      container.find("input").change(function(){
        Model.package.showValue();
      })
      container.find("select").change(function(){
        Model.package.showValue();
      })

      this.additionBind(container);
    },
    getValue:function(container){

      var value="";
      $.each(this.fields,function(index,field){
        value+=field.getValue(container);
      });
      return value;
    },
    additionBind:function(container){  //different Message bind different actions

    }

  }

  var Field = function(name,comment){
    this.name=name;
    this.comment=comment;
    this.values=[];
  }

  Field.prototype={
    init:function(id,rule,type,defaultValue,options){
      this.id=id;
      this.rule=rule;
      this.type=type;
      this.defaultValue=defaultValue;
      this.options=options;

      if(rule=="required"){
        this.isRequired=true;
      }
      return this;
    },
    toString:function(){
      var str="";
      str+=printComment(this.comment);
      str+="  "+this.rule+" "+this.type+" "+this.name+"="+this.id;
      if(this.defaultValue!=undefined){
        if(this.type=="string"){
          str += " [default=\"" + this.defaultValue + "\"]";
        }else {
          str += " [default=" + this.defaultValue + "]";
        }
      }
      str+=";"

      return str;
    },
    renderForm:function(container,valueObj){

      container.find(">.field_"+this.name).remove();

      if(this.id>Model.Config.notShowId){
        return;
      }

      this._enum=Model.package.getEnumByName(this.type);
      if(this.rule=="repeated"){
        this.isRepeated=true;
      }
      if(this._enum){
        this.isEnum=true;
        this.renderEnum(container,valueObj);
      }else{
        this.message=Model.package.getMessageByName(this.type);
        if(this.message){
          this.isMessage=true;
          this.renderMessage(container,valueObj);
        }else{
          this.renderText(container,valueObj);
        }

      }
    },
    renderText:function(container,valueObj){
      var source = $("#textTemplate").html();
      var template = Handlebars.compile(source);
      var html= template(this);
      var $field = $(html).appendTo(container);

      if(this.isRepeated) {
        $field.find(".add-btn:first").click(function () {
          $field.find("input:first").clone().insertAfter($field.find("input:last"));
          $field.find(".remove-btn:first").show();
          return false;
        }).show();
        $field.find(".remove-btn:first").click(function () {
          $field.find("input:last").remove();
          if ($field.find("input").length == 1) {
            $field.find(".remove-btn:first").hide();
          }
          return false;
        });
        if (valueObj) {
          if (valueObj.push) {
            $field.find("input:first").val(valueObj[0]);
            for (var i = 1; i < valueObj.length; i++) {
              $field.find("input:first").clone().insertAfter($field.find("input:last")).val(valueObj[i]);
            }

          } else {
            $field.find("input:first").val(valueObj);
          }
        }
      }else {
          if (valueObj) {
            $field.find("input:first").val(valueObj);
          }

      }

      return $field;
    },
    renderEnum:function(container,valueObj){


      var source = $("#selectTemplate").html();
      var template = Handlebars.compile(source);
      var html= template($.extend({"_enum":this._enum},this));
      var $field = $(html).appendTo(container);


      if(this.isRepeated){
        $field.find(".add-btn:first").click(function(){
          $field.find("select:first").clone().insertAfter($field.find("select:last"));
          $field.find(".remove-btn:first").show();
          return false;
        }).show();
        $field.find(".remove-btn:first").click(function(){
          $field.find("select:last").remove();
          if($field.find("select").length==1){
            $field.find(".remove-btn:first").hide();
          }
          return false;
        });
        if(valueObj){
          if(valueObj.push){
            $field.find("select:first").val(this._enum.getValueFormKey(valueObj[0]));
            for(var i=1;i<valueObj.length;i++){
              $field.find("select:first").clone().insertAfter($field.find("select:last")).val(this._enum.getValueFormKey(valueObj[i]));
            }
          }else{
            $field.find("select").val(this._enum.getValueFormKey(valueObj));
          }
        }
      }else{
        var value=this.defaultValue;
        if(valueObj){
          value=valueObj;
        }
        if(value!=undefined) { //set default value
          $field.find("select").val(this._enum.getValueFormKey(value));
        }

      }
      return $field;
    },
    renderMessage:function(container,valueObj){

      var me=this;
      var source = $("#messageTemplate").html();
      var template = Handlebars.compile(source);
      var html= template(this);
      var $field = $(html).appendTo(container);

      $field.find(".toggle-btn:first").click(toggleMessage);
      $field.find(".toggle-optional-btn:first").click(toggleOptional);

      if(this.isRepeated){
        $field.find(".add-btn:first").click(function(){
          var $newField=$field.clone().insertAfter($field);
          $newField.find(".messageField").empty();
          $newField.find(".toggle-btn:first").click(toggleMessage);
          $newField.find(".add-btn:first").hide();
          $newField.find(".toggle-optional-btn:first").click(toggleOptional);
          $newField.find(".remove-btn:first").click(function(){
            $newField.remove();
          }).show();
          return false;
        }).show();
        if(valueObj){
          me.message.renderForm($field.find(".messageField"),valueObj[0]);

          for(var i=1;i<valueObj.length;i++){
            var $newField=$field.clone().insertAfter($field);
            $newField.find(".messageField").empty();
            $newField.find(".toggle-btn:first").click(toggleMessage);
            $newField.find(".add-btn:first").hide();
            $newField.find(".toggle-optional-btn:first").click(toggleOptional);
            $newField.find(".remove-btn:first").click(function(){
              $newField.remove();
            })
            me.message.renderForm($newField.find(".messageField"),valueObj[i]);
          }
        }
      }else{
        me.message.renderForm($field.find(".messageField"),valueObj);
      }

      function toggleMessage(){
        var text =$(this).text();
        $(this).text(text=="show"?"hide":"show");

        var $message = $(this).parent().parent().find(".messageField:first");
        if($message.html()=="") {
          me.message.renderForm($message);
        }
        $message.toggle();
        return false;
      }

      function toggleOptional(){
        var text =$(this).text();
        $(this).text(text=="show optional"?"hide optional":"show optional");

        var $message = $(this).parent().parent().find(".messageField:first");
        $message.find(">.optional").not(".field_group").toggle();

        return false;
      }


      return $field;
    },
    getValue:function(container){
      var me=this;
      var value="";
      if(this.isEnum){

        var $field=container.find(">.field_"+this.name);

        $field.find("select").each(function(index,field){
          var v=$(field).find("option:selected").text();
          if(v=="please select"){
            return;
          }
          if(v==me.defaultValue){ //default value don't show;
            return;
          }
          value+=me.name+":"+v+"\n";

        });
      }else if(this.isMessage){

        var $fields=container.find(">.field_"+this.name);
        $.each($fields,function(index,field){
          var $message = $(field).find(".messageField:first");
          if($message.html()!=""){
            var v=me.message.getValue($message);
            if(v==""){
              return;
            }

            if(v==me.defaultValue){ //default value don't show;
              return;
            }
            value+=me.name+"{\n"+v+"}\n";
          }
        });

      }else{

        var $field=container.find(">.field_"+this.name);

        $field.find("input").each(function(index,field){
          var v=$(field).val();
          if(v==""){
            return;
          }
          if(me.type=="bool"){
            v = v.toLowerCase()=="true"?true:false;
          }

          if(v==me.defaultValue){ //default value don't show;
            return;
          }
          if(me.type=="string"){
            value+=me.name+":\""+v+"\"\n";
          }else{
            value+=me.name+":"+v+"\n";
          }
        });




      }

      return value;
    }

  }


  var Enum = function(name,comment){

    this.name=name;
    this.comment=comment;
    this.values=[];
  }

  Enum.prototype={
    addValue:function(value){
      this.values.push(value);
    },
    toString:function(){
      var str="";
      str+=printComment(this.comment);
      str+="enum "+this.name+"{"+"\n";
      for(var index in this.values){
        str+=this.values[index].toString()+"\n";
      }
      str+="}";
      return str;
    },
    getValueFormKey:function(key){
      for(var index in this.values){
        if(this.values[index].key==key){
          return this.values[index].value;
        }
      }
      return;
    }
  }

  var EnumValue = function(key,value,comment){
    this.key=key;
    this.value=value;
    this.comment=comment;
  }

  EnumValue.prototype={
    toString:function(){

      return printComment(this.comment)+"  "+this.key+" = "+this.value+"; ";
    }
  }

  function printComment(comment){
    var str="";
    if(comment&&comment!="") {
      var comments = comment.split("\n");
      for(var index in comments){
        str+="//"+comments[index]+"\n";
      }
    }
    return str;
  }

  //begin auto generate

  var packsinga=new Package('singa');

  var ClusterProtoObj=new Message('ClusterProto','');
  ClusterProtoObj.addField(new Field('nworker_groups', '').init(1 ,'optional','int32',1));
  ClusterProtoObj.addField(new Field('nserver_groups', '').init(2 ,'optional','int32',1));
  ClusterProtoObj.addField(new Field('nworkers_per_group', '').init(3 ,'optional','int32',1));
  ClusterProtoObj.addField(new Field('nservers_per_group', '').init(4 ,'optional','int32',1));
  ClusterProtoObj.addField(new Field('nworkers_per_procs', '').init(5 ,'optional','int32',1));
  ClusterProtoObj.addField(new Field('nservers_per_procs', '').init(6 ,'optional','int32',1));
  ClusterProtoObj.addField(new Field('hostfile', 'Used in standalone mode, one ip or hostname per line For YARN or Mesos version, the processes are allocted dynamically, hence no need to specify the hosts statically').init(10 ,'optional','string',""));
  ClusterProtoObj.addField(new Field('server_worker_separate', 'servers and workers in different processes?').init(11 ,'optional','bool',false));
  ClusterProtoObj.addField(new Field('start_port', 'port number is used by ZeroMQ').init(13 ,'optional','int32',6723));
  ClusterProtoObj.addField(new Field('workspace', 'local workspace, train/val/test shards, checkpoint files').init(14,'required','string'));
  ClusterProtoObj.addField(new Field('log_dir', 'relative path to workspace. if not set, use the default dir of glog').init(15 ,'optional','string',"/tmp"));
  ClusterProtoObj.addField(new Field('zookeeper_host', 'ip/hostname : port [, ip/hostname : port]').init(16 ,'optional','string',"localhost:2181"));
  ClusterProtoObj.addField(new Field('stub_timeout', '').init(30 ,'optional','int32',5000));
  ClusterProtoObj.addField(new Field('worker_timeout', '').init(31 ,'optional','int32',5000));
  ClusterProtoObj.addField(new Field('server_timeout', '').init(32 ,'optional','int32',5000));
  ClusterProtoObj.addField(new Field('server_update', 'conduct updates at server side; otherwise do it at worker side').init(40 ,'optional','bool',true));
  ClusterProtoObj.addField(new Field('share_memory', 'share memory space between worker groups in one procs').init(41 ,'optional','bool',true));
  packsinga.addMessage(ClusterProtoObj);

  var PhaseObj=new Enum('Phase','');
  PhaseObj.addValue(new EnumValue('kTrain', 0,''));
  PhaseObj.addValue(new EnumValue('kValidation', 1,''));
  PhaseObj.addValue(new EnumValue('kTest', 2,''));
  PhaseObj.addValue(new EnumValue('kPositive', 3,'postivie phase for contrastive divergence algorithm'));
  PhaseObj.addValue(new EnumValue('kNegative', 4,'negative phase for contrastive divergence algorithm'));
  packsinga.addEnum(PhaseObj);

  var GradCalcAlgObj=new Enum('GradCalcAlg','');
  GradCalcAlgObj.addValue(new EnumValue('kBackPropagation', 1,'BP algorithm for feed-forward models, e.g., CNN, MLP, RNN'));
  GradCalcAlgObj.addValue(new EnumValue('kContrastiveDivergence', 2,'CD algorithm for RBM, DBM etc., models'));
  packsinga.addEnum(GradCalcAlgObj);

  var ModelProtoObj=new Message('ModelProto','');
  ModelProtoObj.addField(new Field('name', 'model name, e.g., "cifar10-dcnn", "mnist-mlp"').init(1,'required','string'));
  ModelProtoObj.addField(new Field('display_frequency', 'frequency of displaying training info').init(3,'required','int32'));
  ModelProtoObj.addField(new Field('train_steps', 'total num of steps for training').init(5,'required','int32'));
  ModelProtoObj.addField(new Field('updater', 'configuration of SGD updater, including learning rate, etc.').init(7,'required','UpdaterProto'));
  ModelProtoObj.addField(new Field('alg', 'gradient calculation algorithm').init(8 ,'required','GradCalcAlg','kBackPropagation'));
  ModelProtoObj.addField(new Field('neuralnet', '').init(9,'required','NetProto'));
  ModelProtoObj.addField(new Field('validation_steps', 'total num of steps for validation').init(30 ,'optional','int32','0'));
  ModelProtoObj.addField(new Field('test_steps', 'total num of steps for test').init(31 ,'optional','int32','0'));
  ModelProtoObj.addField(new Field('validation_frequency', 'frequency of validation').init(32,'optional','int32'));
  ModelProtoObj.addField(new Field('test_frequency', 'frequency of test').init(33 ,'optional','int32','0'));
  ModelProtoObj.addField(new Field('checkpoint_frequency', 'frequency of checkpoint').init(34 ,'optional','int32','0'));
  ModelProtoObj.addField(new Field('warmup_steps', 'send parameters to servers after training for this num of steps').init(35 ,'optional','int32','0'));
  ModelProtoObj.addField(new Field('display_after_steps', 'start display after this num steps').init(60,'optional','int32','0'));
  ModelProtoObj.addField(new Field('checkpoint_after_steps', 'start checkpoint after this num steps').init(61 ,'optional','int32','0'));
  ModelProtoObj.addField(new Field('test_after_steps', 'start test after this num steps').init(62 ,'optional','int32','0'));
  ModelProtoObj.addField(new Field('validation_after_steps', 'start validation after this num steps').init(63 ,'optional','int32','0'));
  ModelProtoObj.addField(new Field('step', 'last snapshot step').init(64 ,'optional','int32','0'));
  ModelProtoObj.addField(new Field('debug', 'display debug info').init(65 ,'optional','bool','false'));
  packsinga.addMessage(ModelProtoObj);

  var NetProtoObj=new Message('NetProto','');
  NetProtoObj.addField(new Field('layer', '').init(1,'repeated','LayerProto'));
  NetProtoObj.addField(new Field('partition_type', 'partitioning type for parallelism').init(3 ,'optional','PartitionType','kNone'));
  packsinga.addMessage(NetProtoObj);

  var InitMethodObj=new Enum('InitMethod','');
  InitMethodObj.addValue(new EnumValue('kConstant', 0,'fix the values of all parameters  a constant in the value field'));
  InitMethodObj.addValue(new EnumValue('kGaussian', 1,'sample gaussian with std and mean'));
  InitMethodObj.addValue(new EnumValue('kUniform', 2,'uniform sampling between low and high'));
  InitMethodObj.addValue(new EnumValue('kPretrained', 3,'copy the content and history which are from previous training'));
  InitMethodObj.addValue(new EnumValue('kGaussainSqrtFanIn', 4,'from Toronto Convnet, let a=1/sqrt(fan_in), w*=a after generating from Gaussian distribution'));
  InitMethodObj.addValue(new EnumValue('kUniformSqrtFanIn', 5,'from Toronto Convnet, rectified linear activation, let a=sqrt(3)/sqrt(fan_in), range is [-a, +a]; no need to set value=sqrt(3), the program will multiply it.'));
  InitMethodObj.addValue(new EnumValue('kUniformSqrtFanInOut', 6,'from Theano MLP tutorial, let a=sqrt(6/(fan_in+fan_out)). for tanh activation, range is [-a, +a], for sigmoid activation, range is [-4a, +4a], put the scale factor to value field. <a href="http://deeplearning.net/tutorial/mlp.html"> Theano MLP</a>'));
  packsinga.addEnum(InitMethodObj);

  var ParamProtoObj=new Message('ParamProto','');
  ParamProtoObj.addField(new Field('init_method', '').init(1 ,'required','InitMethod','kGaussian'));
  ParamProtoObj.addField(new Field('value', 'constant init').init(5 ,'optional','float','1'));
  ParamProtoObj.addField(new Field('low', 'for uniform sampling').init(6 ,'optional','float','-1'));
  ParamProtoObj.addField(new Field('high', '').init(7 ,'optional','float','1'));
  ParamProtoObj.addField(new Field('mean', 'for gaussian sampling').init(8 ,'optional','float','0'));
  ParamProtoObj.addField(new Field('std', '').init(9 ,'optional','float','1'));
  ParamProtoObj.addField(new Field('learning_rate_multiplier', 'multiplied on the global learning rate.').init(15 ,'optional','float','1'));
  ParamProtoObj.addField(new Field('weight_decay_multiplier', 'multiplied on the global weight decay.').init(16 ,'optional','float','1'));
  ParamProtoObj.addField(new Field('partition_dim', 'partition dimension, -1 for no partition').init(30 ,'optional','int32','-1'));
  ParamProtoObj.addField(new Field('shape', 'usually, the program will infer the param shape').init(31,'repeated','int32'));
  ParamProtoObj.addField(new Field('name', 'used for identifying the same params from diff models and display deug info').init(61 ,'optional','string','"param"'));
  ParamProtoObj.addField(new Field('id', 'used interally').init(62,'optional','int32'));
  ParamProtoObj.addField(new Field('split_threshold', 'parameter slice limit (Google Protobuf also has size limit)').init(63 ,'optional','int32','5000000'));
  ParamProtoObj.addField(new Field('owner', 'used internally').init(64 ,'optional','int32','-1'));
  packsinga.addMessage(ParamProtoObj);

  var PartitionTypeObj=new Enum('PartitionType','');
  PartitionTypeObj.addValue(new EnumValue('kDataPartition', 0,''));
  PartitionTypeObj.addValue(new EnumValue('kLayerPartition', 1,''));
  PartitionTypeObj.addValue(new EnumValue('kNone', 2,''));
  packsinga.addEnum(PartitionTypeObj);

  var LayerTypeObj=new Enum('LayerType','');
  LayerTypeObj.addValue(new EnumValue('kBridgeSrc', 15,''));
  LayerTypeObj.addValue(new EnumValue('kBridgeDst', 16,''));
  LayerTypeObj.addValue(new EnumValue('kConvolution', 1,''));
  LayerTypeObj.addValue(new EnumValue('kConcate', 2,''));
  LayerTypeObj.addValue(new EnumValue('kShardData', 3,''));
  LayerTypeObj.addValue(new EnumValue('kDropout', 4,''));
  LayerTypeObj.addValue(new EnumValue('kInnerProduct', 5,''));
  LayerTypeObj.addValue(new EnumValue('kLabel', 18,''));
  LayerTypeObj.addValue(new EnumValue('kLMDBData', 17,''));
  LayerTypeObj.addValue(new EnumValue('kLRN', 6,''));
  LayerTypeObj.addValue(new EnumValue('kMnist', 7,''));
  LayerTypeObj.addValue(new EnumValue('kPooling', 8,''));
  LayerTypeObj.addValue(new EnumValue('kPrefetch', 19,''));
  LayerTypeObj.addValue(new EnumValue('kReLU', 9,''));
  LayerTypeObj.addValue(new EnumValue('kRGBImage', 10,''));
  LayerTypeObj.addValue(new EnumValue('kSoftmaxLoss', 11,''));
  LayerTypeObj.addValue(new EnumValue('kSlice', 12,''));
  LayerTypeObj.addValue(new EnumValue('kSplit', 13,''));
  LayerTypeObj.addValue(new EnumValue('kTanh', 14,''));
  packsinga.addEnum(LayerTypeObj);

  var LayerProtoObj=new Message('LayerProto','');
  LayerProtoObj.addField(new Field('name', 'the layer name used for identification').init(1,'required','string'));
  LayerProtoObj.addField(new Field('srclayers', 'source layer names').init(3,'repeated','string'));
  LayerProtoObj.addField(new Field('param', 'parameters, e.g., weight matrix or bias vector').init(12,'repeated','ParamProto'));
  LayerProtoObj.addField(new Field('exclude', 'all layers are included in the net structure for training phase by default. some layers like data layer for loading test data are not used by training phase should be removed by setting the exclude field.').init(15,'repeated','Phase'));
  LayerProtoObj.addField(new Field('type', 'the layer type from the enum above').init(20,'required','LayerType'));
  LayerProtoObj.addField(new Field('convolution_conf', 'configuration for convolution layer').init(30,'optional','ConvolutionProto'));
  LayerProtoObj.addField(new Field('concate_conf', 'configuration for concatenation layer').init(31,'optional','ConcateProto'));
  LayerProtoObj.addField(new Field('dropout_conf', 'configuration for dropout layer').init(33,'optional','DropoutProto'));
  LayerProtoObj.addField(new Field('innerproduct_conf', 'configuration for inner product layer').init(34,'optional','InnerProductProto'));
  LayerProtoObj.addField(new Field('lmdbdata_conf', 'configuration for local response normalization layer').init(35,'optional','DataProto'));
  LayerProtoObj.addField(new Field('lrn_conf', 'configuration for local response normalization layer').init(45,'optional','LRNProto'));
  LayerProtoObj.addField(new Field('mnist_conf', 'configuration for mnist parser layer').init(36,'optional','MnistProto'));
  LayerProtoObj.addField(new Field('pooling_conf', 'configuration for pooling layer').init(37,'optional','PoolingProto'));
  LayerProtoObj.addField(new Field('prefetch_conf', 'configuration for prefetch layer').init(44,'optional','PrefetchProto'));
  LayerProtoObj.addField(new Field('relu_conf', 'configuration for rectified linear unit layer').init(38,'optional','ReLUProto'));
  LayerProtoObj.addField(new Field('rgbimage_conf', 'configuration for rgb image parser layer').init(39,'optional','RGBImageProto'));
  LayerProtoObj.addField(new Field('sharddata_conf', 'configuration for data layer').init(32,'optional','DataProto'));
  LayerProtoObj.addField(new Field('slice_conf', 'configuration for slice layer').init(41,'optional','SliceProto'));
  LayerProtoObj.addField(new Field('softmaxloss_conf', 'configuration for softmax loss layer').init(40,'optional','SoftmaxLossProto'));
  LayerProtoObj.addField(new Field('split_conf', 'configuration for split layer').init(42,'optional','SplitProto'));
  LayerProtoObj.addField(new Field('tanh_conf', 'configuration for tanh layer').init(43,'optional','TanhProto'));
  LayerProtoObj.addField(new Field('partition_type', 'partition type which overrides the partition type for neural net').init(59,'optional','PartitionType'));
  LayerProtoObj.addField(new Field('datablob', '').init(58 ,'optional','string','"unknow"'));
  LayerProtoObj.addField(new Field('share_param', 'names of parameters shared from other layers').init(60,'repeated','string'));
  LayerProtoObj.addField(new Field('locationid', 'TODO(wangwei): make location ID an array').init(61 ,'optional','int32','0'));
  LayerProtoObj.addField(new Field('partitionid', '').init(62 ,'optional','int32','0'));
  packsinga.addMessage(LayerProtoObj);

  var RGBImageProtoObj=new Message('RGBImageProto','');
  RGBImageProtoObj.addField(new Field('scale', 'scale factor for each pixel').init(1 ,'optional','float','1.0'));
  RGBImageProtoObj.addField(new Field('cropsize', 'size after cropping').init(2 ,'optional','int32','0'));
  RGBImageProtoObj.addField(new Field('mirror', 'mirror the image').init(3 ,'optional','bool','false'));
  RGBImageProtoObj.addField(new Field('meanfile', 'meanfile path').init(4 ,'optional','string','""'));
  packsinga.addMessage(RGBImageProtoObj);

  var PrefetchProtoObj=new Message('PrefetchProto','');
  PrefetchProtoObj.addField(new Field('sublayers', '').init(1,'repeated','LayerProto'));
  packsinga.addMessage(PrefetchProtoObj);

  var SplitProtoObj=new Message('SplitProto','');
  SplitProtoObj.addField(new Field('num_splits', '').init(1 ,'optional','int32','1'));
  packsinga.addMessage(SplitProtoObj);

  var TanhProtoObj=new Message('TanhProto','');
  TanhProtoObj.addField(new Field('outer_scale', 'A of A*tan(B*x)').init(1 ,'optional','float','1.0'));
  TanhProtoObj.addField(new Field('inner_scale', 'B of A*tan(B*x)').init(2 ,'optional','float','1.0'));
  packsinga.addMessage(TanhProtoObj);

  var SoftmaxLossProtoObj=new Message('SoftmaxLossProto','');
  SoftmaxLossProtoObj.addField(new Field('topk', 'computing accuracy against topk results').init(1 ,'optional','int32','1'));
  SoftmaxLossProtoObj.addField(new Field('scale', 'loss scale factor').init(30 ,'optional','float','1'));
  packsinga.addMessage(SoftmaxLossProtoObj);

  var ConvolutionProtoObj=new Message('ConvolutionProto','');
  ConvolutionProtoObj.addField(new Field('num_filters', 'The number of outputs for the layer').init(1,'required','int32'));
  ConvolutionProtoObj.addField(new Field('kernel', 'the kernel height/width').init(2,'required','int32'));
  ConvolutionProtoObj.addField(new Field('pad', 'The padding height/width').init(30 ,'optional','int32','0'));
  ConvolutionProtoObj.addField(new Field('stride', 'the stride').init(31 ,'optional','int32','1'));
  ConvolutionProtoObj.addField(new Field('bias_term', 'whether to have bias terms').init(32 ,'optional','bool','true'));
  packsinga.addMessage(ConvolutionProtoObj);

  var ConcateProtoObj=new Message('ConcateProto','');
  ConcateProtoObj.addField(new Field('concate_dimension', 'on which dimension, starts from 0').init(1,'required','int32'));
  ConcateProtoObj.addField(new Field('concate_num', 'concatenate offset').init(30,'optional','int32'));
  packsinga.addMessage(ConcateProtoObj);

  var DataProtoObj=new Message('DataProto','');
  DataProtoObj.addField(new Field('path', 'path to the data file/folder, absolute or relative to the workspace').init(2,'required','string'));
  DataProtoObj.addField(new Field('batchsize', 'batch size.').init(4,'required','int32'));
  DataProtoObj.addField(new Field('random_skip', 'skip [0,random_skip] records').init(30 ,'optional','int32','0'));
  packsinga.addMessage(DataProtoObj);

  var MnistProtoObj=new Message('MnistProto','');
  MnistProtoObj.addField(new Field('norm_a', 'normalization x/norm_a').init(1 ,'required','float','1'));
  MnistProtoObj.addField(new Field('norm_b', 'normalization x-norm_b').init(2 ,'required','float','0'));
  MnistProtoObj.addField(new Field('kernel', 'elastic distortion').init(30 ,'optional','int32','0'));
  MnistProtoObj.addField(new Field('sigma', '').init(31 ,'optional','float','0'));
  MnistProtoObj.addField(new Field('alpha', '').init(32 ,'optional','float','0'));
  MnistProtoObj.addField(new Field('beta', 'rotation or horizontal shearing').init(33 ,'optional','float','0'));
  MnistProtoObj.addField(new Field('gamma', 'scaling').init(34 ,'optional','float','0'));
  MnistProtoObj.addField(new Field('resize', 'scale to this size as input for deformation').init(35 ,'optional','int32','0'));
  MnistProtoObj.addField(new Field('elastic_freq', '').init(36 ,'optional','int32','0'));
  packsinga.addMessage(MnistProtoObj);

  var DropoutProtoObj=new Message('DropoutProto','');
  DropoutProtoObj.addField(new Field('dropout_ratio', 'dropout ratio').init(30 ,'optional','float','0.5'));
  packsinga.addMessage(DropoutProtoObj);

  var InnerProductProtoObj=new Message('InnerProductProto','');
  InnerProductProtoObj.addField(new Field('num_output', 'number of outputs for the layer').init(1,'required','int32'));
  InnerProductProtoObj.addField(new Field('bias_term', 'use bias vector or not').init(30 ,'optional','bool','true'));
  packsinga.addMessage(InnerProductProtoObj);

  var NormRegionObj=new Enum('NormRegion','');
  NormRegionObj.addValue(new EnumValue('ACROSS_CHANNELS', 0,'across channels, e.g., r,g,b'));
  NormRegionObj.addValue(new EnumValue('WITHIN_CHANNEL', 1,'within channel, e.g., r, g and b are concatenated into one channel'));
  packsinga.addEnum(NormRegionObj);

  var LRNProtoObj=new Message('LRNProto','');
  LRNProtoObj.addField(new Field('local_size', 'local response size').init(1 ,'required','int32','5'));
  LRNProtoObj.addField(new Field('alpha', 'scale factor').init(31 ,'optional','float','1.0'));
  LRNProtoObj.addField(new Field('beta', 'exponential number').init(32 ,'optional','float','0.75'));
  LRNProtoObj.addField(new Field('norm_region', 'normalization objective').init(33 ,'optional','NormRegion','ACROSS_CHANNELS'));
  LRNProtoObj.addField(new Field('knorm', 'offset').init(34 ,'optional','float','1.0'));
  packsinga.addMessage(LRNProtoObj);

  var PoolMethodObj=new Enum('PoolMethod','');
  PoolMethodObj.addValue(new EnumValue('MAX', 0,''));
  PoolMethodObj.addValue(new EnumValue('AVE', 1,''));
  packsinga.addEnum(PoolMethodObj);

  var PoolingProtoObj=new Message('PoolingProto','');
  PoolingProtoObj.addField(new Field('kernel', 'The kernel size (square)').init(1,'required','int32'));
  PoolingProtoObj.addField(new Field('pool', 'The pooling method').init(30 ,'optional','PoolMethod','MAX'));
  PoolingProtoObj.addField(new Field('pad', 'The padding size').init(31 ,'optional','uint32','0'));
  PoolingProtoObj.addField(new Field('stride', 'The stride').init(32 ,'optional','uint32','1'));
  packsinga.addMessage(PoolingProtoObj);

  var SliceProtoObj=new Message('SliceProto','');
  SliceProtoObj.addField(new Field('slice_dimension', '').init(1,'optional','int32'));
  SliceProtoObj.addField(new Field('slice_num', '').init(2,'optional','int32'));
  packsinga.addMessage(SliceProtoObj);

  var ReLUProtoObj=new Message('ReLUProto','');
  ReLUProtoObj.addField(new Field('negative_slope', 'Ref. Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013). Rectifier nonlinearities improve neural network acoustic models. In ICML Workshop on Deep Learning for Audio, Speech, and Language Processing.').init(1 ,'optional','float','0'));
  packsinga.addMessage(ReLUProtoObj);

  var UpdaterTypeObj=new Enum('UpdaterType','');
  UpdaterTypeObj.addValue(new EnumValue('kSGD', 1,'noraml SGD with momentum and weight decay'));
  UpdaterTypeObj.addValue(new EnumValue('kAdaGrad', 2,'adaptive subgradient, http://www.magicbroom.info/Papers/DuchiHaSi10.pdf'));
  UpdaterTypeObj.addValue(new EnumValue('kRMSProp', 3,'http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf'));
  UpdaterTypeObj.addValue(new EnumValue('kNesterov', 4,'Nesterov first optimal gradient method'));
  packsinga.addEnum(UpdaterTypeObj);

  var ChangeMethodObj=new Enum('ChangeMethod','');
  ChangeMethodObj.addValue(new EnumValue('kFixed', 0,''));
  ChangeMethodObj.addValue(new EnumValue('kInverseT', 1,''));
  ChangeMethodObj.addValue(new EnumValue('kInverse', 2,''));
  ChangeMethodObj.addValue(new EnumValue('kExponential', 3,''));
  ChangeMethodObj.addValue(new EnumValue('kLinear', 4,''));
  ChangeMethodObj.addValue(new EnumValue('kStep', 5,''));
  ChangeMethodObj.addValue(new EnumValue('kFixedStep', 6,''));
  packsinga.addEnum(ChangeMethodObj);

  var UpdaterProtoObj=new Message('UpdaterProto','');
  UpdaterProtoObj.addField(new Field('type', 'updater type').init(1 ,'required','UpdaterType','kSGD'));
  UpdaterProtoObj.addField(new Field('rmsprop_conf', 'configuration for RMSProp algorithm').init(50,'optional','RMSPropProto'));
  UpdaterProtoObj.addField(new Field('lr_change', 'change method for learning rate').init(2 ,'required','ChangeMethod','kFixed'));
  UpdaterProtoObj.addField(new Field('fixedstep_conf', '').init(40,'optional','FixedStepProto'));
  UpdaterProtoObj.addField(new Field('step_conf', '').init(41,'optional','StepProto'));
  UpdaterProtoObj.addField(new Field('linear_conf', '').init(42,'optional','LinearProto'));
  UpdaterProtoObj.addField(new Field('exponential_conf', '').init(43,'optional','ExponentialProto'));
  UpdaterProtoObj.addField(new Field('inverse_conf', '').init(44,'optional','InverseProto'));
  UpdaterProtoObj.addField(new Field('inverset_conf', '').init(45,'optional','InverseTProto'));
  UpdaterProtoObj.addField(new Field('momentum', '').init(31 ,'optional','float','0'));
  UpdaterProtoObj.addField(new Field('weight_decay', '').init(32 ,'optional','float','0'));
  UpdaterProtoObj.addField(new Field('base_lr', 'base learning rate').init(34 ,'optional','float','0'));
  UpdaterProtoObj.addField(new Field('delta', 'used to avoid divide by 0, i.e. x/(y+delta)').init(35 ,'optional','float','0.00000001'));
  packsinga.addMessage(UpdaterProtoObj);

  var RMSPropProtoObj=new Message('RMSPropProto','');
  RMSPropProtoObj.addField(new Field('rho', 'history=history*rho_+(1-rho_)*(grad*grad_scale)').init(1,'required','float'));
  packsinga.addMessage(RMSPropProtoObj);

  var FixedStepProtoObj=new Message('FixedStepProto','');
  FixedStepProtoObj.addField(new Field('step', '').init(28,'repeated','int32'));
  FixedStepProtoObj.addField(new Field('step_lr', 'lr = step_lr[i] if current step >= step[i]').init(29,'repeated','float'));
  packsinga.addMessage(FixedStepProtoObj);

  var StepProtoObj=new Message('StepProto','');
  StepProtoObj.addField(new Field('gamma', 'lr = base_lr * gamma^(step/change_freq)').init(35 ,'required','float','1'));
  StepProtoObj.addField(new Field('change_freq', 'lr = base_lr * gamma^(step/change_freq)').init(40,'required','int32'));
  packsinga.addMessage(StepProtoObj);

  var LinearProtoObj=new Message('LinearProto','');
  LinearProtoObj.addField(new Field('change_freq', 'lr = (1 - step / freq) * base_lr + (step / freq) * final_lr').init(40,'required','int32'));
  LinearProtoObj.addField(new Field('final_lr', 'lr = (1 - step / freq) * base_lr + (step / freq) * final_lr').init(39,'required','float'));
  packsinga.addMessage(LinearProtoObj);

  var ExponentialProtoObj=new Message('ExponentialProto','');
  ExponentialProtoObj.addField(new Field('change_freq', 'lr = base / 2^(step/change_freq)').init(40,'required','int32'));
  packsinga.addMessage(ExponentialProtoObj);

  var InverseTProtoObj=new Message('InverseTProto','');
  InverseTProtoObj.addField(new Field('final_lr', 'lr = base_lr / (1+step/final_lr)').init(39,'required','float'));
  packsinga.addMessage(InverseTProtoObj);

  var InverseProtoObj=new Message('InverseProto','');
  InverseProtoObj.addField(new Field('gamma', 'lr = base_lr*(1+gamma*step)^(-pow)').init(1 ,'required','float','1'));
  InverseProtoObj.addField(new Field('pow', 'lr = base_lr*(1+gamma*step)^(-pow)').init(2 ,'required','float','0'));
  packsinga.addMessage(InverseProtoObj);


  //end auto generate by python


  function fieldBindation(container,controlField,fieldGroup){

    var controlClass="group_"+controlField;

    $.each(fieldGroup,function(index,field){
      container.find(".field_"+field+":first").addClass("field_group "+controlClass);

    });

    container.find("."+controlClass).hide();

    container.find(".field_"+controlField+" select").on("change",function(){
      var field=$(this).find("option:selected").text();
      field=field.substr(1).toLowerCase()+"_conf";
      container.find("."+controlClass).hide();
      container.find(".field_"+field+":first").show();
    });


  }

  LayerProtoObj.additionBind=function(container){
    var fieldGroup=["convolution_conf","concate_conf","data_conf","dropout_conf","innerproduct_conf","lrn_conf","mnist_conf","pooling_conf","slice_conf","split_conf","relu_conf","rgbimage_conf","softmaxloss_conf","tanh_conf","prefetch_conf"]
    fieldBindation(container,"type",fieldGroup);

  }

  UpdaterProtoObj.additionBind=function(container){
    var me = this;
    var fieldGroup1=["adagrad_conf","rmsprop_conf"]

    var fieldGroup2=["fixedstep_conf","step_conf","linear_conf","exponential_conf","inverse_conf","inverset_conf"]

    fieldBindation(container,"type",fieldGroup1);

    fieldBindation(container,"lr_change",fieldGroup2);


  }


  packsinga.addRootMessage(ClusterProtoObj);
  packsinga.addRootMessage(ModelProtoObj);

  Model.package=packsinga;

  $("#submitForm").click(function(){
    var testmode=false;
    if(testmode){
      //test
      $.get("/scripts/cluster.conf",function(data){
        clusterConf=data;
        $.get("/scripts/model.conf",function(data2) {
          modelConf = data2;
          workspace ="dbm/";
          $.ajax({
            url:Model.Config.apiUrl,
            type:"POST",
            data:{
              action:"submit",
              cluster:clusterConf,
              model:modelConf,
              workspace:workspace
            },
            success:function(data){
              monitor();
            },
            error:function(e){
              console.log(e);
            }



          });

        });

      })


    }else{
      var clusterConf = $("#value_ClusterProto textarea").val();
      var modelConf = $("#value_ModelProto textarea").val();
      var result = /workspace:(\s*)"(.*)"\n/.exec(clusterConf);
      if(result==null){
        alert("workspace not defined!");
        return;
      }
      var workspace =result[2];
      $.ajax({
        url:Model.Config.apiUrl,
        type:"POST",
        data:{
          action:"submit",
          cluster:clusterConf,
          model:modelConf,
          workspace:workspace
        },
        success:function(data){
          monitor();
        },
        error:function(e){
          console.log(e);
        }



      });

    }

  });
  $("#monitorBtn").click(function() {
      monitor();

  });

  function monitor(){
    $(".config").hide();
    $(".monitor").show();

    $("#chart").empty();
    $("#pic").empty();

    var charts={};
    var pics={};
    var pics_src={"conv":["conv1.jpg","conv2.jpg","conv3.jpg","conv4.jpg"],"filter":["filters_at_epoch_14.png"]};

    var polling=setInterval(function(){
      $.ajax({
        url:Model.Config.apiUrl,
        type:"GET",
        data:{
          action:"polling"
        },
        success:function(data){
          var data =JSON.parse(data);
          for(var index in data){
            var item = data[index];
            if(item.type=="chart") {
              var chart = charts[item.ylabel];
              if (!chart) {
                chart = charts[item.ylabel] = new Model.ChartController({model: item});
              } else{

                chart.addData(item.data);

              }

            }else{
              var pic = pics[item.title];
              if (!pic) {
                pic = pics[item.title] = new Model.PicController({model:item.title});
              }

              pic.addData(item.url);


            }
          }


        },
        error:function(e){
          console.log(e);
        }
      });

    },2000);

    $("#killBtn").click(function(){
      $.ajax({
        url:Model.Config.apiUrl,
        type:"GET",
        data:{
          action:"kill"
        },
        success:function(data){
          console.log(data);
          clearInterval(polling);
        },
        error:function(e){
          console.log(e);
        }
      });


    });


  }


  function parseMessage(result,conf){

    if(conf==null){
      return;
    }
    conf=conf.trim();

    while(true ) {
      var nextConf = getNextConf();
      if (nextConf == null) {
        return result;
      } else {
        if (nextConf.type == "message") {
          var value = {};
          parseMessage(value, nextConf.value);
          addField({key: nextConf.key, value: value});
        } else {
          addField(nextConf);
        }
      }
    }
    function addField(field){
      var oldValue=result[field.key];
      if(oldValue){
        if(oldValue.push){
          oldValue.push(field.value);
        }else{
          result[field.key]=[oldValue,field.value];
        }
      }else{
        result[field.key]=field.value;
      }
    }

    function getNextConf(){
      conf=conf.trim();

      var index1=conf.indexOf(":");
      var index2=conf.indexOf("{");
      if(index1!=-1&&(index2==-1||index1<index2)){

        var key =conf.substr(0,index1).trim();
        var indexEnd= conf.indexOf("\n");
        if(indexEnd==-1){
          indexEnd=conf.length;
        }
        var value=conf.substr(index1+1,indexEnd-index1-1).trim();
        if(value.indexOf("\"")>-1){
          value=value.substr(1,value.length-2);
        }

        conf=conf.substr(indexEnd+1);
        return {key:key,value:value,type:"field"};
      }
      if(index2!=-1&&(index1==-1||index2<index1)) {

        var key =conf.substr(0,index2).trim();
        var num=1;
        for(var indexEnd=index2+1;indexEnd<conf.length;indexEnd++){
          if(conf.charAt(indexEnd)=="{"){
            num++;
          }
          if(conf.charAt(indexEnd)=="}"){
            num--;
          }
          if(num==0){
            break;
          }
        }
        var value=conf.substr(index2+1,indexEnd-index2-1);
        conf=conf.substr(indexEnd+2);
        return {key:key,value:value,type:"message"};

      }

      return null;

    }
  }
})();
