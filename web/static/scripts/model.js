/**
 * Created by Aaron on 6/20/2015.
 */
"use strict";

(function(){
  window.Model={};

  Model.Config={
    unimportId:60,  //if field id is bigger than this, then not show
    host:"/",
    apiUrl:"/api"
  }

  var Package =function(name,comment){
    this.name=name;
    this.comment=comment;
    this.messages={};
    this.enums={};
    this.rootMessage;
  }

  var MessageDef =function(name,comment){
    this.name=name;
    this.comment=comment;
    this.fields={};
    this.constrains=[];
  }

  MessageDef.prototype={
    addConstrain:function(field,changeFunction){
      var me = this;
      var constrain= {
        field:field,
        changeFunction:changeFunction
      }
      var fieldDef=this.fields[field];
      var enumDef = Model.package.enums[fieldDef.type];

      var fieldGroup=[];
      $.each(enumDef.values,function(index,enumValue){
        var fieldName= changeFunction(enumValue.key);
        var targetField = me.fields[fieldName];
        if(targetField){
          fieldGroup.push(fieldName);
        }
      });
      constrain.fieldGroup=fieldGroup;
      this.constrains.push(constrain);

    }

  }
  var Message = function(def){
    var me = this;
    me.id=Message.nextId();
    me.def=def;
    me.fields=[];

    $.each(def.fields,function(name,fieldDef){
      var field=new Field(fieldDef);
      me.addField(field);
    });
  }
  Message.IdSequence=0;
  Message.nextId=function(){
    return Message.IdSequence++;
  }

  Message.prototype={
    init:function(){

      this.setConstrainFields();
    },
    initFromConf:function(confObj){
      var me = this;
      if(me.$dom){
        me.$dom.empty();
      }
      me.fields=[];
      $.each(this.def.fields,function(name,fieldDef){
        var value= confObj[name];

        if(!value){
          var field=new Field(fieldDef);
          me.addField(field);
        }else if(typeof(value)=="object"&&value.length){  //value is array
          $.each(value,function(index,v){
            var field = new Field(fieldDef);
            me.addField(field);
            field.initFromConf(v);
          });
        }else{
          var field = new Field(fieldDef);
          me.addField(field);
          field.initFromConf(value);

        }
      });
      me.sortField();
      me.setConstrainFields();

    },
    setConstrainFields:function(){
      var me = this;
      me.constrains={};
      $.each(me.def.constrains,function(index,constrain){
        var fieldGroup=[];
        $.each(constrain.fieldGroup,function(i,fieldName){
          var field = me.getFieldByName(fieldName);
          fieldGroup.push(field);
        })
        me.constrains[constrain.field]={
          fieldGroup:fieldGroup,
          changeFunction:constrain.changeFunction
        };
      });

    },
    setContainer:function($container){
      this.$container=$container;
    },
    render:function(){
      var me = this;
      var source = $("#messageTemplate").html();
      var template = Handlebars.compile(source);
      var html= template(this);
      me.$dom=$(html).appendTo(me.$container);
      me.$dom.data("data",this);

      me.$fieldContainer=me.$dom.find(".detail");

      //renderFields
      $.each(me.fields,function(index,field){
        field.render(me.$fieldContainer);
      });

      me.hideConstrainFields();

      this.bind();
    },
    bind:function(){
      var me = this;

      me.$dom.find("[data-toggle='tooltip']").tooltip();

      this.$dom.find(".edit-btn").bind("click",function(){
        $(this).hide();
        me.$dom.find(".done-btn").show();
        $.each(me.fields,function(index,field){
          field.edit();
        });
      })
      this.$dom.find(".done-btn").bind("click",function(){
        $(this).hide();
        me.$dom.find(".edit-btn").show();

        $.each(me.fields,function(index,field){
          field.done();
        });
        Model.showConf();
      });
      this.$dom.find(".toggle-btn").bind("click",function(){
        if($(this).hasClass("glyphicon-chevron-up")){
          $(this).removeClass("glyphicon-chevron-up");
          $(this).addClass("glyphicon-chevron-down");
          $.each(me.fields,function(index,field){
            field.isUnimportant()?field.show():"";
          });
        }else{
          $(this).removeClass("glyphicon-chevron-down");
          $(this).addClass("glyphicon-chevron-up");
          $.each(me.fields,function(index,field){
            field.isUnimportant()?field.hide():"";
          });

        }
      })

    },
    show:function(){
      if(!this.$dom){
        this.render();
      }
      this.$dom.show();
    },
    sortField:function(){
      this.fields.sort(compareField);
    },
    addField:function(field){
      this.fields.push(field);
      field.parent=this;

    },
    removeField:function(field){
      this.fields.splice(this.fields.indexOf(field),1);
    },
    destroy:function(){
      this.$dom.remove();
      delete this;
    },
    clone:function(){
      var newMessage = new Message(this.def);
      newMessage.setContainer(this.$container);

      newMessage.fields=[];
      $.each(this.fields,function(id,field){
        var newField = field.clone();
        newMessage.addField(newField);
      });
      newMessage.sortField();
      newMessage.init();
      return newMessage;
    },
    getFieldByName:function(name){
      return this.fields.filter(function(a){return a.def.name==name;})[0];
    },
    checkConstrain:function(field){
      var me = this;
      $.each(this.constrains,function(name,constrain){
        if(field.def.name==name){
          var value = field.value;
          var targetField=me.getFieldByName(constrain.changeFunction(value));

          $.each(constrain.fieldGroup,function(index,field){
            field.hide();
          })
          if(targetField){
            targetField.show()
          }
        }
      });

    },
    hideConstrainFields:function(){
      var me = this;
      $.each(me.constrains,function(name,constrain){
        $.each(constrain.fieldGroup,function(i,field){
          field.hide();
        })
      });
    },
    toConf:function(prefix){
      var conf="",
       me=this;
      $.each(me.fields,function(index,field){
        conf+=field.toConf(prefix);
      });
      return conf;
    }
  };

  var FieldDef = function(id,name,rule,type,comment,_default){
    this.name=name;
    this.comment=comment;
    this.id=id;
    this.type=type;
    this.rule=rule;
    this._default=_default;
  }

  var Field = function(def){
    var me = this;
    me.id=Field.nextId();
    me.def=def;
    if(def._default) {
      me.value = def._default;
    }
  }
  Field.IdSequence=0;
  Field.nextId=function(){
    return Field.IdSequence++;
  }

  Field.prototype={
    isRepeated:function(){
      if(this.def.rule=="repeated"){
        return true;
      }else{
        return false;
      }

    },
    isRequired:function(){
      if(this.def.rule=="required"){
        return true;
      }else{
        return false;
      }
    },
    isEnum:function(){
      if(this.enumDef!=undefined){
        if(this.enumDef==null){
          return false;
        }else{
          return true;
        }
      }
      this.enumDef=Model.package.enums[this.def.type];
      if(this.enumDef){
        return true;
      }else{
        this.enumDef=null;
        return false;
      }
    },
    isMessage:function(){
      if(this.messageDef!=undefined){
        if(this.messageDef==null){
          return false;
        }else{
          return true;
        }
      }
      this.messageDef=Model.package.messages[this.def.type];
      if(this.messageDef){
        return true;
      }else{
        this.messageDef=null;
        return false;
      }
    },
    isUnimportant:function(){
      if(this.def.id>Model.Config.unimportId){
        return true;
      }else{
        return false;
      }
    },
    initFromConf:function(confObj){
      if(this.isMessage()){
        var messageDef= Model.package.messages[this.def.type];
        this.message=  new Message(messageDef);
        this.message.setContainer(this.parent.$container.next());
        this.message.initFromConf(confObj);
      }else {
        this.value=confObj;
      }
    },
    render:function(){
      var me = this;
      me.$container=this.parent.$fieldContainer;

      var source = $("#fieldTemplate").html();
      var template = Handlebars.compile(source);
      var html= template(this);
      me.$dom=$(html).appendTo(me.$container);

      //hide unimportant
      if(me.isUnimportant()){
        me.$dom.hide();
      }
      me.$dom.data("data",this);
      me.bind();
    },
    bind:function(){
      var me = this;

      me.$dom.find(".show-btn").bind("click",function(){
        me.parent.$container.nextAll().find(".message").hide();
        me.$dom.siblings().removeClass("selected");
        me.$dom.addClass("selected");
        if(me.message){
          me.message.show();
        }else{
          var messageDef= Model.package.messages[me.def.type];
          me.message=  new Message(messageDef);
          me.message.init();
          me.message.setContainer(me.parent.$container.next());
          me.message.render();
        }
      });

      me.$dom.find(".plus-btn").bind("click",function(){
        var newField = me.clone();
        me.parent.addField(newField);
        me.parent.sortField();
        newField.render();
        newField.$dom.insertAfter(me.$dom);
        newField.show();
        newField.$dom.find(".show-btn").trigger("click");
        Model.showConf();
      });
      me.$dom.find(".minus-btn").bind("click",function(){
        var $sameField=me.$container.find(".field_"+me.def.name);
        if($sameField.length==1){
          me.empty();
        }else{
          me.destroy();
        }
        Model.showConf();
      });



    },
    edit:function(){
      var me = this;
      if(this.isEnum()){
        me.renderEnum();

      }else if(this.isMessage()){
       // me.renderMessage();
      }else{
        me.renderText();
      }

    },
    done:function(){
      if(this.isMessage()){

      }else if(this.isEnum()){
        var newValue = this.$dom.find("select").val();
        this.value=newValue;
        this.$dom.find(".value").html(this.value?this.value:"");

      }else {

        var newValue = this.$dom.find("input").val();
        this.value=newValue;
        this.$dom.find(".value").html(this.value?this.value:"");
      }
    },
    renderText:function(){
      var source = $("#textTemplate").html();
      var template = Handlebars.compile(source);
      var html= template(this);
      this.$dom.find(".value").html(html);

    },
    renderEnum:function(){
      var me = this;
      var source = $("#selectTemplate").html();
      var template = Handlebars.compile(source);
      var html= template($.extend({"_enum":this.enumDef},this));
      me.$dom.find(".value").html(html);
      me.$dom.find("select")
        .val(this.value)
        .change(function(){
          me.value=$(this).val();
          me.parent.checkConstrain(me);
      });

    },
    show:function(){
      this.$dom.show();
    },
    hide:function(){
      this.$dom.hide();
    },
    empty:function(){
      if(this.isMessage()){
        if(this.message){
          this.message.destroy();
          delete this.message;
        }

      }else{
        delete this.value;
      }

    },
    destroy:function(){
      this.parent.removeField();
      this.$dom.remove();
      delete this;
    },
    clone:function(){
      var newField = new Field(this.def);
      if(this.isMessage()){
        if(this.message){
          newField.message = this.message.clone();
          newField.message.render(this.message.$container);
        }
      }else{
        newField.value=this.value;
      }
      return newField;
    },
    toConf:function(prefix){

      if(this.isMessage()){
        if(this.message){
          return prefix+this.def.name+" { \n"+this.message.toConf(prefix+"  ")+prefix+"}\n";
        }else{
          return "";
        }
      }else if(this.value&&(this.def.type=="required"||this.value!=this.def._default)){
        if(this.isEnum()){
          return prefix+this.def.name+": "+this.value+"\n";
        }else if(this.def.type=="string"){
          return prefix+this.def.name+": \""+this.value+"\"\n";
        }else{
          return prefix+this.def.name+": "+this.value+"\n";
        }
      }else{
        return "";
      }
    }

  }
  var EnumDef = function(name,comment){

    this.name=name;
    this.comment=comment;
    this.values={};
  }

  var EnumValueDef = function(key,value,comment){
    this.key=key;
    this.value=value;
    this.comment=comment;
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
  function compareField(a,b){
    if(a.def.id> b.def.id){
      return 1;
    }else if(a.def.id< b.def.id){
      return -1;
    }else if(a.id > b.id){
      return 1;
    }else if(a.id < b.id){
      return -1;
    }else{
      return 0;
    }

  }



  //begin auto generate
  var packsinga=new Package('singa');
  var JobProto=new MessageDef('JobProto','');
  JobProto.fields['cluster'] = new FieldDef(1,'cluster','required','ClusterProto','');
  JobProto.fields['model'] = new FieldDef(2,'model','required','ModelProto','');
  packsinga.messages['JobProto']=JobProto;

  var ClusterProto=new MessageDef('ClusterProto','');
  ClusterProto.fields['nworker_groups'] = new FieldDef(1,'nworker_groups','optional','int32','');
  ClusterProto.fields['nserver_groups'] = new FieldDef(2,'nserver_groups','optional','int32','');
  ClusterProto.fields['nworkers_per_group'] = new FieldDef(3 ,'nworkers_per_group','optional','int32','','1');
  ClusterProto.fields['nservers_per_group'] = new FieldDef(4 ,'nservers_per_group','optional','int32','','1');
  ClusterProto.fields['nworkers_per_procs'] = new FieldDef(5 ,'nworkers_per_procs','optional','int32','','1');
  ClusterProto.fields['nservers_per_procs'] = new FieldDef(6 ,'nservers_per_procs','optional','int32','','1');
  ClusterProto.fields['server_worker_separate'] = new FieldDef(11 ,'server_worker_separate','optional','bool','servers and workers in different processes?','false');
  ClusterProto.fields['bandwidth'] = new FieldDef(61 ,'bandwidth','optional','int32','bandwidth of ethernet, Bytes per second, default is 1 Gbps','134217728');
  ClusterProto.fields['poll_time'] = new FieldDef(62 ,'poll_time','optional','int32','poll time in milliseconds','100');
  ClusterProto.fields['workspace'] = new FieldDef(63 ,'workspace','optional','string','workspace for checkpoint/vis folder. Overwrite cmd "workspace" if set','workspace');
  ClusterProto.fields['start_port'] = new FieldDef(64 ,'start_port','optional','int32','port number is used by ZeroMQ','6723');
  ClusterProto.fields['server_update'] = new FieldDef(65 ,'server_update','optional','bool','conduct updates at server side; otherwise do it at worker side','true');
  ClusterProto.fields['share_memory'] = new FieldDef(66 ,'share_memory','optional','bool','share memory space between worker groups in one procs','true');
  packsinga.messages['ClusterProto']=ClusterProto;

  var Phase=new EnumDef('Phase','');
  Phase.values['kTrain']=new EnumValueDef('kTrain', 0,'');
  Phase.values['kValidation']=new EnumValueDef('kValidation', 1,'');
  Phase.values['kTest']=new EnumValueDef('kTest', 2,'');
  Phase.values['kPositive']=new EnumValueDef('kPositive', 3,'postivie phase for contrastive divergence algorithm');
  Phase.values['kNegative']=new EnumValueDef('kNegative', 4,'negative phase for contrastive divergence algorithm');
  Phase.values['kForward']=new EnumValueDef('kForward', 5,'');
  Phase.values['kBackward']=new EnumValueDef('kBackward', 6,'');
  Phase.values['kLoss']=new EnumValueDef('kLoss', 7,'');
  packsinga.enums['Phase']=Phase;

  var GradCalcAlg=new EnumDef('GradCalcAlg','');
  GradCalcAlg.values['kBackPropagation']=new EnumValueDef('kBackPropagation', 1,'BP algorithm for feed-forward models, e.g., CNN, MLP, RNN');
  GradCalcAlg.values['kContrastiveDivergence']=new EnumValueDef('kContrastiveDivergence', 2,'CD algorithm for RBM, DBM etc., models');
  packsinga.enums['GradCalcAlg']=GradCalcAlg;

  var ModelProto=new MessageDef('ModelProto','');
  ModelProto.fields['name'] = new FieldDef(1,'name','required','string','model name, e.g., "cifar10-dcnn", "mnist-mlp"');
  ModelProto.fields['display_frequency'] = new FieldDef(3,'display_frequency','required','int32','frequency of displaying training info');
  ModelProto.fields['train_steps'] = new FieldDef(5,'train_steps','required','int32','total num of steps for training');
  ModelProto.fields['updater'] = new FieldDef(7,'updater','required','UpdaterProto','configuration of SGD updater, including learning rate, etc.');
  ModelProto.fields['alg'] = new FieldDef(8 ,'alg','required','GradCalcAlg','gradient calculation algorithm','kBackPropagation');
  ModelProto.fields['neuralnet'] = new FieldDef(9,'neuralnet','required','NetProto','');
  ModelProto.fields['validation_steps'] = new FieldDef(30 ,'validation_steps','optional','int32','total num of steps for validation','0');
  ModelProto.fields['test_steps'] = new FieldDef(31 ,'test_steps','optional','int32','total num of steps for test','0');
  ModelProto.fields['validation_frequency'] = new FieldDef(32,'validation_frequency','optional','int32','frequency of validation');
  ModelProto.fields['test_frequency'] = new FieldDef(33 ,'test_frequency','optional','int32','frequency of test','0');
  ModelProto.fields['checkpoint_frequency'] = new FieldDef(34 ,'checkpoint_frequency','optional','int32','frequency of checkpoint','0');
  ModelProto.fields['warmup_steps'] = new FieldDef(35 ,'warmup_steps','optional','int32','send parameters to servers after training for this num of steps','0');
  ModelProto.fields['resume'] = new FieldDef(36 ,'resume','optional','bool','checkpoint path','false');
  ModelProto.fields['vis_frequency'] = new FieldDef(37 ,'vis_frequency','optional','int32','visualization frequency','0');
  ModelProto.fields['vis_samples'] = new FieldDef(38 ,'vis_samples','optional','int32','num of feature records to extract','0');
  ModelProto.fields['display_after'] = new FieldDef(60,'display_after','optional','int32','start display after this num steps','0');
  ModelProto.fields['checkpoint_after'] = new FieldDef(61 ,'checkpoint_after','optional','int32','start checkpoint after this num steps','0');
  ModelProto.fields['test_after'] = new FieldDef(62 ,'test_after','optional','int32','start test after this num steps','0');
  ModelProto.fields['validation_after'] = new FieldDef(63 ,'validation_after','optional','int32','start validation after this num steps','0');
  ModelProto.fields['vis_after'] = new FieldDef(64 ,'vis_after','optional','int32','start vis after this num steps','0');
  ModelProto.fields['step'] = new FieldDef(65 ,'step','optional','int32','last snapshot step','0');
  ModelProto.fields['debug'] = new FieldDef(66 ,'debug','optional','bool','display debug info','false');
  ModelProto.fields['checkpoint'] = new FieldDef(67,'checkpoint','repeated','string','checkpoint files');
  ModelProto.fields['reset_param_version'] = new FieldDef(68 ,'reset_param_version','optional','bool','reset the version of params loaded from checkpoint file to step','false');
  ModelProto.fields['pcd_k'] = new FieldDef(69 ,'pcd_k','optional','int32','number of steps for gibbs sampling','15');
  packsinga.messages['ModelProto']=ModelProto;

  var NetProto=new MessageDef('NetProto','');
  NetProto.fields['layer'] = new FieldDef(1,'layer','repeated','LayerProto','');
  NetProto.fields['partition_dim'] = new FieldDef(2 ,'partition_dim','optional','int32','partitioning type for parallelism','0');
  packsinga.messages['NetProto']=NetProto;

  var InitMethod=new EnumDef('InitMethod','');
  InitMethod.values['kConstant']=new EnumValueDef('kConstant', 0,'fix the values of all parameters  a constant in the value field');
  InitMethod.values['kGaussian']=new EnumValueDef('kGaussian', 1,'sample gaussian with std and mean');
  InitMethod.values['kUniform']=new EnumValueDef('kUniform', 2,'uniform sampling between low and high');
  InitMethod.values['kPretrained']=new EnumValueDef('kPretrained', 3,'copy the content and history which are from previous training');
  InitMethod.values['kGaussainSqrtFanIn']=new EnumValueDef('kGaussainSqrtFanIn', 4,'from Toronto Convnet, let a=1/sqrt(fan_in), w*=a after generating from Gaussian distribution');
  InitMethod.values['kUniformSqrtFanIn']=new EnumValueDef('kUniformSqrtFanIn', 5,'from Toronto Convnet, rectified linear activation, let a=sqrt(3)/sqrt(fan_in), range is [-a, +a]; no need to set value=sqrt(3), the program will multiply it.');
  InitMethod.values['kUniformSqrtFanInOut']=new EnumValueDef('kUniformSqrtFanInOut', 6,'from Theano MLP tutorial, let a=sqrt(6/(fan_in+fan_out)). for tanh activation, range is [-a, +a], for sigmoid activation, range is [-4a, +4a], put the scale factor to value field. <a href="http://deeplearning.net/tutorial/mlp.html"> Theano MLP</a>');
  packsinga.enums['InitMethod']=InitMethod;

  var ParamProto=new MessageDef('ParamProto','');
  ParamProto.fields['init_method'] = new FieldDef(1 ,'init_method','optional','InitMethod','','kGaussian');
  ParamProto.fields['value'] = new FieldDef(5 ,'value','optional','float','constant init','1');
  ParamProto.fields['low'] = new FieldDef(6 ,'low','optional','float','for uniform sampling','-1');
  ParamProto.fields['high'] = new FieldDef(7 ,'high','optional','float','','1');
  ParamProto.fields['mean'] = new FieldDef(8 ,'mean','optional','float','for gaussian sampling','0');
  ParamProto.fields['std'] = new FieldDef(9 ,'std','optional','float','','1');
  ParamProto.fields['learning_rate_multiplier'] = new FieldDef(15 ,'learning_rate_multiplier','optional','float','multiplied on the global learning rate.','1');
  ParamProto.fields['weight_decay_multiplier'] = new FieldDef(16 ,'weight_decay_multiplier','optional','float','multiplied on the global weight decay.','1');
  ParamProto.fields['vis'] = new FieldDef(17 ,'vis','optional','bool','dump this param for visualization','false');
  ParamProto.fields['partition_dim'] = new FieldDef(30,'partition_dim','optional','int32','partition dimension, -1 for no partition');
  ParamProto.fields['shape'] = new FieldDef(31,'shape','repeated','int32','usually, the program will infer the param shape');
  ParamProto.fields['name'] = new FieldDef(61 ,'name','optional','string','used for identifying the same params from diff models and display deug info','');
  ParamProto.fields['share_from'] = new FieldDef(62,'share_from','optional','string','name of the owner param from which this param shares the values');
  ParamProto.fields['id'] = new FieldDef(63,'id','optional','int32','used interally');
  ParamProto.fields['split_threshold'] = new FieldDef(64 ,'split_threshold','optional','int32','parameter slice limit (Google Protobuf also has size limit)','5000000');
  ParamProto.fields['owner'] = new FieldDef(65 ,'owner','optional','int32','used internally','-1');
  packsinga.messages['ParamProto']=ParamProto;

  var PartitionType=new EnumDef('PartitionType','');
  PartitionType.values['kDataPartition']=new EnumValueDef('kDataPartition',0,'');
  PartitionType.values['kLayerPartition']=new EnumValueDef('kLayerPartition',1,'');
  PartitionType.values['kNone']=new EnumValueDef('kNone',2,'');
  packsinga.enums['PartitionType']=PartitionType;

  var LayerType=new EnumDef('LayerType','');
  LayerType.values['kBridgeSrc']=new EnumValueDef('kBridgeSrc', 15,'');
  LayerType.values['kBridgeDst']=new EnumValueDef('kBridgeDst', 16,'');
  LayerType.values['kConvolution']=new EnumValueDef('kConvolution', 1,'');
  LayerType.values['kConcate']=new EnumValueDef('kConcate', 2,'');
  LayerType.values['kShardData']=new EnumValueDef('kShardData', 3,'');
  LayerType.values['kDropout']=new EnumValueDef('kDropout', 4,'');
  LayerType.values['kInnerProduct']=new EnumValueDef('kInnerProduct', 5,'');
  LayerType.values['kLabel']=new EnumValueDef('kLabel', 18,'');
  LayerType.values['kLMDBData']=new EnumValueDef('kLMDBData', 17,'');
  LayerType.values['kLRN']=new EnumValueDef('kLRN', 6,'');
  LayerType.values['kMnist']=new EnumValueDef('kMnist', 7,'');
  LayerType.values['kPooling']=new EnumValueDef('kPooling', 8,'');
  LayerType.values['kPrefetch']=new EnumValueDef('kPrefetch', 19,'');
  LayerType.values['kReLU']=new EnumValueDef('kReLU', 9,'');
  LayerType.values['kRGBImage']=new EnumValueDef('kRGBImage', 10,'');
  LayerType.values['kSoftmaxLoss']=new EnumValueDef('kSoftmaxLoss', 11,'');
  LayerType.values['kSlice']=new EnumValueDef('kSlice', 12,'');
  LayerType.values['kSplit']=new EnumValueDef('kSplit', 13,'');
  LayerType.values['kTanh']=new EnumValueDef('kTanh', 14,'');
  LayerType.values['kRBMVis']=new EnumValueDef('kRBMVis', 23,'');
  LayerType.values['kRBMHid']=new EnumValueDef('kRBMHid', 24,'');
  packsinga.enums['LayerType']=LayerType;

  var LayerProto=new MessageDef('LayerProto','');
  LayerProto.fields['name'] = new FieldDef(1,'name','required','string','the layer name used for identification');
  LayerProto.fields['srclayers'] = new FieldDef(3,'srclayers','repeated','string','source layer names');
  LayerProto.fields['param'] = new FieldDef(12,'param','repeated','ParamProto','parameters, e.g., weight matrix or bias vector');
  LayerProto.fields['exclude'] = new FieldDef(15,'exclude','repeated','Phase','all layers are included in the net structure for training phase by default. some layers like data layer for loading test data are not used by training phase should be removed by setting the exclude field.');
  LayerProto.fields['vis'] = new FieldDef(16 ,'vis','optional','bool','dump feature of this layer for visualization','false');
  LayerProto.fields['type'] = new FieldDef(20,'type','required','LayerType','the layer type from the enum above');
  LayerProto.fields['convolution_conf'] = new FieldDef(30,'convolution_conf','optional','ConvolutionProto','configuration for convolution layer');
  LayerProto.fields['concate_conf'] = new FieldDef(31,'concate_conf','optional','ConcateProto','configuration for concatenation layer');
  LayerProto.fields['dropout_conf'] = new FieldDef(33,'dropout_conf','optional','DropoutProto','configuration for dropout layer');
  LayerProto.fields['innerproduct_conf'] = new FieldDef(34,'innerproduct_conf','optional','InnerProductProto','configuration for inner product layer');
  LayerProto.fields['lmdbdata_conf'] = new FieldDef(35,'lmdbdata_conf','optional','DataProto','configuration for local response normalization layer');
  LayerProto.fields['lrn_conf'] = new FieldDef(45,'lrn_conf','optional','LRNProto','configuration for local response normalization layer');
  LayerProto.fields['mnist_conf'] = new FieldDef(36,'mnist_conf','optional','MnistProto','configuration for mnist parser layer');
  LayerProto.fields['pooling_conf'] = new FieldDef(37,'pooling_conf','optional','PoolingProto','configuration for pooling layer');
  LayerProto.fields['prefetch_conf'] = new FieldDef(44,'prefetch_conf','optional','PrefetchProto','configuration for prefetch layer');
  LayerProto.fields['relu_conf'] = new FieldDef(38,'relu_conf','optional','ReLUProto','configuration for rectified linear unit layer');
  LayerProto.fields['rgbimage_conf'] = new FieldDef(39,'rgbimage_conf','optional','RGBImageProto','configuration for rgb image parser layer');
  LayerProto.fields['sharddata_conf'] = new FieldDef(32,'sharddata_conf','optional','DataProto','configuration for data layer');
  LayerProto.fields['slice_conf'] = new FieldDef(41,'slice_conf','optional','SliceProto','configuration for slice layer');
  LayerProto.fields['softmaxloss_conf'] = new FieldDef(40,'softmaxloss_conf','optional','SoftmaxLossProto','configuration for softmax loss layer');
  LayerProto.fields['split_conf'] = new FieldDef(42,'split_conf','optional','SplitProto','configuration for split layer');
  LayerProto.fields['tanh_conf'] = new FieldDef(43,'tanh_conf','optional','TanhProto','configuration for tanh layer');
  LayerProto.fields['rbmvis_conf'] = new FieldDef(48,'rbmvis_conf','optional','RBMVisProto','configuration for rbmvis layer');
  LayerProto.fields['rbmhid_conf'] = new FieldDef(49,'rbmhid_conf','optional','RBMHidProto','configuration for rbmhid layer');
  LayerProto.fields['partition_dim'] = new FieldDef(59 ,'partition_dim','optional','int32','overrides the partition dimension for neural net','-1');
  LayerProto.fields['datablob'] = new FieldDef(58 ,'datablob','optional','string','','unknow');
  LayerProto.fields['share_param'] = new FieldDef(60,'share_param','repeated','string','names of parameters shared from other layers');
  LayerProto.fields['partition_id'] = new FieldDef(62 ,'partition_id','optional','int32','','0');
  packsinga.messages['LayerProto']=LayerProto;

  var RGBImageProto=new MessageDef('RGBImageProto','');
  RGBImageProto.fields['scale'] = new FieldDef(1 ,'scale','optional','float','scale factor for each pixel','1.0');
  RGBImageProto.fields['cropsize'] = new FieldDef(2 ,'cropsize','optional','int32','size after cropping','0');
  RGBImageProto.fields['mirror'] = new FieldDef(3 ,'mirror','optional','bool','mirror the image','false');
  RGBImageProto.fields['meanfile'] = new FieldDef(4 ,'meanfile','optional','string','meanfile path','');
  packsinga.messages['RGBImageProto']=RGBImageProto;

  var PrefetchProto=new MessageDef('PrefetchProto','');
  PrefetchProto.fields['sublayers'] = new FieldDef(1,'sublayers','repeated','LayerProto','');
  packsinga.messages['PrefetchProto']=PrefetchProto;

  var SplitProto=new MessageDef('SplitProto','');
  SplitProto.fields['num_splits'] = new FieldDef(1 ,'num_splits','optional','int32','','1');
  packsinga.messages['SplitProto']=SplitProto;

  var TanhProto=new MessageDef('TanhProto','');
  TanhProto.fields['outer_scale'] = new FieldDef(1 ,'outer_scale','optional','float','A of A*tan(B*x)','1.0');
  TanhProto.fields['inner_scale'] = new FieldDef(2 ,'inner_scale','optional','float','B of A*tan(B*x)','1.0');
  packsinga.messages['TanhProto']=TanhProto;

  var SoftmaxLossProto=new MessageDef('SoftmaxLossProto','');
  SoftmaxLossProto.fields['topk'] = new FieldDef(1 ,'topk','optional','int32','computing accuracy against topk results','1');
  SoftmaxLossProto.fields['scale'] = new FieldDef(30 ,'scale','optional','float','loss scale factor','1');
  packsinga.messages['SoftmaxLossProto']=SoftmaxLossProto;

  var ConvolutionProto=new MessageDef('ConvolutionProto','');
  ConvolutionProto.fields['num_filters'] = new FieldDef(1,'num_filters','required','int32','The number of outputs for the layer');
  ConvolutionProto.fields['kernel'] = new FieldDef(2,'kernel','required','int32','the kernel height/width');
  ConvolutionProto.fields['pad'] = new FieldDef(30 ,'pad','optional','int32','The padding height/width','0');
  ConvolutionProto.fields['stride'] = new FieldDef(31 ,'stride','optional','int32','the stride','1');
  ConvolutionProto.fields['bias_term'] = new FieldDef(32 ,'bias_term','optional','bool','whether to have bias terms','true');
  packsinga.messages['ConvolutionProto']=ConvolutionProto;

  var ConcateProto=new MessageDef('ConcateProto','');
  ConcateProto.fields['concate_dim'] = new FieldDef(1,'concate_dim','required','int32','on which dimension, starts from 0');
  packsinga.messages['ConcateProto']=ConcateProto;

  var DataProto=new MessageDef('DataProto','');
  DataProto.fields['path'] = new FieldDef(2,'path','required','string','path to the data file/folder, absolute or relative to the workspace');
  DataProto.fields['batchsize'] = new FieldDef(4,'batchsize','required','int32','batch size.');
  DataProto.fields['random_skip'] = new FieldDef(30 ,'random_skip','optional','int32','skip [0,random_skip] records','0');
  packsinga.messages['DataProto']=DataProto;

  var MnistProto=new MessageDef('MnistProto','');
  MnistProto.fields['norm_a'] = new FieldDef(1 ,'norm_a','required','float','normalization x/norm_a','1');
  MnistProto.fields['norm_b'] = new FieldDef(2 ,'norm_b','required','float','normalization x-norm_b','0');
  MnistProto.fields['kernel'] = new FieldDef(30 ,'kernel','optional','int32','elastic distortion','0');
  MnistProto.fields['sigma'] = new FieldDef(31 ,'sigma','optional','float','','0');
  MnistProto.fields['alpha'] = new FieldDef(32 ,'alpha','optional','float','','0');
  MnistProto.fields['beta'] = new FieldDef(33 ,'beta','optional','float','rotation or horizontal shearing','0');
  MnistProto.fields['gamma'] = new FieldDef(34 ,'gamma','optional','float','scaling','0');
  MnistProto.fields['resize'] = new FieldDef(35 ,'resize','optional','int32','scale to this size as input for deformation','0');
  MnistProto.fields['elastic_freq'] = new FieldDef(36 ,'elastic_freq','optional','int32','','0');
  packsinga.messages['MnistProto']=MnistProto;

  var DropoutProto=new MessageDef('DropoutProto','');
  DropoutProto.fields['dropout_ratio'] = new FieldDef(30 ,'dropout_ratio','optional','float','dropout ratio','0.5');
  packsinga.messages['DropoutProto']=DropoutProto;

  var RBMVisProto=new MessageDef('RBMVisProto','');
  RBMVisProto.fields['num_output'] = new FieldDef(1,'num_output','optional','int32','');
  RBMVisProto.fields['bias_term'] = new FieldDef(2 ,'bias_term','optional','bool','','true');
  packsinga.messages['RBMVisProto']=RBMVisProto;

  var RBMHidProto=new MessageDef('RBMHidProto','');
  RBMHidProto.fields['hid_dim'] = new FieldDef(1,'hid_dim','optional','int32','');
  RBMHidProto.fields['bias_term'] = new FieldDef(2 ,'bias_term','optional','bool','','true');
  packsinga.messages['RBMHidProto']=RBMHidProto;

  var InnerProductProto=new MessageDef('InnerProductProto','');
  InnerProductProto.fields['num_output'] = new FieldDef(1,'num_output','required','int32','number of outputs for the layer');
  InnerProductProto.fields['bias_term'] = new FieldDef(30 ,'bias_term','optional','bool','use bias vector or not','true');
  packsinga.messages['InnerProductProto']=InnerProductProto;

  var NormRegion=new EnumDef('NormRegion','');
  NormRegion.values['ACROSS_CHANNELS']=new EnumValueDef('ACROSS_CHANNELS', 0,'across channels, e.g., r,g,b');
  NormRegion.values['WITHIN_CHANNEL']=new EnumValueDef('WITHIN_CHANNEL', 1,'within channel, e.g., r, g and b are concatenated into one channel');
  packsinga.enums['NormRegion']=NormRegion;

  var LRNProto=new MessageDef('LRNProto','');
  LRNProto.fields['local_size'] = new FieldDef(1 ,'local_size','required','int32','local response size','5');
  LRNProto.fields['alpha'] = new FieldDef(31 ,'alpha','optional','float','scale factor','1.0');
  LRNProto.fields['beta'] = new FieldDef(32 ,'beta','optional','float','exponential number','0.75');
  LRNProto.fields['norm_region'] = new FieldDef(33 ,'norm_region','optional','NormRegion','normalization objective','ACROSS_CHANNELS');
  LRNProto.fields['knorm'] = new FieldDef(34 ,'knorm','optional','float','offset','1.0');
  packsinga.messages['LRNProto']=LRNProto;

  var PoolMethod=new EnumDef('PoolMethod','');
  PoolMethod.values['MAX']=new EnumValueDef('MAX', 0,'');
  PoolMethod.values['AVE']=new EnumValueDef('AVE', 1,'');
  packsinga.enums['PoolMethod']=PoolMethod;

  var PoolingProto=new MessageDef('PoolingProto','');
  PoolingProto.fields['kernel'] = new FieldDef(1,'kernel','required','int32','The kernel size (square)');
  PoolingProto.fields['pool'] = new FieldDef(30 ,'pool','optional','PoolMethod','The pooling method','MAX');
  PoolingProto.fields['pad'] = new FieldDef(31 ,'pad','optional','uint32','The padding size','0');
  PoolingProto.fields['stride'] = new FieldDef(32 ,'stride','optional','uint32','The stride','1');
  packsinga.messages['PoolingProto']=PoolingProto;

  var SliceProto=new MessageDef('SliceProto','');
  SliceProto.fields['slice_dim'] = new FieldDef(1,'slice_dim','required','int32','');
  packsinga.messages['SliceProto']=SliceProto;

  var ReLUProto=new MessageDef('ReLUProto','');
  ReLUProto.fields['negative_slope'] = new FieldDef(1 ,'negative_slope','optional','float','Ref. Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013). Rectifier nonlinearities improve neural network acoustic models. In ICML Workshop on Deep Learning for Audio, Speech, and Language Processing.','0');
  packsinga.messages['ReLUProto']=ReLUProto;

  var UpdaterType=new EnumDef('UpdaterType','');
  UpdaterType.values['kSGD']=new EnumValueDef('kSGD', 1,'noraml SGD with momentum and weight decay');
  UpdaterType.values['kAdaGrad']=new EnumValueDef('kAdaGrad', 2,'adaptive subgradient, http://www.magicbroom.info/Papers/DuchiHaSi10.pdf');
  UpdaterType.values['kRMSProp']=new EnumValueDef('kRMSProp', 3,'http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf');
  UpdaterType.values['kNesterov']=new EnumValueDef('kNesterov', 4,'Nesterov first optimal gradient method');
  packsinga.enums['UpdaterType']=UpdaterType;

  var ChangeMethod=new EnumDef('ChangeMethod','');
  ChangeMethod.values['kFixed']=new EnumValueDef('kFixed', 0,'');
  ChangeMethod.values['kInverseT']=new EnumValueDef('kInverseT', 1,'');
  ChangeMethod.values['kInverse']=new EnumValueDef('kInverse', 2,'');
  ChangeMethod.values['kExponential']=new EnumValueDef('kExponential', 3,'');
  ChangeMethod.values['kLinear']=new EnumValueDef('kLinear', 4,'');
  ChangeMethod.values['kStep']=new EnumValueDef('kStep', 5,'');
  ChangeMethod.values['kFixedStep']=new EnumValueDef('kFixedStep', 6,'');
  packsinga.enums['ChangeMethod']=ChangeMethod;

  var UpdaterProto=new MessageDef('UpdaterProto','');
  UpdaterProto.fields['type'] = new FieldDef(1 ,'type','required','UpdaterType','updater type','kSGD');
  UpdaterProto.fields['rmsprop_conf'] = new FieldDef(50,'rmsprop_conf','optional','RMSPropProto','configuration for RMSProp algorithm');
  UpdaterProto.fields['lr_change'] = new FieldDef(2 ,'lr_change','required','ChangeMethod','change method for learning rate','kFixed');
  UpdaterProto.fields['fixedstep_conf'] = new FieldDef(40,'fixedstep_conf','optional','FixedStepProto','');
  UpdaterProto.fields['step_conf'] = new FieldDef(41,'step_conf','optional','StepProto','');
  UpdaterProto.fields['linear_conf'] = new FieldDef(42,'linear_conf','optional','LinearProto','');
  UpdaterProto.fields['exponential_conf'] = new FieldDef(43,'exponential_conf','optional','ExponentialProto','');
  UpdaterProto.fields['inverse_conf'] = new FieldDef(44,'inverse_conf','optional','InverseProto','');
  UpdaterProto.fields['inverset_conf'] = new FieldDef(45,'inverset_conf','optional','InverseTProto','');
  UpdaterProto.fields['momentum'] = new FieldDef(31 ,'momentum','optional','float','','0');
  UpdaterProto.fields['weight_decay'] = new FieldDef(32 ,'weight_decay','optional','float','','0');
  UpdaterProto.fields['base_lr'] = new FieldDef(34 ,'base_lr','optional','float','base learning rate','0');
  UpdaterProto.fields['delta'] = new FieldDef(35 ,'delta','optional','float','used to avoid divide by 0, i.e. x/(y+delta)','0.00000001');
  packsinga.messages['UpdaterProto']=UpdaterProto;

  var RMSPropProto=new MessageDef('RMSPropProto','');
  RMSPropProto.fields['rho'] = new FieldDef(1,'rho','required','float','history=history*rho_+(1-rho_)*(grad*grad_scale)');
  packsinga.messages['RMSPropProto']=RMSPropProto;

  var FixedStepProto=new MessageDef('FixedStepProto','');
  FixedStepProto.fields['step'] = new FieldDef(28,'step','repeated','int32','');
  FixedStepProto.fields['step_lr'] = new FieldDef(29,'step_lr','repeated','float','lr = step_lr[i] if current step >= step[i]');
  packsinga.messages['FixedStepProto']=FixedStepProto;

  var StepProto=new MessageDef('StepProto','');
  StepProto.fields['gamma'] = new FieldDef(35 ,'gamma','required','float','lr = base_lr * gamma^(step/change_freq)','1');
  StepProto.fields['change_freq'] = new FieldDef(40,'change_freq','required','int32','lr = base_lr * gamma^(step/change_freq)');
  packsinga.messages['StepProto']=StepProto;

  var LinearProto=new MessageDef('LinearProto','');
  LinearProto.fields['change_freq'] = new FieldDef(40,'change_freq','required','int32','lr = (1 - step / freq) * base_lr + (step / freq) * final_lr');
  LinearProto.fields['final_lr'] = new FieldDef(39,'final_lr','required','float','lr = (1 - step / freq) * base_lr + (step / freq) * final_lr');
  packsinga.messages['LinearProto']=LinearProto;

  var ExponentialProto=new MessageDef('ExponentialProto','');
  ExponentialProto.fields['change_freq'] = new FieldDef(40,'change_freq','required','int32','lr = base / 2^(step/change_freq)');
  packsinga.messages['ExponentialProto']=ExponentialProto;

  var InverseTProto=new MessageDef('InverseTProto','');
  InverseTProto.fields['final_lr'] = new FieldDef(39,'final_lr','required','float','lr = base_lr / (1+step/final_lr)');
  packsinga.messages['InverseTProto']=InverseTProto;

  var InverseProto=new MessageDef('InverseProto','');
  InverseProto.fields['gamma'] = new FieldDef(1 ,'gamma','required','float','lr = base_lr*(1+gamma*step)^(-pow)','1');
  InverseProto.fields['pow'] = new FieldDef(2 ,'pow','required','float','lr = base_lr*(1+gamma*step)^(-pow)','0');
  packsinga.messages['InverseProto']=InverseProto;


  //end python

  Model.package=packsinga;
  Model.package.rootMessage=JobProto;



  //add Some Constrains between fields

  function changeFunction(value){
    var fieldName=value.substr(1).toLowerCase()+"_conf";
    return fieldName;

  }

  LayerProto.addConstrain("type",changeFunction);

  UpdaterProto.addConstrain("type",changeFunction);

  UpdaterProto.addConstrain("lr_change",changeFunction);


  Model.init=function($container) {
    Model.root = new Message(Model.package.rootMessage);
    Model.root.init();
    Model.root.setContainer($container.find(".column:first"));
    Model.root.render();

    var $scroll = $container.find("#left");
    $container.find(".column").bind("click",function(){
      var firstColumnOffset=$(".column:first").offset().left;

      var index=$(this).attr("_index");
      index=parseInt(index);
      $scroll.animate({scrollLeft: ((index-2)*200+100)},100);

    });

  }

  Model.showConf=function(){
    $("#info_conf").find("textarea").val(Model.root.toConf(""));


  };

  $("#uploadButton").click(function(){
    $("#uploadModal").modal("show");
  });

  $("#uploadModal .submit").on("click",function(){
    var file=$("#uploadModal input").prop("files")[0];

    readBlobAsText(file, function (err, result) {
      if (err) {
        alert(err.name + '\n' + err.message);
        return;
      }
      $("#info_conf textarea").val(result);
      var confObj={}
      parseMessage(confObj,result);
      console.log(confObj);
      Model.root.initFromConf(confObj);
      Model.root.render();

    });
    $("#uploadModal").modal("hide");
    return false;
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



  $("#submitForm").click(function(){

    var jobConf = $("#info_conf textarea").val();
    var result = /workspace:(\s*)"(.*)"\n/.exec(jobConf);
    if(result==null){
      alert("workspace not defined!");
      return;
    }
    var workspace =result[2];
    $.ajax({
      url:Model.Config.apiUrl+"/submit",
      type:"POST",
      data:{
        jobconf:jobConf,
        workspace:workspace
      },
      success:function(data){
        data =JSON.parse(data);
        if(data.result=="success"){
          Model.jobId=data.data.jobid;
          monitor();
        }else{
          alert(data.data);
        }
      },
      error:function(e){
        console.log(e);
      }

    });

  });

  $("#monitorBtn").click(function() {
    monitor();

  });

  function monitor(){
    $("#configure").hide();
    $("#monitor").show();
    $(".monitor").show();
    $(".configure").hide();

    $("#chart").empty();
    $("#pic").empty();

    var charts={};
    var pics={};
    var pics_src={"conv":["conv1.jpg","conv2.jpg","conv3.jpg","conv4.jpg"],"filter":["filters_at_epoch_14.png"]};

    var canPolling=true;
    polling();

    function polling(){
      $.ajax({
        url:Model.Config.apiUrl+"/poll/"+Model.jobId,
        type:"GET",
        success:function(data){
          data =JSON.parse(data);
          if(data.result=="success") {
            for (var index in data.data) {
              var item = data.data[index];
              if (item.type == "chart") {
                var chart = charts[item.ylabel];
                if (!chart) {
                  chart = charts[item.ylabel] = new Model.ChartController({model: item});
                } else {
                  chart.addData(item);
                }
              } else {
                var pic = pics[item.title];
                if (!pic) {
                  pic = pics[item.title] = new Model.PicController({model:item.title});
                }
                pic.addData(item);
              }
            }
            canPolling?polling():"";
          }else{
            canPolling?setTimeout(polling, 5000):"";
            alert(data.data);
          }


        },
        error:function(e){
          console.log(e);

        }
      });




    };


    $("#killBtn").click(function(){
      $.ajax({
        url:Model.Config.apiUrl+"/kill/"+Model.jobId,
        type:"GET",
        success:function(data){
          data =JSON.parse(data);
          if(data.result=="success") {
            alert(data.data)
          }else{
            alert(data.data);
          }
          canPolling=false;
        },
        error:function(e){
          console.log(e);
        }
      });


    });


  }

  $("#demoBtn").bind("click",function(){

    $("#demoModal").modal("show");

  });

  $("#uploadSubmit").bind("click",function(){
    $("#demoModal").modal("hide");

    var fd = new FormData(document.getElementById("demoForm"));
    $.ajax({
      url: Model.Config.apiUrl+"/upload",
      type: "POST",
      data: fd,
      processData: false,  // tell jQuery not to process the data
      contentType: false,   // tell jQuery not to set contentType
      success:function(data){
        data =JSON.parse(data);
        if(data.result=="success"){
          Model.jobId=data.data.jobid;
          demo();
        }else{
          alert(data.data);
        }

      },
      error:function(e){
        console.log(e);
      }
    });



  });

  function demo(){
    $("#configure").hide();
    $("#monitor").show();

    $(".monitor").show();
    $(".configure").hide();

    $("#chart").empty();
    $("#pic").empty();

    var charts={};
    var pics={};

    var canPolling=true;
    polling();

    function polling() {
      $.ajax({
        url: Model.Config.apiUrl + "/get/" + Model.jobId,
        type: "GET",
        success: function (data) {
          data = JSON.parse(data);
          if (data.result == "success") {
            for (var index in data.data) {
              var item = data.data[index];
              if (item.type == "chart") {
                var chart = charts[item.ylabel];
                if (!chart) {
                  chart = charts[item.ylabel] = new Model.ChartController({model: item});
                } else {
                  chart.addData(item);
                }
              } else {
                var pic = pics[item.title];
                if (!pic) {
                  pic = pics[item.title] = new Model.PicController({model: item.title});
                }
                pic.addData(item);
              }
            }
            canPolling ? setTimeout(polling, 1000) : "";

          } else {
            canPolling ? setTimeout(polling, 5000) : "";
            alert(data.data);
          }


        },
        error: function (e) {
          console.log(e);

        }
      });
    }
    $("#killBtn").click(function(){
      $.ajax({
        url:Model.Config.apiUrl+"/kill/"+Model.jobId,
        type:"GET",
        success:function(data){
          data =JSON.parse(data);
          if(data.result=="success") {
            alert(data.data)
          }else{
            alert(data.data);
          }
          canPolling=false;
        },
        error:function(e){
          console.log(e);
        }
      });


    });

  }



})();
