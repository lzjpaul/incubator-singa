(function ($) {

  'use strict';
  Model.PicController = function (options) {
    this.op = $.extend({}, Model.PicController.defaultOptions, options)
    this.model=this.op.model;
    this.init();
  }
  Model.PicController.defaultOptions = {

    pointSize: 20
  }
  Model.PicController.prototype = {
    init: function () {
      console.log("Pic controller init...");

      this.render();

    },
    render: function () {
      var me = this;

      var source = $("#pic_template").html();
      var template = Handlebars.compile(source);
      var html = template({name:me.model});
      this.container=$(html).appendTo($("#pic"));

      this.bind();
    },
    bind: function () {
      var me = this;
      var changeInterval;
      this.container.find(".btn").on("click",function(){
        var pics=me.container.find("img");
        $("#picModal").modal({backdrop:false,show:true});
        var i=0;
        changeInterval = setInterval(function(){
          $("#picModal img").attr("src",$(pics[i]).attr("src"));
          i=(i+1)%pics.length;
        },100);


      });
      $("#picModal .close").click(function(){

        clearInterval(changeInterval);

      })


    },
    addData: function (data) {
      this.container.find(".pic_container").append("<img src='"+Model.Config.host+data.url+"' title='"+data.step+"'/>")

    }
  }

})(jQuery);


