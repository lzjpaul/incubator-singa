(function ($) {

  'use strict';
  Model.ChartController = function (options) {
    this.op = $.extend({}, Model.ChartController.defaultOptions, options)
    this.model = this.op.model;
    this.init();
  }
  Model.ChartController.defaultOptions = {
    container: $("#monitor"),
    pointSize: 20  //the number of points that show in a chart
  }
  Model.ChartController.prototype = {
    init: function () {
      console.log("Chart controller init...");
      this.container = $("<div></div>").attr("id", this.model.ylabel).appendTo($("#chart"));
      this.render();

    },
    render: function () {
      var me = this;

      var source = $("#chart_template").html();
      var template = Handlebars.compile(source);
      var html = template(me.model);

      var chartOptions = $.extend({}, Model.ChartDefines["spline"]);
      var xlabel=me.model.xlabel;
      var ylabel=me.model.ylabel;
      var phase= me.model.phase;
      chartOptions.title = {text: ylabel};

      var data=[];
      $.each(me.model.data,function(index,d){
        data.push({x: parseInt(d["x"]),y:parseFloat(d["y"])})

      });

      chartOptions.series = [{
        name: phase,
        data: data
      }];
      me.$dom = $(html).prependTo(me.container).find(".highchart");
      me.$dom.highcharts(chartOptions);


      this.bind();
    },
    bind: function () {



    },
    addData: function (originData) {
      var phase = originData.phase;
      var series = this.$dom.highcharts().series;
      var serie=series.filter(function(s){return s.name==phase;})[0];
      if(!serie){
        var data=[];
        $.each(originData.data,function(index,d){
          data.push({x: parseInt(d["x"]),y:parseFloat(d["y"])})
        });

        this.$dom.highcharts().addSeries({
          name:phase,
          data:data
        });
      }else{
        $.each(originData.data,function(index,d){
          serie.addPoint([parseInt(d["x"]), parseFloat(d["y"])], true, false);
        });

      }
    }
  }


  Model.ChartDefines = {
    "spline": {
      chart: {
        type: 'spline',
        animation: Highcharts.svg, // don't animate in old IE
        marginRight: 10
      },
      title: {
        text: 'Live random data'
      },
      xAxis: {},
      yAxis: {
        title: {
          text: 'Value'
        },
        plotLines: [{
          value: 0,
          width: 1,
          color: '#808080'
        }]
      },
      tooltip: {
        formatter: function () {
          return '<b>' + this.series.name + '</b><br/>' +
            this.x+'<br/>' +
            Highcharts.numberFormat(this.y, 2);
        }
      },
      legend: {
        enabled: false
      },
      exporting: {
        enabled: false
      }
    }
  }


})(jQuery);


