<template >
  <!-- 这里写Html -->
  <div class="groupWrapper" style="position: relative">
    <el-row style="height: 41px">
      <div class="pred_rect">
        <div class="pred_title">Prediction</div>
      </div>
      <div class="clus_rect">
        <div class="clus_title">Clusters</div>
        <div class="link_filter_title">Link Fliter</div>
        <div class="slider" style="position: absolute; top: 2px; left: 250px">
          <!-- <span class="demonstration"></span> -->
          <el-slider
            v-model="slider_value"
            range
            :min="0"
            :max="100"
            :format-tooltip="formatTooltip"
            style="width: 200px"
            ref="slider"
          >
          </el-slider>
        </div>
        <el-button
          size="small"
          @click="filter_link()"
          icon="el-icon-check"
          circle
          style="position: absolute; top: 3px; left: 470px"
        ></el-button>

        <div class="counter_module">
          <div class="counter_module_title">Counterfactual Module</div>
        <el-select @change="cf_module_change" class="module_select" v-model="module_value" placeholder="Please choose module.">
          <el-option
            v-for="item in counter_modules"
            :key="item.value"
            :label="item.label"
            :value="item.value"
            :id="item.label"
          >
          </el-option>
        </el-select>
        </div>

        <svg id="clusLegend"></svg>
      </div>
      <div class="indiport_rect">
        <div class="indiport_title">Indi Port</div>
      </div>
    </el-row>

    <el-scrollbar style="height: 830px; width: 1998px">
      <div style="position: relative; height: 825px">
        <div v-if="updated_flag" style="position: absolute; left: 0; top: 0">
          <groupTemplate
            v-for="(d, i) in data"
            :key="i"
            :id="i"
            :data="d"
          ></groupTemplate>
        </div>
        <div
          style="
            width: 1750px;
            height: 825px;
            position: relative;
            top: 0;
            left: 0;
            z-index: -2;
          "
        >
          <svg id="lineSvg" :style="lineSvgStyle"></svg>
        </div>
      </div>
    </el-scrollbar>
    <div class="ob_sequence">
      <div class="obse_rect">
        <div class="obse_title">Observation Sequence</div>
        <svg id="obseLegend"></svg>
      </div>
      <svg
        id="comparisonSvg"
        style="height: 320; width: 1998; position: absolute; top: 46"
      ></svg>
    </div>
  </div>
</template>
<script src="./script.js"></script>

<style type="text/css">
.groupWrapper .el-scrollbar .el-scrollbar__wrap {
  overflow-x: hidden;
  height: 825px;
  /* overflow-y: hidden; */
}
/* background: #D27D1C10; */
.groupWrapper .el-scrollbar__bar.is-horizontal {
  height: 0px !important;
  left: 0px !important;
}
.counter_module_title{
  position: absolute;
  left: -190px;
  top: 7px;
  font: 16px "KaiseiOptiRegular";
}
.counter_module{
  position: absolute;
  left: 740px;
  width: 70px;
}
.obse_rect {
  height: 41px;
  width: 2000px;
  position: absolute;
  left: 1px;
  top: 0px;
  background: #bde3f3;
  border-radius: 10px;
  /* border-top-right-radius: 10px; */
  /* border-top-left-radius: 10px; */
}
.obse_title {
  height: 41px;
  position: absolute;
  left: 20px;
  top: 0px;
  font: 24px "KaiseiOptiMedium";
}
.ob_sequence {
  position: relative;
  top: -3px;
  width: 1998px;
  height: 370px;
  /* border-top: 0.25px solid #d6d6d6; */
}
#clusLegend {
  width: 620px;
  height: 40px;
  position: absolute;
  right: 10px;
  top: 0px;
}
#obseLegend {
  width: 800px;
  height: 40px;
  position: absolute;
  right: 20px;
  top: 0px;
}
.pred_rect {
  height: 41px;
  width: 226px;
  position: absolute;
  left: 0px;
  top: 0px;
  background: #f7cdc650;
  border-top-right-radius: 10px;
  border-top-left-radius: 10px;
}
.pred_title {
  width: 135px;
  height: 41px;
  position: absolute;
  left: 23px;
  top: 4px;
  font: 20px "KaiseiOptiRegular";
}
.clus_rect {
  height: 41px;
  width: 1614px;
  position: absolute;
  left: 226px;
  top: 0px;
  background: #f7cdc650;
  border-top-right-radius: 10px;
  border-top-left-radius: 10px;
}
.clus_title {
  width: 115px;
  height: 41px;
  position: absolute;
  left: 20px;
  top: 4px;
  font: 20px "KaiseiOptiRegular";
}
.indiport_rect {
  height: 41px;
  width: 160px;
  position: absolute;
  left: 1840px;
  top: 0px;
  background: #f8d6ae50;
  border-top-right-radius: 10px;
  border-top-left-radius: 10px;
}
.indiport_title {
  width: 115px;
  height: 41px;
  position: absolute;
  left: 20px;
  top: 4px;
  font: 20px "KaiseiOptiRegular";
}
.link_filter_title {
  width: 115px;
  height: 41px;
  position: absolute;
  left: 150px;
  top: 7px;
  font: 16px "KaiseiOptiRegular";
}
.el-slider__bar {
  background-color: #daa097;
}

.el-slider__button-wrapper .el-slider__button {
  border: 2px solid #bf5c4b;
  width: 15px;
  height: 15px;
}
.el-input{
  padding-top: 5px;
  width: 170px;
}
.el-input__inner{
  height: 30px;
  font-family: "KaiseiOptiRegular";
}
.el-input__icon{
  margin-top: 2px;
  line-height:5px;
}
</style>


