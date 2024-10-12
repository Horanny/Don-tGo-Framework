<template >
  <!-- 这里写Html -->
  <div class="timeWrapper">
    <div class="timeline_rect">
      <div class="timeline_title">Prediction Outcomes</div>
    </div>
    <div class="options">
      <el-checkbox-group v-model="checkedStatus">
        <el-checkbox
          v-for="status in all_status"
          :id="status.label"
          :label="status.label"
          :key="status.value"
          :style="{ color: status.color, border: '1px solid ' + status.color }"
          border
        >
          {{ status.label }}
        </el-checkbox>
      </el-checkbox-group>
      <div class="upload">
        <el-button
          icon="el-icon-check"
          circle
          @click="send_days()"
        ></el-button>
      </div>
    </div>
    <svg id="timeSvg"></svg>
    <svg id="timeLegend"></svg>
    <div class="save_rect">
      <div class="save_title">SAVE</div>
    </div>
    <div class="save">
      <el-button icon="el-icon-upload" circle @click="save()"></el-button>
    </div>
  </div>
</template>
<script src="./script.js"></script>

<style type="text/css">
.timeline_rect {
  height: 41px;
  width: 270px;
  position: absolute;
  left: -105px;
  top: 5px;
  background: #d4ea9f;
  border-radius: 10px;
}
.timeline_title {
  width: 270px;
  height: 41px;
  position: absolute;
  left: 12px;
  top: 2px;
  font: 24px "KaiseiOptiMedium";
}
.save_rect {
  height: 41px;
  width: 103px;
  position: absolute;
  right: 705px;
  top: 5px;
  background: #d4ea9f;
  border-radius: 10px;
}
.save_title {
  width: 135px;
  height: 41px;
  position: absolute;
  right: -50px;
  top: 2px;
  font: 24px "KaiseiOptiMedium";
}
.options {
  position: absolute;
  right: 1520px;
  top: 6px;
}
#timeSvg {
  width: 600px;
  height: 80px;
  position: absolute;
  right: 840px;
  top: -15px;
}
#timeLegend {
  width: 600px;
  height: 80px;
  position: absolute;
  right: 0px;
  top: -15px;
}
.upload {
  position: absolute;
  right: -70px;
  top: 0px;
}
.select {
  margin: 20px 0 0 0;
  font: 24px "KaiseiOptiRegular";
  position: absolute;
  right: 1200px;
  top: 0px;
}
.save {
  position: absolute;
  top: 3px;
  right: 651px;
  font: 24px "KaiseiOptiRegular";
}

.el-checkbox.is-bordered{
  margin-right: 0px;
}

.el-checkbox.is-bordered.is-checked {
  border: 5px;
}

#high.el-checkbox.is-bordered.is-checked {
  background-color: #666a5320 !important;
}

#high > .el-checkbox__input.is-checked + .el-checkbox__label {
  color: #666a53 !important;
}

#high > .el-checkbox__input.is-checked .el-checkbox__inner,
.el-checkbox__input.is-indeterminate .el-checkbox__inner {
  background-color: #666a53;
  border-color: #666a53;
}

#high > .el-checkbox__input > .el-checkbox__inner:hover {
  border-color: #666a53 !important;
}

#high > .el-checkbox__input.is-focus .el-checkbox__inner {
  border-color: #666a53 !important;
}

#med.el-checkbox.is-bordered.is-checked {
  background-color: #d27d1c20 !important;
}

#med > .el-checkbox__input.is-checked + .el-checkbox__label {
  color: #d27d1c !important;
}

#med > .el-checkbox__input.is-checked .el-checkbox__inner,
.el-checkbox__input.is-indeterminate .el-checkbox__inner {
  background-color: #d27d1c;
  border-color: #d27d1c;
}

#med > .el-checkbox__input > .el-checkbox__inner:hover {
  border-color: #d27d1c !important;
}

#med > .el-checkbox__input.is-focus .el-checkbox__inner {
  border-color: #d27d1c !important;
}

#low.el-checkbox.is-bordered.is-checked {
  background-color: #72a6c120 !important;
}

#low > .el-checkbox__input.is-checked + .el-checkbox__label {
  color: #72a6c1 !important;
  border-color: #72a6c1;
}

#low > .el-checkbox__input.is-checked .el-checkbox__inner,
.el-checkbox__input.is-indeterminate .el-checkbox__inner {
  background-color: #72a6c1;
  border-color: #72a6c1;
}

#low > .el-checkbox__input > .el-checkbox__inner:hover {
  border-color: #72a6c1 !important;
}

#low > .el-checkbox__input.is-focus .el-checkbox__inner {
  border-color: #72a6c1 !important;
}

#churn.el-checkbox.is-bordered.is-checked {
  background-color: #b2372220 !important;
}

#churn > .el-checkbox__input.is-checked + .el-checkbox__label {
  color: #b23722 !important;
  border-color: #b23722;
}

#churn > .el-checkbox__input.is-checked .el-checkbox__inner,
.el-checkbox__input.is-indeterminate .el-checkbox__inner {
  background-color: #b23722;
  border-color: #b23722;
}

#churn > .el-checkbox__input > .el-checkbox__inner:hover {
  border-color: #b23722 !important;
}

#churn > .el-checkbox__input.is-focus .el-checkbox__inner {
  border-color: #b23722 !important;
}

.el-checkbox__label{
  font-family: "KaiseiOptiRegular ";
}
</style>


