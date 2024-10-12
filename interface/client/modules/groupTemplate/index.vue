<template >
  <!-- 这里写Html -->
  <div class="groupTemplateWrapper">
    <div class="pred_port">
      <div class="pred">
        <svg :id="'predSvg' + id" style="width: 223px; height: 275px"></svg>
      </div>
      <div class="port">
        <svg
          :id="'portraitSvg' + id"
          style="width: 1616px; height: 275px"
        ></svg>
        <el-button
          size="medium"
          @click="send_range_to_group()"
          icon="el-icon-check" circle
          style="position: absolute; top: 50px; right: 30px"
        ></el-button>
        <!-- <el-button
          size="medium"
          @click="send_range_to_individual()"
          icon="el-icon-refresh" circle
          style="position: absolute; top: 91.4px; right: 30px"
        ></el-button> -->
        <el-button
          size="medium"
          @click="show = !show"
          :icon="show ? 'el-icon-caret-top' : 'el-icon-caret-bottom'" circle
          style="position: absolute; bottom: 50px; right: 30px"
        ></el-button>
      </div>
      <div class="indi_port">
        <svg
          :id="'indi_portSvg' + id"
          style="width: 160px; height: 275px"
        ></svg>
      </div>
    </div>

    <div>
      <el-collapse-transition>
        <div v-show="show" style="border-bottom: 2px solid #B6433010">
          <div class="counter" id="counter">
            <svg
              :id="'counterfactualSvg' + id"
              style="
                position: relative;
                z-index: 20;
                width: 1614px;
                height: 548px;
              "
            ></svg>
          </div>
        </div>
      </el-collapse-transition>
    </div>
  </div>
</template>
<script src="./script.js"></script>

<style type="text/css">
.groupTemplateWrapper {
  overflow: hidden;
  position: relative;
}
.pred_port {
  width: 1998px;
  height: 275px;
  border-bottom: 2px solid #B6433010;
}
.pred {
  width: 223px;
  height: 275px;
  border-right: 2px solid #B6433010;
  /* border-bottom: 2px solid #B6433010; */
  /* border-bottom: 2px solid #d6d6d6; */
  position: absolute;
}
.port {
  width: 1616px;
  height: 275px;
  position: absolute;
  left: 223px;
  border-right: 2px solid #B6433010;
  /* border-bottom: 2px solid #B6433010; */
}
.indi_port {
  width: 158px;
  height: 275px;
  position: absolute;
  left: 1839px;
  z-index: 10;
  border-bottom: 2px solid #D27D1C07;
  /* border-right: 2px solid #D27D1C10; */
}
.counter {
  position: relative;
  left: 223px;
  width: 1614px;
  height: 548px;
  border-left: 2px solid #B6433010;
  border-right: 2px solid #B6433010;
}
.step {
  z-index: 30;
  position: absolute;
  left: 0px;
  top: 275px;
}
</style>


