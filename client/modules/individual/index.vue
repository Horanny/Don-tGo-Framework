<template >
  <!-- 这里写Html -->
  <div class="IndividualWrapper">
    <el-row style="height: 41px">
      <div class="table_rect">
        <div class="table_title">Indi Record</div>
      </div>
    </el-row>

    <el-scrollbar style="height: 359px; width: 100%">
      <div class="table">
        <el-table
          :data="tableData"
          empty-text="no data at this time"
          style="width: 100%"
          height="400"
          highlight-current-row
          @current-change="handleCurrentChange"
          :row_style="{ height: '55px !important' }"
          :cell-style="{ padding: '4px 0' }"
        >
          <el-table-column prop="date" width="90" label="date" align="center">
          </el-table-column>
          <el-table-column
            prop="portrait"
            width="90"
            label="portrait"
            align="center"
          >
          </el-table-column>
          <el-table-column prop="uid" label="uid" align="center">
          </el-table-column>
          <el-table-column
            prop="status"
            label="prediction"
            width="120"
            align="center"
            :filters="[
              { text: 'high', value: 'high' },
              { text: 'med', value: 'med' },
              { text: 'low', value: 'low' },
              { text: 'churn', value: 'churn' },
            ]"
            :filter-method="filterStatus"
            filter-placement="bottom-end"
          >
            <template slot-scope="scope">
              <el-tag :class="tag_bg[scope.row.status]" disable-transitions>
                {{ scope.row.status }}
              </el-tag>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </el-scrollbar>

    <div class="boxplot">
      <div class="box_rect">
        <div class="box_title">Feature Distribution</div>
        <svg id="boxLegend"></svg>
      </div>
      <svg id="boxSvg" style="width: 521px; height: 382px; position: absolute; top:441px"></svg>
    </div>
    <div class="shap">
      <div class="shap_rect">
        <div class="shap_title">Shapley Value</div>
        <svg id="shapLegend"></svg>
      </div>
      <svg id="shapSvg" style="width: 521px; height: 382px; position: absolute; top:864px;"></svg>
    </div>
  </div>
</template>
<script src="./script.js"></script>

<style type="text/css">
.IndividualWrapper .el-scrollbar .el-scrollbar__wrap {
  overflow-x: hidden;
  /* overflow-y: hidden; */
}
.table {
  width: 100%;
  border-left: 2px solid #d27d1c10;
  /* height: 400px; */
}
.el-table__body-wrapper::-webkit-scrollbar {
  width: 0;
}
.el-table__body-wrapper::-webkit-scrollbar-track {
  border: none;
}
.el-table--scrollable-y .el-table__body-wrapper::-webkit-scrollbar {
  width: 0;
}
.el-table .el-table__fixed-right-patch {
  width: 0px !important;
}
.el-table th.gutter {
  display: none;
  width: 0;
}
.el-table colgroup col[name="gutter"] {
  display: none;
  width: 0;
}
.el-table__body {
  width: 100% !important;
}
.bg_high {
  height: 25px;
  width: 57px;
  line-height: 22px;
  color: #666a53;
  background: #666a5320;
  border-color: #666a53;
}
.bg_med {
  height: 25px;
  width: 57px;
  line-height: 22px;
  color: #d27d1c;
  background: #d27d1c20;
  border-color: #d27d1c;
}
.bg_low {
  height: 25px;
  width: 57px;
  line-height: 22px;
  color: #72a6c1;
  background: #72a6c120;
  border-color: #72a6c1;
}
.bg_churn {
  height: 25px;
  width: 57px;
  line-height: 22px;
  color: #b64330;
  background: #b6433020;
  border-color: #b64330;
}
.el-table--striped .el-table__body tr.el-table__row--striped.current-row td,
.el-table__body tr.current-row > td {
  background-color: #faf0e550 !important;
}
.el-table__body tr:hover > td {
  background-color: #faf0e550 !important;
}
.table_rect {
  height: 41px;
  width: 523px;
  position: absolute;
  left: 0px;
  top: 0px;
  background: #f8d6ae50;
  border-top-right-radius: 10px;
  border-top-left-radius: 10px;
}
.table_title {
  width: 200px;
  height: 41px;
  position: absolute;
  left: 20px;
  top: 4px;
  font: 20px "KaiseiOptiRegular";
}

.box_rect {
  height: 41px;
  width: 523px;
  position: absolute;
  left: 0px;
  top: 400px;
  background: #f8d6ae50;
  border-top-right-radius: 10px;
  border-top-left-radius: 10px;
}
.box_title {
  width: 300px;
  height: 41px;
  position: absolute;
  left: 20px;
  top: 4px;
  font: 20px "KaiseiOptiRegular";
}
#boxLegend{
  width: 523px;
  height: 40px;
  position: absolute;
  right: 0px;
}
.boxplot {
  width: 100%;
  height: 423px;
  /* border-top: 2px solid #d27d1c10; */
  border-left: 2px solid #f8d6ae50;
  /* border-bottom: 2px solid #d27d1c10; */
}

.shap_rect {
  height: 41px;
  width: 523px;
  position: absolute;
  left: 0px;
  top: 826px;
  background: #f8d6ae50;
  border-top-right-radius: 10px;
  border-top-left-radius: 10px;
}
.shap_title {
  width: 300px;
  height: 41px;
  position: absolute;
  left: 20px;
  top: 4px;
  font: 20px "KaiseiOptiRegular";
}
#shapLegend{
  width: 523px;
  height: 40px;
  position: absolute;
  right: 0px;
}
.shap {
  border-left: 2px solid #f8d6ae50;
  width: 100%;
  height: 425px;
}

</style>


