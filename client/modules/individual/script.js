/* eslint-disable no-undef */
/* eslint-disable no-unused-vars */
import HttpHelper from "common/utils/axios_helper.js";
import { link } from "d3";
import { h } from "vue";

const beeswarm = require("d3-beeswarm")
const underscore = require('underscore');

export default {
    components: { // 依赖组件
    },
    data() { // 本页面数据
        return {
            tableData: [],
            tag_bg: {
                'high': 'bg_high',
                'med': 'bg_med',
                'low': 'bg_low',
                'churn': 'bg_churn'
            },
            currentRow: null,
            feature_names: [
                'kcore',
                'cn',
                'pr',
                'tran_sex',
                'tran_combatscore',
                'tran_deltatime',
                'tran_bindcash',
                'tran_grade',
                'tran_school',
                'sex',
                'combatscore',
                'deltatime',
                'bindcash',
                'grade',
                'school',
            ],
            shap_values: {
                school: [
                    0.25,
                    0,
                    0,
                    0,
                    0,
                    -0.2088607594936709,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.1757546251217137,
                    0.1757546251217137,
                    0.1757546251217137,
                    0,
                    0.1757546251217137,
                    0.1757546251217137,
                    0,
                    -0.2088607594936709,
                    0,
                    0,
                    0.25,
                    0.1757546251217137,
                    0,
                    0.1757546251217137,
                    -0.7088607594936709,
                    0.1757546251217137,
                    0,
                    0.1757546251217137,
                    0,
                    -0.2088607594936709,
                    0,
                    0,
                    0,
                    0,
                    -0.75,
                    0.25,
                    0.1757546251217137,
                    -0.7088607594936709,
                    -0.2088607594936709,
                    0.1757546251217137,
                    0.1757546251217137,
                    0,
                    0,
                    0,
                    -0.2088607594936709,
                    0.1757546251217137,
                    -0.2088607594936709,
                    0.1757546251217137,
                    0.25,
                    -0.7088607594936709,
                    0.1757546251217137,
                    -0.7088607594936709,
                    0,
                    0.1757546251217137,
                    0.1757546251217137,
                    -0.2088607594936709,
                    -0.2088607594936709,
                    0.25,
                    0,
                    0.1757546251217137,
                    0.1757546251217137,
                    0.25,
                    0.1757546251217137,
                    0.25,
                    0,
                    -0.2088607594936709
                ],
                grade: [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -0.4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                bindcash: [
                    0,
                    0,
                    0,
                    0.16666666666666663,
                    0.16666666666666663,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.16666666666666663,
                    0.03538461538461546,
                    0.03538461538461546,
                    0.03538461538461546,
                    0,
                    0.03538461538461546,
                    0.03538461538461546,
                    0.16666666666666663,
                    0,
                    0,
                    0,
                    0,
                    0.03538461538461546,
                    0,
                    0.03538461538461546,
                    0,
                    0.03538461538461546,
                    0,
                    0.03538461538461546,
                    0.16666666666666663,
                    0,
                    0.16666666666666663,
                    0,
                    0.16666666666666663,
                    0,
                    0,
                    0,
                    0.03538461538461546,
                    0,
                    0,
                    0.03538461538461546,
                    0.03538461538461546,
                    0,
                    0.16666666666666663,
                    0,
                    0,
                    0.03538461538461546,
                    0,
                    0.03538461538461546,
                    0,
                    0,
                    0.03538461538461546,
                    0,
                    0,
                    0.03538461538461546,
                    0.03538461538461546,
                    0,
                    0,
                    0,
                    0.16666666666666663,
                    0.03538461538461546,
                    0.03538461538461546,
                    0,
                    0.03538461538461546,
                    0,
                    0,
                    0
                ],
                deltatime: [
                    0.16694514062935117,
                    -0.5830548593706488,
                    -0.3330548593706488,
                    -0.3330548593706488,
                    -0.3330548593706488,
                    0.22765775197487392,
                    -0.3330548593706488,
                    -0.3330548593706488,
                    -0.3330548593706488,
                    -0.3330548593706488,
                    -0.3330548593706488,
                    0.22765775197487392,
                    0.22765775197487392,
                    0.22765775197487392,
                    -0.3330548593706488,
                    0.22765775197487392,
                    0.22765775197487392,
                    -0.3330548593706488,
                    0.7276577519748739,
                    -0.3330548593706488,
                    -0.5830548593706488,
                    0.16694514062935117,
                    0.22765775197487392,
                    -0.3330548593706488,
                    0.22765775197487392,
                    0.22765775197487392,
                    0.22765775197487392,
                    -0.3330548593706488,
                    0.22765775197487392,
                    -0.3330548593706488,
                    0.7276577519748739,
                    -0.3330548593706488,
                    -0.3330548593706488,
                    -0.3330548593706488,
                    -0.3330548593706488,
                    0.16694514062935117,
                    0.16694514062935117,
                    0.22765775197487392,
                    0.22765775197487392,
                    0.7276577519748739,
                    0.22765775197487392,
                    0.22765775197487392,
                    -0.3330548593706488,
                    -0.3330548593706488,
                    -0.3330548593706488,
                    0.7276577519748739,
                    0.22765775197487392,
                    0.7276577519748739,
                    0.22765775197487392,
                    0.16694514062935117,
                    0.22765775197487392,
                    0.22765775197487392,
                    0.22765775197487392,
                    -0.3330548593706488,
                    0.22765775197487392,
                    0.22765775197487392,
                    0.7276577519748739,
                    0.7276577519748739,
                    0.16694514062935117,
                    -0.3330548593706488,
                    0.22765775197487392,
                    0.22765775197487392,
                    0.16694514062935117,
                    0.22765775197487392,
                    0.16694514062935117,
                    -0.3330548593706488,
                    0.22765775197487392
                ],
                combatscore: [
                    0.10185185185185186,
                    0.10185185185185186,
                    -0.14814814814814814,
                    0.6851851851851852,
                    0.6851851851851852,
                    0.125,
                    -0.14814814814814814,
                    -0.14814814814814814,
                    -0.14814814814814814,
                    -0.14814814814814814,
                    0.6851851851851852,
                    0,
                    0,
                    0.23076923076923073,
                    -0.14814814814814814,
                    0,
                    0,
                    0.6851851851851852,
                    0.125,
                    -0.14814814814814814,
                    0.10185185185185186,
                    0.10185185185185186,
                    0,
                    -0.14814814814814814,
                    0,
                    0,
                    0,
                    -0.14814814814814814,
                    0,
                    0.6851851851851852,
                    0.125,
                    0.6851851851851852,
                    -0.14814814814814814,
                    0.6851851851851852,
                    -0.14814814814814814,
                    0.10185185185185186,
                    0.10185185185185186,
                    0,
                    0,
                    0.125,
                    0,
                    0,
                    -0.14814814814814814,
                    0.6851851851851852,
                    -0.14814814814814814,
                    0.125,
                    0,
                    0.125,
                    -0.36923076923076925,
                    0.10185185185185186,
                    0,
                    0.23076923076923073,
                    0,
                    -0.14814814814814814,
                    0.23076923076923073,
                    0.23076923076923073,
                    0.125,
                    0.125,
                    0.10185185185185186,
                    0.6851851851851852,
                    0.23076923076923073,
                    0,
                    0.10185185185185186,
                    0,
                    0.10185185185185186,
                    -0.14814814814814814,
                    0.125
                ],
                sex: [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                tran_school: [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.061224489795918324,
                    0.061224489795918324,
                    -0.1695447409733124,
                    0,
                    0.061224489795918324,
                    0.061224489795918324,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.061224489795918324,
                    0,
                    0.061224489795918324,
                    0,
                    0.061224489795918324,
                    0,
                    0.061224489795918324,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.061224489795918324,
                    0,
                    0,
                    0.061224489795918324,
                    0.061224489795918324,
                    0,
                    0,
                    0,
                    0,
                    0.061224489795918324,
                    0,
                    -0.1695447409733124,
                    0,
                    0,
                    -0.1695447409733124,
                    0,
                    0,
                    -0.1695447409733124,
                    -0.1695447409733124,
                    0,
                    0,
                    0,
                    0,
                    -0.1695447409733124,
                    0.061224489795918324,
                    0,
                    0.061224489795918324,
                    0,
                    0,
                    0
                ],
                tran_grade: [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                tran_bindcash: [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                tran_deltatime: [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                tran_combatscore: [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                tran_sex: [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0892857142857143,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0892857142857143,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0892857142857143,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0892857142857143,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0892857142857143,
                    0,
                    0.0892857142857143,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0892857142857143,
                    0.0892857142857143,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0892857142857143
                ],
                cn: [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                kcore: [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                pr: [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
            },
            boxplot_data: [],
            indi_data: [],
            shap_data: [],
            table_port_data: [],
            table_port_idx: [],
            // churn group
            churn_group: {
                0: 'high',
                1: 'med',
                2: 'low',
                3: 'churn'
            },
            churn_idx: {
                'high': 0,
                'med': 1,
                'low': 2,
                'churn': 3
            },
            // day group
            day_group: {
                'Day11': 0,
                'Day12': 1,
                'Day13': 2,
                'Day14': 3
            }
        };
    },
    mounted() {
        this.plot_box_legend()
        this.plot_shap_legend()
        this.wait_data()
    },
    watch: {
    },
    methods: { // 这里写本页面自定义方法
        async handleCurrentChange(val) {
            this.currentRow = val;

            if (this.currentRow) {
                var uid = this.currentRow.uid + ''
                var day = this.currentRow.date.slice(1,2)*1
                var churn_status = this.churn_idx[this.currentRow.status]
                var portrait_class = this.currentRow.portrait

                //BOXPLOT: send uid to backend
                var boxplot_link = await HttpHelper.axiosPost('/individual',
                    { 'uid': String(uid), 'day': day }, 600000)
                // boxplot_link.pop()
                this.indi_data = boxplot_link

                this.plot_box()

                //SHAP: send uid to backend
                var shap_data = await HttpHelper.axiosPost('/shap',
                    churn_status, 600000)
                console.log('INDIVIDUAL >> SHAP DATA FINISH.')
                delete shap_data.churn

                this.shap_values = shap_data
                console.log('shap values', this.shap_values)

                // LOADING
                const rLoading = this.openLoading()
                this.plot_shap()
                // loading close
                rLoading.close()
                console.log('INDIVIDUAL >> SHAP FINISH.')

                // INDI PORTRAIT
                if (uid == '') {
                    this.$bus.$emit("indi port to template", []);
                } else {
                    var uid_idx = this.table_port_idx[uid]
                    var uid_port = []
                    uid_idx.forEach(idx => {
                        uid_port.push(this.table_port_data[idx])
                    })
                    console.log(uid_port)
                    // send indi portrait to groupTemplate
                    this.$bus.$emit("indi port to template", uid_port);
                }
            }
            // clear currentRow
            this.currentRow = null
        },
        formatter(row, column) {
            return row.address;
        },
        filterStatus(value, row) {
            return row.status === value;
        },
        filterHandler(value, row, column) {
            const property = column['status'];
            return row[property] === value;
        },
        plot_box() {
            let _this = this;
            //#region data
            data = _this.boxplot_data.length == 0 ? data : _this.boxplot_data

            if (_this.indi_data.length != 0) {
                var indi_data = new Array(_this.feature_names.length).fill(0)
                // arrange indi_data
                _this.indi_data.forEach(feat => {
                    var name = feat.key
                    var idx = _this.feature_names.indexOf(name)
                    indi_data[idx] = feat
                })
            }
            else
                indi_data = []

            //#endregion

            //#region layout and default configs
            const svg = d3.select('#boxSvg');
            svg.selectAll('*').remove()
            var margin = { top: 20, right: 40, bottom: 20, left: 150 }
            var width = 519, height = 382
            const boxWidth = 10;
            const feature_group = data.map(d => d.key)

            //#endregion

            //#region functions
            var y = d3.scaleBand()
                .range([height - margin.bottom, margin.top])
                .domain(_this.feature_names)
                .paddingInner(1)
                .paddingOuter(.5)

            function x(feat) {
                var idx = feature_group.indexOf(feat)
                return d3.scaleLinear()
                    .domain(data[idx].range)
                    .range([margin.left + 10, width - margin.right])
            }

            var yAxis = g => g
                .attr("transform", `translate(${margin.left}, ${0})`)
                .call(d3.axisLeft(y).ticks(null, "s").tickSize(0))

            function xAxis(feat) {
                return g => g
                    .attr("transform", `translate(${0},${height - margin.bottom})`)
                    .call(d3.axisBottom(x(feat)))
            }

            //#endregion

            //#region plot
            const groups = svg.selectAll("g")
                .data(data)
                .enter()
                .append('g')
                .attr("transform", d => `translate(0, ${y(d.key)})`)
                .attr("class", d => d.key);

            groups
                .selectAll("vertLine")
                .data(d => [d])
                .enter()
                .append('line')
                .attr("class", "vertLine")
                .attr("stroke", "#C0C0C0")
                .attr('stroke-width', '1px')
                .style("width", 40)
                .attr("x1", d => {
                    var x1 = x(d.key)(d.range[0])
                    return x1
                })
                .attr("x2", d => {
                    var x2 = x(d.key)(d.range[1])
                    return x2
                })
                .attr("y1", 0)
                .attr("y2", 0);

            groups
                .selectAll("box")
                .data(d => [d])
                .enter()
                .append('rect')
                .attr("class", "box")
                .attr("y", -boxWidth / 2)
                .attr("x", d => x(d.key)(d.quartiles[0]))
                .attr("width", d => {
                    return x(d.key)(d.quartiles[2]) - x(d.key)(d.quartiles[0])
                })
                .attr("height", boxWidth)
                .attr("stroke", "#808080")
                .style("fill", "rgb(255, 255, 255)")
                .style("fill-opacity", 0.7);

            groups
                .selectAll("verticalLine")
                .data(d => [[d.key, d.quartiles[0]],
                [d.key, d.quartiles[1]],
                [d.key, d.quartiles[2]]])
                .enter()
                .append('line')
                .attr("class", "verticalLine")
                .attr("stroke", "#808080")
                .attr('stroke-width', '1px')
                .style("width", 10)
                .attr("y1", -boxWidth / 2)
                .attr("y2", +boxWidth / 2)
                .attr("x1", d => x(d[0])(d[1]))
                .attr("x2", d => x(d[0])(d[1]));

            svg.append("g")
                .call(yAxis);

            svg.select('.domain').remove()

            //#endregion

            //links
            var link_list = []
            for (var i = 0; i < indi_data.length - 1; i++) {
                var pos_list = {}

                var first_ele = indi_data[i]
                var last_ele = indi_data[i + 1]

                //get pos
                var first_x = x(first_ele.key)(first_ele.value),
                    first_y = y(first_ele.key)

                var last_x = x(last_ele.key)(last_ele.value),
                    last_y = y(last_ele.key)

                pos_list.sourcePos = [first_x, first_y]
                pos_list.targetPos = [last_x, last_y]
                link_list.push(pos_list)
            }

            //draw links
            var line = svg.append('g')

            for (var i = 0; i < link_list.length; i++) {
                line.append('path')
                    .attr('d', d3.linkVertical()({
                        source: link_list[i].sourcePos,
                        target: link_list[i].targetPos,
                    }))
                    .attr('fill', 'none')
                    .attr('stroke', "#af5f68")
                    .attr('stroke-width', 3)
                    .attr('stroke-opacity', 0.5)
            }

            // #endregion
        },
        plot_box_legend() {
            var svg = d3.select('#boxLegend')

            var group = svg.append('g')

            group.append('line')
                .attr('x1', 385)
                .attr('x2', 425)
                .attr('y1', 20)
                .attr('y2', 20)
                .attr('stroke', '#af5f68')
                .attr('stroke-width', 3)

            group.append('text')
                .attr('x', 435)
                .attr('y', 25)
                .text('individual')
        },
        plot_shap() {
            let _this = this

            //#region data
            //get values of shap
            var shap = underscore.values(_this.shap_values)
            var feature_names = Object.keys(_this.shap_values)

            //reverse first
            shap = shap.reverse()
            var flat_shap = shap.flat()
            //#endregion

            //#region layout and default configs
            const svg = d3.select('#shapSvg');
            svg.selectAll('*').remove()
            var margin = { top: 20, right: 40, bottom: 20, left: 150 }
            var width = 519, height = 382
            var xaxis_height = 20
            //#endregion

            //#region functions
            const xScale = d3.scaleLinear()
                .domain(d3.extent(flat_shap, d => d))
                .range([margin.left + 10, width - margin.right]);

            const y = d3.scaleBand()
                .range([height - margin.bottom, margin.top + xaxis_height])
                .domain(_this.feature_names)
                .paddingInner(1)
                .paddingOuter(.5)

            var yAxis = g => g
                .attr("transform", `translate(${margin.left}, ${0})`)
                .call(d3.axisLeft(y).ticks(null, "s").tickSize(0))
                .attr('class', 'yAxis')

            //define colors
            let linearArr = shap.map(d => {
                return d3.scaleLinear().domain(d3.extent(d, dd => +dd)).range([0, 1])
            })
            // let compute = d3.interpolate('red', 'steelblue')
            let compute = d3.interpolate('#B64330', '#72A6C1')

            //#endregion

            //#region plot
            // define arrangement
            var swarm = shap.map(function (d, i) {
                // console.log('shap map', d)
                return beeswarm.beeswarm()
                    .data(d) // set the data to arrange
                    .distributeOn(function (dd) {
                        // set the value accessor to distribute on
                        return xScale(+dd); // evaluated once on each element of data
                    }) // when starting the arrangement
                    .radius(0.02) // set the radius for overlapping detection
                    .orientation('horizontal') // set the orientation of the arrangement
                    // could also be 'vertical'
                    .side('symetric') // set the side(s) available for accumulation
                    // could also be 'positive' or 'negative'
                    .arrange(); // launch arrangement computation;
            })

            svg.selectAll("g")
                .data(swarm)
                .enter()
                .append("g")
                .attr('transform', function (d, i) {
                    return `translate(0,${y(feature_names[i])})`
                })
                .each(function (d, i) {
                    d3.select(this)
                        .selectAll('circle')
                        .data(d)
                        .enter()
                        .append('circle')
                        .attr('cx', function (bee) {
                            return bee.x;
                        })
                        .attr('cy', function (bee) {
                            return bee.y;
                        })
                        .attr('r', 2)
                        .style("fill", (dd, j) => {
                            return shap[i][j] ? compute(linearArr[i](+shap[i][j])) : "#666A5370"
                        })
                })

            //first append yaxis
            svg.append("g")
                .call(yAxis);
            // only remove axis of yaxis
            svg.select('.domain').remove()

            svg.append("g")
                .attr('transform', function (d, i) { return `translate(0,${margin.top})` })
                .call(d3.axisBottom(xScale).ticks(5))


            //#endregion
        },
        plot_shap_legend() {
            var svg = d3.select('#shapLegend')
            var group = svg.append('g')

            var start_color = '#72A6C1', end_color = '#B64330'

            const defs = group
                .append("g")
                .append("defs");
            const linearGradient = defs
                .append("linearGradient")
                .attr("id", "gradient");
            linearGradient 
                .append("stop")
                .attr("offset", "0%") 
                .attr("stop-color", start_color);
            linearGradient
                .append("stop")
                .attr("offset", "100%")
                .attr("stop-color", end_color);

            var rect = group
                .append("rect")
                .attr("transform", "translate(" + 255 + ", " + 19 + ")")
                .attr("height", 4)
                .attr("width", 150)
                .style("fill", "url('#gradient')");
            var text = group
                .append('text')
                .attr("transform", "translate(" + 415 + ", " + 25 + ")")
                .text('feature value')
        },
        async wait_data() {
            this.$bus.$on("template to table", msg => {
                var data = msg

                // save table data and table portrait data
                var table_data = [], table_port_data = [], table_port_idx = {}

                data.forEach((element, i) => {
                    // table
                    var per_row = {}
                    per_row.date = 'D' + this.day_group[element.date_group]
                    per_row.portrait = element.class
                    per_row.uid = element.uid
                    per_row.status = this.churn_group[element.pred]
                    table_data.push(per_row)

                    // port
                    var per_port = {}
                    per_port.uid = element.uid
                    per_port.group = element.class
                    per_port.date = this.day_group[element.date_group]
                    per_port.indi_portrait = []
                    per_port.indi_portrait.push({ key: 'school', value: element.school })
                    per_port.indi_portrait.push({ key: 'grade', value: element.grade })
                    per_port.indi_portrait.push({ key: 'bindcash', value: element.bindcash })
                    per_port.indi_portrait.push({ key: 'deltatime', value: element.deltatime })
                    per_port.indi_portrait.push({ key: 'combatscore', value: element.combatscore })
                    per_port.indi_portrait.push({ key: 'sex', value: element.sex })
                    table_port_data.push(per_port)

                    if (typeof (table_port_idx[element.uid]) == 'undefined')
                        table_port_idx[element.uid] = []
                    table_port_idx[element.uid].push(i)
                });

                this.tableData = table_data
                this.table_port_data = table_port_data
                this.table_port_idx = table_port_idx
            });

            this.$bus.$on("template to boxplot", msg => {
                var data = msg
                this.boxplot_data = data
                this.plot_box()

            })
        },
    },
    created() { // 生命周期中，组件被创建后调用
    },
};