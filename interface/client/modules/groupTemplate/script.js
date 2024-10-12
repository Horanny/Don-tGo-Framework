/* eslint-disable no-undef */
/* eslint-disable no-unused-vars */
import { data } from "browserslist";
import HttpHelper from "common/utils/axios_helper.js";
import { cluster, max, svg, thresholdScott } from "d3";
import Vue, { h } from "vue";

export default {
    components: { // 依赖组件

    },
    props: ['id', 'data'],
    data() { // 本页面数据
        return {
            show: false,
            bar_opacity: 0.8,
            // cluster range
            portrait_max: [5.594203, 52.32969, 369998.6366,
                349502000, 128218.3588, 1.35125],
            // range infos
            range: [
                { school: [] },
                { grade: [] },
                { bindcash: [] },
                { combatscore: [] },
                { sex: [] },
                { deltatime: [] },
                { pr: [] },
                { kcore: [] },
                { cn: [] },
                { tran_school: [] },
                { tran_grade: [] },
                { tran_deltatime: [] },
                { tran_bindcash: [] },
                { tran_combatscore: [] },
                { tran_sex: [] },
            ],
            change: { change: [] },
            feat_idx: {
                school: 0,
                grade: 1,
                bindcash: 2,
                combatscore: 3,
                sex: 4,
                deltatime: 5,
                pr: 6,
                kcore: 7,
                cn: 8,
                tran_school: 9,
                tran_grade: 10,
                tran_deltatime: 11,
                tran_bindcash: 12,
                tran_combatscore: 13,
                tran_sex: 14,
            },
            status_idx: {
                'high': 0,
                'med': 1,
                'low': 2,
                'churn': 3,
            },
            indi_port_data: [],

            // prediction interactions
            pred_click_flag: { high: false, med: false, low: false, churn: false },
            clicked_pred: [],

            // cluster interactions
            clu_click_flag: { 0: false, 1: false, 2: false, 3: false, 4: false, 5: false, 6: false },
            clicked_clus: [],

            // counterfactual interactions
            change_flag: {
                school: false,
                grade: false,
                bindcash: false,
                combatscore: false,
                sex: false,
                deltatime: false,
                pr: false,
                kcore: false,
                cn: false,
                tran_school: false,
                tran_grade: false,
                tran_deltatime: false,
                tran_bindcash: false,
                tran_combatscore: false,
                tran_sex: false,
            },

            change_button: [],
        };
    },
    mounted() {
        this.plot_portrait('portraitSvg' + this.id)
        this.plot_pred('predSvg' + this.id)

        //IF NO DATA IN RAW CF, DO NOT PLOT IT
        if (typeof (this.data.cf_raw) != 'undefined')
            this.plot_counterfactual_raw("counterfactualSvg" + this.id)

        //IF NO DATA IN CF, DO NOT PLOT IT
        if (typeof (this.data.cf) != 'undefined') {
            this.plot_counterfactual("counterfactualSvg" + this.id)
        }

        this.wait_data()

    },
    watch: {
        async data() {
            if (typeof (this.data.cf_change) != 'undefined') {
                this.change_button = this.data.cf_change
            }

            this.plot_portrait('portraitSvg' + this.id)
            this.plot_pred('predSvg' + this.id)
            this.change_pred_style('predSvg' + this.id)
            this.change_clus_style('portraitSvg' + this.id)

            //IF NO DATA IN RAW CF, DO NOT PLOT IT
            if (typeof (this.data.cf_raw) != 'undefined')
                this.plot_counterfactual_raw("counterfactualSvg" + this.id)

            //IF NO DATA IN CF, DO NOT PLOT IT
            if (typeof (this.data.cf) != 'undefined') {
                await this.plot_counterfactual("counterfactualSvg" + this.id)
                if (this.change_button.length != 0)
                    this.change_button_style("counterfactualSvg" + this.id)
            }
            this.wait_data()

        },
        show() {
            this.$bus.$emit("template to group", [this.id, this.show]);
        }
    },
    methods: { // 这里写本页面自定义方法
        plot_portrait(id) {
            let _this = this

            //#region dealing with data
            var data = _this.data.values
            // console.log('TEMPLATE >> PORTRAIT >> port data', data)

            //get max values in each dim of portraits and num
            function maxValues(da) {
                let status_vals = []
                for (var i = 0; i < da[0].num.length; i++) {
                    status_vals.push(d3.max(da.map(d => {
                        return d.num[i].value
                    })))
                }
                return status_vals
            }
            // var max_val = maxValues(data)[0], 
            var max_status = maxValues(data)
            var max_val = _this.portrait_max
            //#endregion

            //#region basic layout config
            const margin = { top: 40, right: 125, bottom: 40, left: 40 }  // focus
            var svg = d3.select('#' + id)
            svg.selectAll('*').remove()

            //portrait
            const radius = 50, dotRadius = 3, selected_dotRadius = 5;
            const num_dim = 6;
            const angleSlice = 2 * Math.PI / num_dim;
            const axisCircles = 2, axisLabelFactor = 1.2;
            const port_group = ['school', 'grade', 'bindcash', 'deltatime', 'combatscore', 'sex']
            const delta_len = 125 + radius * 2, left_start = 40 + radius, height = 110

            //num bars
            // var bgColor = "#D6EAF8"
            var bgColor = "#d6d6d6"
            var status_color = ['#676b54', '#d27d1c', '#4881b3', '#b23722']
            const status_group = ['high', 'med', 'low', 'churn']
            var num_Yrange = [0, d3.max(max_status)]
            var num_height = 30, num_width = 20, delta_port_num = 15
            //#endregion

            //#region functions
            //scale each dim, return a list
            var rScale = max_val.map(el => d3.scaleLinear().domain([0, el]).range([0, radius]))

            //get line for each dim
            var radarLine = d3.lineRadial()
                .curve(d3.curveCardinalClosed)
                .radius((d, i) => rScale[i](d))
                .angle((d, i) => i * angleSlice)

            var colors = d3.scaleOrdinal()
                .domain(d3.range(0, data.length - 1))
                // .range([
                //     '#243258',//1
                //     '#297245',//2
                //     '#f6be64',//3
                //     '#eeb0b0',//4
                //     '#963460',//5
                //     "#8d5729",//6
                //     '#008a9b',//7
                // ])
                // .range([
                //     '#555775',//1
                //     '#528d6a',//2
                //     '#c0d06b',//3
                //     '#ebd168',//4
                //     '#edc0ba',//5
                //     "#aa5d7b",//6
                //     '#30a3b1',//7
                // ]);
                .range([
                    '#9799b6',//1
                    '#78c497',//2
                    '#c0d068',//3
                    '#e8d168',//4
                    '#edc0ba',//5
                    "#d8a0eb",//6
                    '#7fc4cd',//7
                ]);


            //functions for num bars
            var num_x = d3.scaleBand()
                .domain(status_group)
                .range([0, radius * 3.5])
                .padding(0.1)

            var num_y = d3.scaleLinear()
                .domain(num_Yrange)
                .range([num_height, 0])

            var num_color = d3.scaleOrdinal()
                .domain(status_group)
                .range(status_color)

            //#region mouse functions
            function polygon_overed(event, d) {
                var mom = d3.select(this)
                var mom_class = mom.attr('class')
                var group = mom_class.substring(7)

                // polygon opacity
                svg.select('.' + mom_class)
                    .attr('fill-opacity', 0.7)

                // dot radius
                svg.selectAll('.dot' + group)
                    .attr('r', selected_dotRadius)

                // lines
                d3.selectAll('._group_' + _this.data.id + group)
                    .attr('opacity', 1)
            }

            function polygon_outed(event, d) {
                var mom = d3.select(this)
                var mom_class = mom.attr('class')
                var group = mom_class.substring(7)

                if (_this.clu_click_flag[group] == false) {
                    // polygon opacity
                    svg.select('.' + mom_class)
                        .attr('fill-opacity', 0.5)

                    var group = mom_class.substring(7)
                    // dot radius
                    svg.selectAll('.dot' + group)
                        .attr('r', dotRadius)

                    d3.selectAll('._group_' + _this.data.id + group)
                        .attr('opacity', 0)
                }

            }

            function polygon_clicked(event, d) {
                var mom = d3.select(this)
                var mom_class = mom.attr('class')
                var group = mom_class.substring(7)

                if (_this.clu_click_flag[group] == false) {
                    // polygon opacity
                    svg.select('.' + mom_class)
                        .attr('fill-opacity', 0.7)

                    // dot radius
                    svg.selectAll('.dot' + group)
                        .attr('r', selected_dotRadius)

                    // lines
                    d3.selectAll('._group_' + _this.data.id + group)
                        .attr('opacity', 1)

                    // add clicked cluster name to global param clicked_cluster
                    var cluster_name = '._group_' + _this.data.id + group
                    _this.$clicked_cluster.push(cluster_name)

                    // add clicked cluster(mom_class) to local clicked_clus
                    _this.clicked_clus.push(mom_class)
                    console.log('TEMPLATE >> CLUS >> CLICK >> mom class', _this.clicked_clus)

                    // global cf $ table range
                    if (typeof (_this.$glo_table_range[_this.data.id].class) == 'undefined') {
                        _this.$glo_table_range[_this.data.id].class = []
                    }
                    if (_this.$glo_table_range[_this.data.id]['class'].indexOf(group * 1) == -1)
                        _this.$glo_table_range[_this.data.id].class.push(group * 1)
                    console.log('TEMPLATE >> CLUSTER>> CLICK >> class push', _this.$glo_table_range[_this.data.id].class)

                } else {
                    // polygon opacity
                    svg.select('.' + mom_class)
                        .attr('fill-opacity', 0.5)

                    var group = mom_class.substring(7)
                    // dot radius
                    svg.selectAll('.dot' + group)
                        .attr('r', dotRadius)

                    // lines
                    d3.selectAll('._group_' + _this.data.id + group)
                        .attr('opacity', 1)

                    // delete clicked cluster name from global param clicked_cluster
                    var cluster_name = '._group_' + _this.data.id + group
                    var idx = _this.$clicked_cluster.indexOf(cluster_name)
                    _this.$clicked_cluster.splice(idx, 1)

                    // delete clicked cluster(mom_class) to local clicked_clus
                    _this.clicked_clus.splice(mom_class, 1)
                    console.log('TEMPLATE >> CLUS >> CLICK >> mom class', _this.clicked_clus)

                    // global table range
                    _this.$glo_table_range[_this.data.id].class
                        .splice(_this.$glo_table_range[_this.data.id].class.indexOf(group * 1), 1)
                    console.log('TEMPLATE >> CLUSTER>> CLICK >> class splice', _this.$glo_table_range[_this.data.id].class)
                }

                _this.clu_click_flag[group] = !_this.clu_click_flag[group]

            }
            //#endregion

            //#endregion

            //#region plot
            var portrait = svg.append('g')
                .selectAll('g')
                .data(data)
                .enter()
                .append('g')
                .attr("transform", (d, i) => `translate(${left_start + i * delta_len},${margin.top + radius})`)
                .attr("class", (_, i) => "id" + this.data.id + "portrait" + i)

            //#region portraits
            var bgCircle = portrait
                .selectAll('.levels')
                .data(d3.range(1, (axisCircles + 1)).reverse())//[2,1]
                .enter()
                .append('circle')
                .attr('r', (d, i) => radius / axisCircles * d)
                .style("fill", 'none')
                .style("stroke", bgColor)
                .style("fill-opacity", 0.3)
                .attr('class', (d, i) => 'bgCircle' + i)

            var axis = portrait.selectAll('g.axis')
                .data(port_group)
                .enter()
                .append('g')
                .attr('class', 'axis')

            var axis_line = axis
                .append('line')
                .attr("x1", 0)
                .attr("y1", 0)
                .attr("x2", (d, i) => {
                    return radius * 1.1 * Math.cos(angleSlice * i - Math.PI / 2)
                })
                .attr("y2", (d, i) => radius * 1.1 * Math.sin(angleSlice * i - Math.PI / 2))
                .attr("class", "line")
                .style("stroke", bgColor)
                .style("stroke-width", "1px");

            var axis_text = axis
                .append("text")
                .attr("class", "legend")
                .style("font-size", "16px")
                .attr("text-anchor", "middle")
                .attr("font-family", "monospace")
                .attr("dy", "0.35em")
                .attr("x", (d, i) => radius * axisLabelFactor * Math.cos(angleSlice * i - Math.PI / 2))
                .attr("y", (d, i) => radius * axisLabelFactor * Math.sin(angleSlice * i - Math.PI / 2))
                .text(d => d);

            var polygon = portrait.append('g')
                .append('path')
                .attr("d", d => radarLine(d.portrait.map(v => v.value)))
                .attr("fill", (d, i) => {
                    return colors(d.group)
                })
                .attr("fill-opacity", 0.5)
                .attr("stroke", (d, i) => colors(d.group))
                .attr("stroke-width", 1)
                .attr("class", (d, i) => {
                    return "polygon" + d.group
                })
                .on('mouseover', polygon_overed)
                .on('mouseout', polygon_outed)
                .on('click', polygon_clicked)


            var dot = portrait.append('g')
                .selectAll('circle')
                .data(d => {
                    var port = d.portrait
                    port.forEach(element => {
                        element.group = d.group
                    });
                    return port
                })
                .enter()
                .append('circle')
                .attr('r', dotRadius)
                .attr("cx", (d, i) => rScale[i](d.value) * Math.cos(angleSlice * i - Math.PI / 2))
                .attr("cy", (d, i) => rScale[i](d.value) * Math.sin(angleSlice * i - Math.PI / 2))
                .attr("fill", (d, i) => {
                    return colors(d.group)
                })
                .style("fill-opacity", 0.8)
                .attr("class", (d, i) => {
                    return "dot" + d.group
                })
            //#endregion

            //#region num bars
            var num_bars = portrait.append('g')
                .selectAll('rect')
                .data(d => d.num)
                .enter()
                .append('rect')
                .attr('x', d => num_x(d.key))
                .attr('y', d => num_y(d.value) + 5)
                .attr('width', num_width)
                .attr('height', d => num_height - num_y(d.value))
                .attr('fill', d => num_color(d.key))
                .attr('fill-opacity', _this.bar_opacity)
                .attr("transform", (d, i) => `translate(${-1.75 * radius + 10},${radius + delta_port_num})`)
                .attr('stroke', d => num_color(d.key))
                .attr('stroke-width', 1)
                .attr('stroke-opacity', 1)
                .attr('class', d => 'portid' + _this.data.id + d.key)

            // var num_bars_text = num_bars
            //     .append("text")
            //     .attr("class", "bar_text")
            //     .style("font-size", "14px")
            //     .attr("text-anchor", "middle")
            //     .attr("font-family", "monospace")
            //     .attr("fill", "#333533")
            //     .text(d => d.value)
            //     .attr('opacity', 1)

            var num_bars_text = portrait.append('g')
                .selectAll('text')
                .data(d => d.num)
                .enter()
                .append('text')
                .attr('x', d => num_x(d.key) + num_width / 2)
                .attr('y', num_height + 25)
                .attr("transform", (d, i) => `translate(${-1.75 * radius + 10},${radius + delta_port_num})`)
                .text(d => d.value)
                .attr("class", "bar_text")
                .style("font-size", "14px")
                .attr("text-anchor", "middle")
                .attr("font-family", "monospace")


            //#endregion

            //#endregion
        },
        plot_pred(id) {
            let _this = this
            //#region dealing with data
            var data = _this.data.pred

            //#region basic layout config
            const margin = { top: 40, right: 10, bottom: 36, left: 10 }  // focus
            const svg = d3.select('#' + id)
            svg.selectAll('*').remove()
            var width = 223, height = 220
            var width = width - margin.right - margin.left
            var height = height - margin.top - margin.bottom

            //#region params for functions, including range,...
            //get range group of x-axis
            var Xrange = d3.map(data, d => d.date).keys()
            var Xcount = []
            Xrange.forEach((element, i) => {
                Xcount.push(i)
            });

            //get range of y-axis based on min and max of three status of churn
            var churn_range = d3.extent(data, d => [d.high, d.med, d.low, d.churn])
            var Yrange = [0, d3.max(churn_range[1])]

            //param group
            var status_group = ['high', 'med', 'low', 'churn']

            //color
            var color_group = ['#676b54', '#d27d1c', '#4881b3', '#b23722']
            // var color_group = ['#666a53', '#d27d1c', '#72a6c1', '#b23722']
            //#endregion

            //#region functions
            var x = d3.scaleBand()
                .domain(Xrange)
                .range([0, width])
                .padding(0.1)

            var y = d3.scaleLinear()
                .domain(Yrange)
                .range([height, margin.top])

            var group_x = d3.scaleBand()
                .domain(status_group)
                .range([0, x.bandwidth()])
                .padding(0.4)

            var color = d3.scaleOrdinal()
                .domain(status_group)
                .range(color_group)

            //#region mouse functions

            function overed(event, d) {
                var mom = d3.select(this)
                var mom_class = mom.attr('class')
                svg.select('.' + mom_class)
                    .attr('fill-opacity', 1)

                console.log('TEMPLATE >> PRED >> HOVER', _this.$glo_table_range[_this.data.id]['pred'])
            }

            function outed(event, d) {
                var mom = d3.select(this)
                var mom_class = mom.attr('class')

                svg.select('.' + mom_class)
                    .attr('fill-opacity', 0.8)
            }

            function clicked(event, d) {
                var mom = d3.select(this)
                var mom_class = mom.attr('class')
                // console.log(mom_class)

                var status = mom_class.substring(7)

                if (_this.pred_click_flag[status] == false) {
                    svg.select('.' + mom_class)
                        .attr('fill-opacity', 1)
                        .attr('stroke-width', 3)

                    // add clicked pred to clicked_pred
                    var clicked_pred = '.' + mom_class
                    _this.clicked_pred.push(clicked_pred)
                    console.log('TEMPLATE >> PRED >> clicked pred', _this.clicked_pred)

                    // global table range
                    if (typeof (_this.$glo_table_range[_this.data.id]['pred']) == 'undefined') {
                        _this.$glo_table_range[_this.data.id]['pred'] = []
                    }
                    if (_this.$glo_table_range[_this.data.id]['pred'].indexOf(_this.status_idx[status] * 1) == -1)
                        _this.$glo_table_range[_this.data.id]['pred'].push(_this.status_idx[status] * 1)
                    console.log('TEMPLATE >> PRED >> CLICK >> pred push', _this.$glo_table_range)
                }
                else {
                    svg.select('.' + mom_class)
                        .attr('fill-opacity', 0.8)
                        .attr('stroke-width', 1)

                    // delete clicked pred name from clicked pred
                    var clicked_pred = '.' + mom_class
                    _this.clicked_pred.splice(clicked_pred, 1)
                    console.log('TEMPLATE >> PRED >> clicked pred', _this.clicked_pred)

                    // global table range
                    _this.$glo_table_range[_this.data.id].pred
                        .splice(_this.$glo_table_range[_this.data.id]['pred'].indexOf(_this.status_idx[status] * 1), 1)

                    console.log('TEMPLATE >> PRED >> CLICK >> pred splice', _this.$glo_table_range)
                }

                _this.pred_click_flag[status] = !_this.pred_click_flag[status]
            }

            //#endregion

            //#endregion

            //#region chart
            var time_line_g = svg.append('g')

            var daytext = time_line_g.append('g')
                .append('text')
                .text('Day ' + _this.data.id)
                .attr('transform', (d, i) =>
                    `translate(${4},${19})`)
                .attr('text-anchor', 'start')
                .style('font-size', 20)
                .attr('opacity', 1)

            var daygroup = time_line_g.append('g')
                .selectAll('g')
                .data(data)
                .enter()
                .append('g')
                .attr("transform", (d, i) => {
                    return `translate(${margin.left + x(d.date)},${margin.top})`
                })

            var bars = daygroup.append('g')
                .selectAll('rect')
                .data(d => {
                    var test = status_group.map(function (key) {
                        return {
                            key: key,
                            value: d[key]
                        }
                    })
                    return test
                })
                .enter()
                .append('rect')
                .attr('x', d => group_x(d.key))
                .attr('y', d => y(d.value))
                .attr('width', group_x.bandwidth())
                .attr('height', d => height - y(d.value))
                .attr('fill', d => color(d.key))
                .attr('fill-opacity', _this.bar_opacity)
                .attr('stroke', d => color(d.key))
                .attr('stroke-width', 1)
                .attr('stroke-opacity', 1)
                .attr('class', d => 'pred' + 'id' + _this.data.id + d.key)
                .on('mouseover', overed)
                .on('mouseout', outed)
                .on('click', clicked)

            // text
            var texts = daygroup.append('g')
                .selectAll('text')
                .data(d => {
                    var test = status_group.map(function (key) {
                        return {
                            key: key,
                            value: d[key]
                        }
                    })
                    return test
                })
                .enter()
                .append('text')
                .text(d => {
                    return d.value
                })
                .attr('transform', (d, i) =>
                    `translate(${group_x(d.key) + group_x.bandwidth() / 2},
                     ${y(d.value) - 10})`)
                .attr('text-anchor', 'middle')
                .style('font-size', 14)
                .attr('opacity', 1)

            // text
            var pred_texts = daygroup.append('g')
                .selectAll('text')
                .data(d => {
                    var test = status_group.map(function (key) {
                        return {
                            key: key,
                            value: d[key]
                        }
                    })
                    return test
                })
                .enter()
                .append('text')
                .text(d => {
                    return d.key
                })
                .attr('transform', (d, i) =>
                    `translate(${group_x(d.key) + group_x.bandwidth() / 2},
                             ${height+25})`)
                .attr('text-anchor', 'middle')
                .style('font-size', 14)
                .attr('opacity', 1)

            // bars.append('g')
            //     .append('text')
            //     .text(d =>{
            //         console.log(d)
            //         return d.value
            //     })
            //     .attr('transform', (d, i) =>
            //         `translate(${0}, ${0})`)
            //     .attr('text-anchor', 'middle')
            //     // .style('font-family', 'CormorantSemiBold')
            //     .style('font-size', 12)
            //     .attr('opacity', 1)

            //#endregion
        },
        async plot_counterfactual(id) {
            let _this = this
            var svg = d3.select('#' + id)
            svg.selectAll('*').remove()

            const margin = { top: 30, right: 30, bottom: 30, left: 50 }
            var width = 1616, height = 548
            var boxWidth = 5
            var button_wid = 50, button_hei = 15
            //#endregion

            //#region data
            var data = [
                {
                    name: 'school',
                    id: 0,
                    split_num: 5,
                    count_range: [0, 293],
                    counter_count_range: [0, 99],
                    value_range: [3, 9],
                    counter_values: [
                        {
                            range: [3, 5],
                            name: 'school',
                            detail: [
                                { count_range: [0, 99], key: 'high', split_num: 5, value: 4, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'med', split_num: 5, value: 0, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'low', split_num: 5, value: 82, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'churn', split_num: 5, value: 13, value_range: [3, 9] },
                            ],
                        },
                        {
                            range: [5, 6],
                            name: 'school',
                            detail: [
                                { count_range: [0, 99], key: 'high', split_num: 5, value: 3, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'med', split_num: 5, value: 0, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'low', split_num: 5, value: 13, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'churn', split_num: 5, value: 3, value_range: [3, 9] },
                            ],
                        },
                        {
                            range: [6, 7],
                            name: 'school',
                            detail: [
                                { count_range: [0, 99], key: 'high', split_num: 5, value: 3, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'med', split_num: 5, value: 0, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'low', split_num: 5, value: 8, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'churn', split_num: 5, value: 11, value_range: [3, 9] },
                            ],
                        },
                        {
                            range: [7, 8],
                            name: 'school',
                            detail: [
                                { count_range: [0, 99], key: 'high', split_num: 5, value: 1, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'med', split_num: 5, value: 0, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'low', split_num: 5, value: 17, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'churn', split_num: 5, value: 5, value_range: [3, 9] },
                            ],
                        },
                        {
                            range: [8, 9],
                            name: 'school',
                            detail: [
                                { count_range: [0, 99], key: 'high', split_num: 5, value: 1, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'med', split_num: 5, value: 0, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'low', split_num: 5, value: 10, value_range: [3, 9] },
                                { count_range: [0, 99], key: 'churn', split_num: 5, value: 6, value_range: [3, 9] },
                            ],
                        },
                    ],
                    band_domain: [3, 5, 6, 7, 8, 9],
                    values: [
                        {
                            range: [3, 5],
                            name: 'school',
                            detail: [
                                { count_range: [0, 293], key: 'high', split_num: 5, value: 2, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'med', split_num: 5, value: 38, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'low', split_num: 5, value: 291, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'churn', split_num: 5, value: 37, value_range: [3, 9] },
                            ],
                        },
                        {
                            range: [5, 6],
                            name: 'school',
                            detail: [
                                { count_range: [0, 293], key: 'high', split_num: 5, value: 0, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'med', split_num: 5, value: 129, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'low', split_num: 5, value: 156, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'churn', split_num: 5, value: 163, value_range: [3, 9] },
                            ],
                        },
                        {
                            range: [6, 7],
                            name: 'school',
                            detail: [
                                { count_range: [0, 293], key: 'high', split_num: 5, value: 1, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'med', split_num: 5, value: 190, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'low', split_num: 5, value: 190, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'churn', split_num: 5, value: 189, value_range: [3, 9] },
                            ],
                        },
                        {
                            range: [7, 8],
                            name: 'school',
                            detail: [
                                { count_range: [0, 293], key: 'high', split_num: 5, value: 0, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'med', split_num: 5, value: 211, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'low', split_num: 5, value: 210, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'churn', split_num: 5, value: 123, value_range: [3, 9] },
                            ],
                        },
                        {
                            range: [8, 9],
                            name: 'school',
                            detail: [
                                { count_range: [0, 293], key: 'high', split_num: 5, value: 1, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'med', split_num: 5, value: 4, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'low', split_num: 5, value: 80, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'churn', split_num: 5, value: 84, value_range: [3, 9] },
                            ],
                        }
                    ],
                    quartiles: [5, 6, 7],
                    counter_quartiles: [4, 6, 7],
                    real_value_range: [3, 8],
                    counter_value_range: [3, 8],
                },
            ]
            data = _this.data.cf
            console.log('GROUP TEMPLATE >> cf data', data)

            //#endregion

            //#region params for functions, including range,...
            //feature width
            // var feat_width = 177, feat_height = 60, padding = 20
            // var feat_width = 270, feat_height = 55, padding = 50
            var feat_width = 350, feat_height = 55, padding = 50

            var counter_y_padding = 25, y_padding = 20
            var xaxis_padding = 10
            //param group
            var status_group = ['high', 'med', 'low', 'churn']
            //color
            var color_group = ['#666a53', '#d27d1c', '#72a6c1', '#B64330']
            var bar_opacity = 0.7
            //#endregion

            //#region functions
            Number.prototype.toFixed = function (s) {
                var times = Math.pow(10, s);
                //如果是正数，则+0.5，是负数，则-0.5
                const adjust = this >= 0 ? 0.5 : -0.5;
                var des = this * times + adjust;
                des = parseInt(des, 10) / times;
                return des + '';
            }

            function split(range, split_num) {
                var min = range[0], max = range[1]
                var delta = (max - min) / split_num
                var band_domain = []
                for (var i = 0; i < split_num + 1; i++) {
                    band_domain.push((min + i * delta).toFixed(6))
                }
                return band_domain
            }

            function x(band_domain) {
                return d3.scaleBand()
                    .domain(band_domain)
                    .range([0, feat_width])
                    .padding(0.2)
            }

            function box_x(domain, range) {
                return d3.scaleLinear()
                    .domain(domain)
                    .range(range)
            }

            function box_get_domain(d) {
                var x1 = x(d.band_domain)(d.value_range[0])
                var x2 = x(d.band_domain)(d.value_range[1])

                var move_x = x(d.band_domain).step() / 2
                var band_x = x(d.band_domain).bandwidth() * 0.2 / 2
                x1 = x1 + move_x - band_x
                x2 = x2 + move_x - band_x

                // validation success
                var range = [x1, x2]
                return box_x(d.value_range, range)
            }

            function x_reverse(band_domain) {
                return d3.scaleQuantize()
                    .domain([0, feat_width])
                    .range(band_domain)
            }

            function y(range) {
                return d3.scaleLinear()
                    .domain(range)
                    .range([feat_height, margin.top])
            }

            function counter_y(range) {
                return d3.scaleLinear()
                    .domain(range)
                    .range([feat_height + counter_y_padding, 2 * feat_height + counter_y_padding - margin.top])
            }

            function group_x(band_domain) {
                return d3.scaleBand()
                    .domain(status_group)
                    .range([0, x(band_domain).bandwidth()])
                    .padding(0.25)
            }

            var color = d3.scaleOrdinal()
                .domain(status_group)
                .range(color_group)

            //#region mouse function
            function button_clicked(event, d) {
                var feat_name = event.name
                var idx = _this.feat_idx[feat_name]

                // flag = !flag
                _this.change_flag[feat_name] = !_this.change_flag[feat_name]

                // update change in range sent to backend
                if (_this.change_flag[feat_name] == true) {
                    _this.change.change.push(feat_name)
                    _this.change_button.push(feat_name)

                    // global cf range
                    if (typeof (_this.$glo_cf_range[_this.data.id]['change']) == 'undefined') {
                        _this.$glo_cf_range[_this.data.id]['change'] = []
                    }
                    _this.$glo_cf_range[_this.data.id]['change'].push(feat_name)
                    console.log('TEMPLATE >> CF_RAW >> BRUSH >> glo cf range', _this.$glo_cf_range)
                    // console.log('TEMPLATE >> CF >> change button', _this.change_button)
                } else {
                    _this.change.change
                        .splice(_this.change.change.indexOf(feat_name), 1)
                    _this.change_button
                        .splice(_this.change_button.indexOf(feat_name), 1)

                    // global cf range
                    _this.$glo_cf_range[_this.data.id].change
                        .splice(_this.$glo_cf_range[_this.data.id]['change'].indexOf(feat_name), 1)
                    console.log('TEMPLATE >> CF_RAW >> BRUSH >> glo cf range', _this.$glo_cf_range)
                }

                // change button style
                if (_this.change_flag[feat_name] == true) {
                    svg.select('.button_' + feat_name)
                        .attr('fill-opacity', 0.5)
                } else {
                    svg.select('.button_' + feat_name)
                        .attr('fill-opacity', 0)
                }
            }

            function button_overed(event, d) {
                var mom = d3.select(this)
                var mom_class = mom.attr("class")
                var feat_name = event.name

                if (_this.change_flag[feat_name] == false) {
                    svg.select('.' + mom_class)
                        .attr('fill-opacity', 0.2)
                }

            }

            function button_outed(event, d) {
                var mom = d3.select(this)
                var mom_class = mom.attr("class")
                var feat_name = event.name

                if (_this.change_flag[feat_name] == false) {
                    svg.select('.' + mom_class)
                        .attr('fill-opacity', 0)
                }
            }
            //#endregion

            //#endregion

            //#region plot
            //plot rect
            var grect = svg.append('g')
                .append('rect')
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', width)
                .attr('height', height)
                .attr('fill', '#ffffff')
                .attr('opacity', 1)

            var feat_g = svg.append('g')
                .selectAll('g')
                .data(data)
                .enter()
                .append('g')
                .attr("transform", (d, i) => {
                    var x_count, y_count
                    if (i < 4) {
                        x_count = i
                        y_count = 0
                    } else if (i >= 4 && i < 8) {
                        x_count = i - 4
                        y_count = 1
                    } else if (i >= 8 && i < 12) {
                        x_count = i - 8
                        y_count = 2
                    }
                    else {
                        x_count = i - 12
                        y_count = 3
                    }

                    return `translate(${margin.left + x_count * (feat_width + padding)},
                        ${y_count * (feat_height * 2 + y_padding)})`

                })
                .attr('class', d => d.name)

            // plot text
            var feat_text = feat_g.append('g')
                .append('text')
                .attr('text-anchor', 'middle')
                .style('font-size', '18px')
                .style('font-family', "KaiseiOptiRegular")
                .text(d => d.name)
                .attr('transform', `translate(${feat_width / 2}, ${margin.top})`)

            // #region buttons
            var button_rect = feat_g.append('g')
                .append('rect')
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', button_wid)
                .attr('height', button_hei)
                .attr('fill', '#B64330')
                .attr('fill-opacity', 0)
                .attr('transform', `translate(${0}, ${margin.top / 1.5})`)
                .attr('stroke', '#B64330')
                .attr('stroke-opacity', 0.5)
                .attr('rx', 3)
                .attr('class', d => 'button_' + d.name)
                .on('click', button_clicked)
                .on('mouseover', button_overed)
                .on('mouseout', button_outed)


            // #endregion

            // #region real bars
            var bar_group = feat_g.append('g')
                .selectAll('g')
                .data(d => {
                    var val = []
                    d.values.forEach(ele => {
                        ele.band_domain = d.band_domain
                    })
                    return d.values
                })
                .enter()
                .append('g')
                .attr("transform", (d, i) => {
                    var move_x = x(d.band_domain).step() / 2
                    return `translate(${x(d.band_domain)(d.range[0]) + move_x}
                        ,${margin.top})`
                })
                .attr('class', (d, i) => 'day' + id + '_' + d.name + '_' + 'valuegroup' + '_' + i)

            var bars = bar_group.selectAll('rect')
                .data(d => {
                    d.detail.forEach(de => {
                        de.band_domain = d.band_domain
                    })
                    return d.detail
                })
                .enter()
                .append('rect')
                .attr('x', d => {
                    return group_x(d.band_domain)(d.key)
                })
                .attr('y', d => {
                    // console.log(d)
                    return y(d.count_range)(d.value)
                })
                .attr('width', d => group_x(d.band_domain).bandwidth())
                .attr('height', d => feat_height - y(d.count_range)(d.value))
                .attr('fill', d => color(d.key))
                .attr('opacity', bar_opacity)
                .attr('class', d => {
                    return 'day' + id + '_' + d.key
                })
            //#endregion

            // #region counterfactual bars
            var counter_bar_group = feat_g.append('g')
                .selectAll('g')
                .data(d => {
                    if (d.counter_values[0] == 0 && d.counter_values[1] == 0) {
                        return []
                    } else {
                        d.counter_values.forEach(ele => {
                            ele.band_domain = d.band_domain
                        })
                        return d.counter_values
                    }
                })
                .enter()
                .append('g')
                .attr("transform", (d, i) => {
                    var move_x = x(d.band_domain).step() / 2
                    return `translate(${x(d.band_domain)(d.range[0]) + move_x}
                        ,${margin.top})`
                })
                .attr('class', (d, i) => 'day' + id + '_' + d.name + '_' + 'countergroup' + '_' + i)

            var counter_bars = counter_bar_group.selectAll('rect')
                .data(d => {
                    d.detail.forEach(de => {
                        de.band_domain = d.band_domain
                    })
                    return d.detail
                })
                .enter()
                .append('rect')
                .attr('x', d => {
                    return group_x(d.band_domain)(d.key)
                })
                .attr('y', d => {
                    return feat_height + counter_y_padding
                })
                .attr('width', d => group_x(d.band_domain).bandwidth())
                .attr('height', d => counter_y(d.count_range)(d.value) - feat_height - counter_y_padding)
                .attr('fill', d => color(d.key))
                .attr('opacity', bar_opacity)
                .attr('class', d => {
                    return 'day' + id + '_' + d.key + 'counter'
                })

            //#endregion

            // #region box plot
            var box = feat_g.append('g')
                .attr("transform", (d, i) => {
                    return `translate(${0}, ${margin.top + feat_height + counter_y_padding / 3})`
                })

            box.append('line')
                .attr("class", "vertLine")
                .attr("stroke", "#C0C0C0")
                .attr('stroke-width', '1px')
                .style("width", 40)
                .attr("x1", d => {
                    var x1 = box_get_domain(d)(d.real_value_range[0])
                    return x1
                })
                .attr("x2", d => {
                    var x2 = box_get_domain(d)(d.real_value_range[1])
                    return x2
                })
                .attr("y1", 0)
                .attr("y2", 0);

            box
                .append('rect')
                .attr("class", "box")
                .attr("y", -boxWidth / 2)
                .attr("x", d => {
                    var xpos = box_get_domain(d)(d.quartiles[1])
                    return xpos
                })
                .attr("width", d => {
                    var wid = box_get_domain(d)(d.quartiles[2]) - box_get_domain(d)(d.quartiles[1])
                    return wid
                })
                .attr("height", boxWidth)
                .attr("stroke", "#808080")
                .attr('stroke-width', 0.5)
                .style("fill", "rgb(255, 255, 255)")
                .style("fill-opacity", 0.7);

            box
                .selectAll("verticalLine")
                .data(d => [[d, d.real_value_range[0]],
                [d, d.quartiles[1]],
                [d, d.real_value_range[1]]])
                .enter()
                .append('line')
                .attr("class", "verticalLine")
                .attr("stroke", "#808080")
                .attr('stroke-width', '1px')
                .style("width", 10)
                .attr("y1", -boxWidth / 2)
                .attr("y2", +boxWidth / 2)
                .attr("x1", d => box_get_domain(d[0])(d[1]))
                .attr("x2", d => box_get_domain(d[0])(d[1]));

            // #endregion

            // #region box plot for counterfactuals
            var box_counter = feat_g.append('g')
                .attr("transform", (d, i) => {
                    return `translate(${0}, ${margin.top + feat_height + counter_y_padding / 3 * 2})`
                })

            box_counter.append('line')
                .attr("class", "vertLine")
                .attr("stroke", "#C0C0C0")
                .attr('stroke-width', '1px')
                .style("width", 40)
                .attr("x1", d => {
                    var x1 = box_get_domain(d)(d.counter_value_range[0])
                    return x1
                })
                .attr("x2", d => {
                    var x2 = box_get_domain(d)(d.counter_value_range[1])
                    return x2
                })
                .attr("y1", 0)
                .attr("y2", 0);

            box_counter
                .append('rect')
                .attr("class", "box")
                .attr("y", -boxWidth / 2)
                .attr("x", d => {
                    var xpos = box_get_domain(d)(d.counter_quartiles[0])
                    return xpos
                })
                .attr("width", d => {
                    var wid = box_get_domain(d)(d.counter_quartiles[2]) - box_get_domain(d)(d.counter_quartiles[0])
                    return wid
                })
                .attr("height", boxWidth)
                .attr("stroke", "#808080")
                .attr('stroke-width', 0.5)
                .style("fill", "rgb(255, 255, 255)")
                .style("fill-opacity", 0.7);

            box_counter
                .selectAll("verticalLine")
                .data(d => [[d, d.counter_value_range[0]],
                [d, d.counter_quartiles[1]],
                [d, d.counter_value_range[1]]])
                .enter()
                .append('line')
                .attr("class", "verticalLine")
                .attr("stroke", "#808080")
                .attr('stroke-width', '1px')
                .style("width", 10)
                .attr("y1", -boxWidth / 2)
                .attr("y2", +boxWidth / 2)
                .attr("x1", d => box_get_domain(d[0])(d[1]))
                .attr("x2", d => box_get_domain(d[0])(d[1]));

            // #endregion

            // #region axes
            //x-axis
            var xaxis = feat_g.append('g')
                .attr("transform", `translate(${0},${margin.top + xaxis_padding})`)
                .each(function (d, i) {
                    var range = d.value_range
                    var name = d.name
                    var copy_domain_dict = _this.deepCopy(d.band_domain)
                    var band_domain = []
                    for (var key in copy_domain_dict) {
                        band_domain.push(copy_domain_dict[key])
                    }
                    // console.log('band_domain', band_domain)
                    var me = d3.select(this)

                    if (name == 'deltatime' || name == 'tran_deltatime') {
                        band_domain.forEach((ba, i) => {
                            band_domain[i] = Math.ceil(band_domain[i] / 1000)
                        })
                    }

                    if (name == 'pr' || name == 'cn') {
                        me.call(d3.axisBottom(x(band_domain))
                            .ticks(5)
                            .tickSize(5)
                            .tickSizeOuter(0)
                            .tickFormat(d3.format("0.5f")))
                            .attr('class', 'axis')
                            .attr('id', d => 'x_axis_' + name)
                    } else {
                        me.call(d3.axisBottom(x(band_domain))
                            .ticks(5)
                            .tickSize(5)
                            .tickSizeOuter(0)
                            .tickFormat(d3.format("0.1f")))
                            .attr('class', 'axis')
                            .attr('id', 'x_axis_' + name)
                    }
                })

            //real y-axis
            feat_g.append('g')
                .attr("transform", `translate(${0},${margin.top})`)
                .each(function (d, i) {
                    var range = d.count_range
                    var me = d3.select(this)
                    me.call(d3.axisLeft(y(range))
                        .ticks(2)
                        .tickSize(2)
                        .tickSizeOuter(0))
                        .attr('class', 'axis')
                })

            //counterfactual y-axis
            feat_g.append('g')
                .attr("transform", `translate(${0},${margin.top})`)
                .each(function (d, i) {
                    var range = d.counter_count_range
                    var me = d3.select(this)
                    me.call(d3.axisLeft(counter_y(range))
                        .ticks(2)
                        .tickSize(2)
                        .tickSizeOuter(0))
                        .attr('class', 'axis')
                })

            //change text size
            svg.selectAll('.axis')
                .selectAll('text')
                .style('font-size', 11)
                .style('opacity', 1)

            //#endregion

            // #region brush
            function brush(d) {
                return d3.brushX()
                    .extent([
                        [2, margin.top * 2], [feat_width, feat_height + margin.top - 1]
                    ])
                    .on('brush', brushed)
            }

            const defaultSelection = [2, feat_width];

            function brushed(d) {
                var range = d.value_range
                var name = d.name
                var band_domain = d.band_domain
                const selection = d3.event.selection;
                // console.log(selection)
                var x_begin = x_reverse(band_domain)(selection[0])
                var x_end = x_reverse(band_domain)(selection[1])

                var idx = _this.feat_idx[name]
                _this.range[idx][name] = [x_begin, x_end]

                if (typeof (_this.$glo_cf_range[_this.data.id]) == 'undefined') {
                    _this.$glo_cf_range[_this.data.id] = {}
                    _this.$glo_table_range[_this.data.id] = {}
                }
                _this.$glo_cf_range[_this.data.id][name] = [x_begin, x_end]
                _this.$glo_table_range[_this.data.id][name] = [x_begin, x_end]
                // console.log('TEMPLATE >> CF_RAW >> BRUSH >> glo cf range', _this.$glo_cf_range)
            }

            const gBrush = feat_g.append('g')
                .attr('class', 'counter_brush')
                .each(function (d) {
                    var me = d3.select(this)
                    me.call(brush(d))
                        .call(brush(d).move, defaultSelection)
                })

            //custom handlers
            //handle group
            const ghandles = gBrush.selectAll('g.handles')
                .data(['handle--o', 'handle--e'])
                .enter()
                .append('g')
                .attr('class', 'steelblue')
                .attr('transform', d => {
                    const x = d == 'handle--o' ? 0 : width / 15;
                    return `translate(${x}, 0)`;
                });

            //get brush
            var b = feat_g.selectAll('.brush')

            // change brush color
            // svg.selectAll('.selection')
            //     .attr('fill', '#72A6C1')
            //     .attr('opacity', 0.5)
            //     .attr('stroke', '#72A6C1')

            svg.selectAll('.selection')
                .attr('fill', '#B64330')
                .attr('opacity', 0.4)
                .attr('stroke', '#B64330')

            //#endregion

            //#endregion

        },
        plot_counterfactual_raw(id) {
            let _this = this
            var svg = d3.select('#' + id)
            svg.selectAll('*').remove()

            const margin = { top: 30, right: 30, bottom: 30, left: 50 }
            var width = 1616, height = 548
            var boxWidth = 10
            var button_wid = 50, button_hei = 15
            //#endregion

            //#region data
            var data = [
                {
                    name: 'school',
                    id: 0,
                    split_num: 5,
                    count_range: [0, 293],
                    value_range: [3, 9],
                    band_domain: [3, 5, 6, 7, 8, 9],
                    values: [
                        {
                            range: [3, 5],
                            name: 'school',
                            detail: [
                                { count_range: [0, 293], key: 'high', split_num: 5, value: 2, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'med', split_num: 5, value: 38, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'low', split_num: 5, value: 291, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'churn', split_num: 5, value: 37, value_range: [3, 9] },
                            ],
                        },
                        {
                            range: [5, 6],
                            name: 'school',
                            detail: [
                                { count_range: [0, 293], key: 'high', split_num: 5, value: 0, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'med', split_num: 5, value: 129, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'low', split_num: 5, value: 156, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'churn', split_num: 5, value: 163, value_range: [3, 9] },
                            ],
                        },
                        {
                            range: [6, 7],
                            name: 'school',
                            detail: [
                                { count_range: [0, 293], key: 'high', split_num: 5, value: 1, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'med', split_num: 5, value: 190, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'low', split_num: 5, value: 190, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'churn', split_num: 5, value: 189, value_range: [3, 9] },
                            ],
                        },
                        {
                            range: [7, 8],
                            name: 'school',
                            detail: [
                                { count_range: [0, 293], key: 'high', split_num: 5, value: 0, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'med', split_num: 5, value: 211, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'low', split_num: 5, value: 210, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'churn', split_num: 5, value: 123, value_range: [3, 9] },
                            ],
                        },
                        {
                            range: [8, 9],
                            name: 'school',
                            detail: [
                                { count_range: [0, 293], key: 'high', split_num: 5, value: 1, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'med', split_num: 5, value: 4, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'low', split_num: 5, value: 80, value_range: [3, 9] },
                                { count_range: [0, 293], key: 'churn', split_num: 5, value: 84, value_range: [3, 9] },
                            ],
                        }
                    ],
                    quartiles: [5, 6, 7],
                    real_value_range: [3, 8],
                },
            ]
            data = _this.data.cf_raw
            console.log('TEMPLATE >> CF ROW >> data', data)

            //#endregion

            //#region params for functions, including range,...
            // feature width
            // var feat_width = 177, feat_height = 60, padding = 20
            // var feat_width = 270, feat_height = 55, padding = 50
            var feat_width = 350, feat_height = 55, padding = 50

            var counter_y_padding = 25, y_padding = 20
            var xaxis_padding = 10
            //param group
            var status_group = ['high', 'med', 'low', 'churn']
            //color
            var color_group = ['#666a53', '#d27d1c', '#72a6c1', '#B64330']
            var bar_opacity = 0.6
            //#endregion

            //#region functions
            Number.prototype.toFixed = function (s) {
                var times = Math.pow(10, s);
                //如果是正数，则+0.5，是负数，则-0.5
                const adjust = this >= 0 ? 0.5 : -0.5;
                var des = this * times + adjust;
                des = parseInt(des, 10) / times;
                return des + '';
            }

            function x(band_domain) {
                return d3.scaleBand()
                    .domain(band_domain)
                    .range([0, feat_width])
                    .padding(0.2)
            }

            function box_x(domain, range) {
                return d3.scaleLinear()
                    .domain(domain)
                    .range(range)
            }

            function box_get_domain(d) {
                var x1 = x(d.band_domain)(d.value_range[0])
                var x2 = x(d.band_domain)(d.value_range[1])

                var move_x = x(d.band_domain).step() / 2
                var band_x = x(d.band_domain).bandwidth() * 0.2 / 2
                x1 = x1 + move_x - band_x
                x2 = x2 + move_x - band_x

                // validation success
                var range = [x1, x2]
                return box_x(d.value_range, range)
            }

            function x_reverse(band_domain) {
                return d3.scaleQuantize()
                    .domain([0, feat_width])
                    .range(band_domain)
            }

            function y(range) {
                return d3.scaleLinear()
                    .domain(range)
                    .range([feat_height, margin.top])
            }

            function group_x(band_domain) {
                return d3.scaleBand()
                    .domain(status_group)
                    .range([0, x(band_domain).bandwidth()])
                    .padding(0.2)
            }

            var color = d3.scaleOrdinal()
                .domain(status_group)
                .range(color_group)

            //#region mouse function
            function button_clicked(event, d) {
                var feat_name = event.name
                var idx = _this.feat_idx[feat_name]

                // flag = !flag
                _this.change_flag[feat_name] = !_this.change_flag[feat_name]

                // update change in range sent to backend
                if (_this.change_flag[feat_name] == true) {
                    // change
                    _this.change.change.push(feat_name)
                    // change button
                    _this.change_button.push(feat_name)

                    // global cf range
                    if (typeof (_this.$glo_cf_range[_this.data.id]['change']) == 'undefined') {
                        _this.$glo_cf_range[_this.data.id]['change'] = []
                    }
                    _this.$glo_cf_range[_this.data.id]['change'].push(feat_name)
                    console.log('TEMPLATE >> CF_RAW >> BRUSH >> glo cf range', _this.$glo_cf_range)
                } else {
                    // change
                    _this.change.change
                        .splice(_this.change.change.indexOf(feat_name), 1)
                    // change button
                    _this.change_button
                        .splice(_this.change_button.indexOf(feat_name), 1)

                    // global cf range
                    _this.$glo_cf_range[_this.data.id].change
                        .splice(_this.$glo_cf_range[_this.data.id]['change'].indexOf(feat_name), 1)
                    console.log('TEMPLATE >> CF_RAW >> BRUSH >> glo cf range', _this.$glo_cf_range)
                }

                // console.log(_this.change)

                // change button style
                if (_this.change_flag[feat_name] == true) {
                    svg.select('.button_' + feat_name)
                        .attr('fill-opacity', 0.5)
                } else {
                    svg.select('.button_' + feat_name)
                        .attr('fill-opacity', 0)
                }
            }

            function button_overed(event, d) {
                var mom = d3.select(this)
                var mom_class = mom.attr("class")
                var feat_name = event.name

                if (_this.change_flag[feat_name] == false) {
                    svg.select('.' + mom_class)
                        .attr('fill-opacity', 0.2)
                }

            }

            function button_outed(event, d) {
                var mom = d3.select(this)
                var mom_class = mom.attr("class")
                var feat_name = event.name

                if (_this.change_flag[feat_name] == false) {
                    svg.select('.' + mom_class)
                        .attr('fill-opacity', 0)
                }
            }
            //#endregion

            //#endregion

            //#region plot
            //plot rect
            var grect = svg.append('g')
                .append('rect')
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', width)
                .attr('height', height)
                .attr('fill', '#ffffff')
                .attr('opacity', 1)

            var feat_g = svg.append('g')
                .selectAll('g')
                .data(data)
                .enter()
                .append('g')
                .attr("transform", (d, i) => {
                    var x_count, y_count
                    if (i < 4) {
                        x_count = i
                        y_count = 0
                    } else if (i >= 4 && i < 8) {
                        x_count = i - 4
                        y_count = 1
                    } else if (i >= 8 && i < 12) {
                        x_count = i - 8
                        y_count = 2
                    }
                    else {
                        x_count = i - 12
                        y_count = 3
                    }

                    return `translate(${margin.left + x_count * (feat_width + padding)},
                        ${y_count * (feat_height * 2 + y_padding)})`

                })
                .attr('class', d => d.name)

            // plot text
            var feat_text = feat_g.append('g')
                .append('text')
                .attr('text-anchor', 'middle')
                .style('font-size', '18px')
                .style('font-family', "KaiseiOptiRegular")
                .text(d => d.name)
                .attr('transform', `translate(${feat_width / 2}, ${margin.top})`)

            // #region buttons
            var button_rect = feat_g.append('g')
                .append('rect')
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', button_wid)
                .attr('height', button_hei)
                .attr('fill', '#B64330')
                .attr('fill-opacity', 0)
                .attr('transform', `translate(${0}, ${margin.top / 1.6})`)
                .attr('stroke', '#B64330')
                .attr('stroke-opacity', 0.5)
                .attr('rx', 3)
                .attr('class', d => 'button_' + d.name)
                .on('click', button_clicked)
                .on('mouseover', button_overed)
                .on('mouseout', button_outed)


            // #endregion

            // #region real bars
            var bar_group = feat_g.append('g')
                .selectAll('g')
                .data(d => {
                    var val = []
                    d.values.forEach(ele => {
                        ele.band_domain = d.band_domain
                    })
                    return d.values
                })
                .enter()
                .append('g')
                .attr("transform", (d, i) => {
                    var move_x = x(d.band_domain).step() / 2
                    return `translate(${x(d.band_domain)(d.range[0]) + move_x}
                        ,${margin.top})`
                })
                .attr('class', (d, i) => 'day' + id + '_' + d.name + '_' + 'valuegroup' + '_' + i)

            var bars = bar_group.selectAll('rect')
                .data(d => {
                    d.detail.forEach(de => {
                        de.band_domain = d.band_domain
                    })
                    return d.detail
                })
                .enter()
                .append('rect')
                .attr('x', d => {
                    return group_x(d.band_domain)(d.key)
                })
                .attr('y', d => {
                    // console.log(d)
                    return y(d.count_range)(d.value)
                })
                .attr('width', d => group_x(d.band_domain).bandwidth())
                .attr('height', d => feat_height - y(d.count_range)(d.value))
                .attr('fill', d => color(d.key))
                .attr('opacity', bar_opacity)
                .attr('class', d => {
                    return 'day' + id + '_' + d.key
                })
            //#endregion

            // #region box plot
            var box = feat_g.append('g')
                .attr("transform", (d, i) => {
                    return `translate(${0}, ${margin.top + feat_height + counter_y_padding / 3})`
                })

            box.append('line')
                .attr("class", "vertLine")
                .attr("stroke", "#C0C0C0")
                .attr('stroke-width', '1px')
                .style("width", 40)
                .attr("x1", d => {
                    var x1 = box_get_domain(d)(d.real_value_range[0])
                    return x1
                })
                .attr("x2", d => {
                    var x2 = box_get_domain(d)(d.real_value_range[1])
                    return x2
                })
                .attr("y1", 0)
                .attr("y2", 0);

            box
                .append('rect')
                .attr("class", "box")
                .attr("y", -boxWidth / 2)
                .attr("x", d => {
                    var xpos = box_get_domain(d)(d.quartiles[1])
                    return xpos
                })
                .attr("width", d => {
                    var wid = box_get_domain(d)(d.quartiles[2]) - box_get_domain(d)(d.quartiles[1])
                    return wid
                })
                .attr("height", boxWidth)
                .attr("stroke", "#474747")
                .attr('stroke-width', 0.5)
                .style("fill", "rgb(255, 255, 255)")
                .style("fill-opacity", 0.7);

            box
                .selectAll("verticalLine")
                .data(d => [[d, d.real_value_range[0]],
                [d, d.quartiles[1]],
                [d, d.real_value_range[1]]])
                .enter()
                .append('line')
                .attr("class", "verticalLine")
                .attr("stroke", "#474747")
                .attr('stroke-width', '1px')
                .style("width", 10)
                .attr("y1", -boxWidth / 2)
                .attr("y2", +boxWidth / 2)
                .attr("x1", d => box_get_domain(d[0])(d[1]))
                .attr("x2", d => box_get_domain(d[0])(d[1]));

            // #endregion

            // #region axes
            //x-axis
            var xaxis = feat_g.append('g')
                .attr("transform", `translate(${0},${margin.top + xaxis_padding})`)
                .each(function (d, i) {
                    var range = d.value_range
                    var name = d.name
                    var copy_domain_dict = _this.deepCopy(d.band_domain)
                    var band_domain = []
                    for (var key in copy_domain_dict) {
                        band_domain.push(copy_domain_dict[key])
                    }
                    // console.log('band_domain', band_domain)
                    var me = d3.select(this)

                    if (name == 'deltatime' || name == 'tran_deltatime') {
                        band_domain.forEach((ba, i) => {
                            band_domain[i] = Math.ceil(band_domain[i] / 1000)
                        })
                    }
                    if (name == 'pr' || name == 'cn') {
                        me.call(d3.axisBottom(x(band_domain))
                            .ticks(4)
                            .tickSize(5)
                            .tickSizeOuter(0)
                            .tickFormat(d3.format("0.4f")))
                            .attr('class', 'axis')
                            .attr('id', d => 'x_axis_' + name)
                    } else {
                        me.call(d3.axisBottom(x(band_domain))
                            .ticks(4)
                            .tickSize(5)
                            .tickSizeOuter(0)
                            .tickFormat(d3.format("0.1f")))
                            .attr('class', 'axis')
                            .attr('id', 'x_axis_' + name)
                    }
                })

            //real y-axis
            feat_g.append('g')
                .attr("transform", `translate(${0},${margin.top})`)
                .each(function (d, i) {
                    var range = d.count_range
                    var me = d3.select(this)
                    me.call(d3.axisLeft(y(range))
                        .ticks(2)
                        .tickSize(2)
                        .tickSizeOuter(0))
                        .attr('class', 'axis')
                })


            //change text size
            svg.selectAll('.axis')
                .selectAll('text')
                .style('font-size', 11)
                .style('opacity', 1)

            //#endregion

            // #region brush
            function brush(d) {
                return d3.brushX()
                    .extent([
                        [2, margin.top * 2], [feat_width, feat_height + margin.top - 1]
                    ])
                    .on('brush', brushed)
            }

            const defaultSelection = [2, feat_width];

            function brushed(d) {
                var range = d.value_range
                var name = d.name
                var band_domain = d.band_domain
                const selection = d3.event.selection;
                // console.log(selection)
                var x_begin = x_reverse(band_domain)(selection[0])
                var x_end = x_reverse(band_domain)(selection[1])

                var idx = _this.feat_idx[name]
                _this.range[idx][name] = [x_begin, x_end]

                if (typeof (_this.$glo_cf_range[_this.data.id]) == 'undefined') {
                    _this.$glo_cf_range[_this.data.id] = {}
                    _this.$glo_table_range[_this.data.id] = {}
                }
                _this.$glo_cf_range[_this.data.id][name] = [x_begin, x_end]
                _this.$glo_table_range[_this.data.id][name] = [x_begin, x_end]
                // console.log('TEMPLATE >> CF_RAW >> BRUSH >> glo cf range', _this.$glo_cf_range)

            }

            const gBrush = feat_g.append('g')
                .attr('class', 'counter_brush')
                .each(function (d) {
                    var me = d3.select(this)
                    me.call(brush(d))
                        .call(brush(d).move, defaultSelection)
                })

            //custom handlers
            //handle group
            const ghandles = gBrush.selectAll('g.handles')
                .data(['handle--o', 'handle--e'])
                .enter()
                .append('g')
                .attr('class', 'steelblue')
                .attr('transform', d => {
                    const x = d == 'handle--o' ? 0 : width / 15;
                    return `translate(${x}, 0)`;
                });

            //get brush
            var b = feat_g.selectAll('.brush')

            // change brush color
            // svg.selectAll('.selection')
            //     .attr('fill', '#72A6C1')
            //     .attr('opacity', 0.5)
            //     .attr('stroke', '#72A6C1')
            svg.selectAll('.selection')
                .attr('fill', '#B64330')
                .attr('opacity', 0.4)
                .attr('stroke', '#B64330')

            //#endregion

            //#endregion

        },
        plot_indiport(id) {
            let _this = this;

            //data
            var data = [
                {
                    'group': 0,
                    'portrait': [
                        { key: 'school', value: 5.411279 },
                        { key: 'grade', value: 45.74553 },
                        { key: 'bindcash', value: 222562.883081 },
                        { key: 'deltatime', value: 22977060 },
                        { key: 'combatscore', value: 86744.749656 },
                        { key: 'sex', value: 0.923659 }
                    ],
                    'indi_portrait': [
                        { key: 'school', value: 5.411133 },
                        { key: 'grade', value: 23.572045 },
                        { key: 'bindcash', value: 50374.590629 },
                        { key: 'deltatime', value: 2149119 },
                        { key: 'combatscore', value: 26594.809406 },
                        { key: 'sex', value: 0.288004 }
                    ]
                },
            ]
            data = _this.indi_port_data.length == 0 ? data : [_this.indi_port_data]

            var max_val = _this.portrait_max

            var svg = d3.select('#' + id)
            svg.selectAll('*').remove()

            const width = 160, height = 220

            const radius = 40, dotRadius = 3;
            const num_circle = 5, num_dim = 6;
            const angleSlice = 2 * Math.PI / num_dim;
            const axisCircles = 2, axisLabelFactor = 1.2;
            const port_group = ['school', 'grade', 'bindcash', 'deltatime', 'combatscore', 'sex']

            var bgColor = "#d6d6d6"

            //scale each dim, return a list
            var rScale = max_val.map(el => d3.scaleLinear().domain([0, el]).range([0, radius]))

            //get line for each dim
            var radarLine = d3.lineRadial()
                .curve(d3.curveCardinalClosed)
                .radius((d, i) => rScale[i](d))
                .angle((d, i) => i * angleSlice)

            var colors = d3.scaleOrdinal()
                .domain(d3.range(0, 7))
                .range([
                    '#9799b6',//1
                    '#78c497',//2
                    '#c0d068',//3
                    '#ebd168',//4
                    '#edc0ba',//5
                    "#d8a0eb",//6
                    '#7fc4cd',//7
                ]);

            var portrait = svg.append('g')
                .selectAll('g')
                .data(data)
                .enter()
                .append('g')
                .attr("transform", (d, i) => `translate(${(i + 1) * width / 2},${height / 2})`)
                .attr("class", (_, i) => "portrait" + i)

            var bgCircle = portrait
                .selectAll('.levels')
                .data(d3.range(1, (axisCircles + 1)).reverse())//[2,1]
                .enter()
                .append('circle')
                .attr('class', 'bgCircle')
                .attr('r', (d, i) => radius / axisCircles * d)
                .style("fill", "none")
                .style("stroke", bgColor)
                .style("fill-opacity", 0.3);

            //#region axis
            var axis = portrait.selectAll('g.axis')
                .data(port_group)
                .enter()
                .append('g')
                .attr('class', 'axis')

            var axis_line = axis
                .append('line')
                .attr("x1", 0)
                .attr("y1", 0)
                .attr("x2", (d, i) => {
                    return radius * 1.1 * Math.cos(angleSlice * i - Math.PI / 2)
                })
                .attr("y2", (d, i) => radius * 1.1 * Math.sin(angleSlice * i - Math.PI / 2))
                .attr("class", "line")
                .style("stroke", bgColor)
                .style("stroke-width", "2px");

            var axis_text = axis
                .append("text")
                .attr("class", "legend")
                .style("font-size", "14px")
                .attr("text-anchor", "middle")
                .attr("font-family", "monospace")
                .attr("dy", "0.35em")
                .attr("x", (d, i) => radius * axisLabelFactor * Math.cos(angleSlice * i - Math.PI / 2))
                .attr("y", (d, i) => radius * axisLabelFactor * Math.sin(angleSlice * i - Math.PI / 2))
                .text(d => d);
            //#endregion

            // cluster radar
            var polygon = portrait.append('g')
                .append('path')
                .attr("d", d => radarLine(d.portrait.map(v => v.value)))
                .attr("fill", (d, i) => {
                    return colors(d.group)
                })
                .attr("fill-opacity", 0.5)
                .attr("stroke", (d, i) => colors(d.group))
                .attr("stroke-width", 1);

            var dot = portrait.append('g')
                .selectAll('circle')
                .data(d => {
                    return d.portrait
                })
                .enter()
                .append('circle')
                .attr('r', dotRadius)
                .attr("cx", (d, i) => rScale[i](d.value) * Math.cos(angleSlice * i - Math.PI / 2))
                .attr("cy", (d, i) => rScale[i](d.value) * Math.sin(angleSlice * i - Math.PI / 2))
                .attr("fill", (d, i) => {
                    return colors(d.group)
                })
                .style("fill-opacity", 0.9);

            // individual radar
            var indi_polygon = portrait.append('g')
                .append('path')
                .attr("d", d => {
                    return radarLine(d.indi_portrait.map(v => v.value))
                })
                .attr("fill", 'none')
                .attr("stroke", (d, i) => {
                    return colors(d.group)
                })
                .attr("stroke-width", 1);
        },
        change_button_style(id) {
            let _this = this
            var svg = d3.select('#' + id)

            _this.change_button.forEach(change_feat => {
                // CHANGE OPACITY
                svg.select('.button_' + change_feat)
                    .attr('fill-opacity', 0.5)

                // CHANGE FLAG TO TRUE
                _this.change_flag[change_feat] = true
            })
        },
        change_pred_style(id) {
            let _this = this
            var svg = d3.select('#' + id)

            _this.clicked_pred.forEach(pred => {
                svg.select(pred)
                    .attr('fill-opacity', 0.8)
                    .attr('stroke-width', 3)
            })
        },
        change_clus_style(id) {
            let _this = this
            var svg = d3.select('#' + id)

            _this.clicked_clus.forEach(clus => {
                var group = clus.substring(7)

                // polygon opacity
                svg.select('.' + clus)
                    .attr('fill-opacity', 0.7)

                // dot raduis
                svg.selectAll('.dot' + group)
                    .attr('r', 5)
            })
        },
        deepCopy(data) {
            if (typeof data !== 'object' || data === null) {
                throw new TypeError('传入参数不是对象')
            }
            let newData = {};
            const dataKeys = Object.keys(data);
            dataKeys.forEach(value => {
                const currentDataValue = data[value];
                // 基本数据类型的值和函数直接赋值拷贝 
                if (typeof currentDataValue !== "object" || currentDataValue === null) {
                    newData[value] = currentDataValue;
                } else if (Array.isArray(currentDataValue)) {
                    // 实现数组的深拷贝
                    newData[value] = [...currentDataValue];
                } else if (currentDataValue instanceof Set) {
                    // 实现set数据的深拷贝
                    newData[value] = new Set([...currentDataValue]);
                } else if (currentDataValue instanceof Map) {
                    // 实现map数据的深拷贝
                    newData[value] = new Map([...currentDataValue]);
                } else {
                    // 普通对象则递归赋值
                    newData[value] = deepCopy(currentDataValue);
                }
            });
            return newData;
        },
        async send_range_to_group() {
            var day = this.data.id * 1
            // var send2cf_range = {}
            // var send2table_range = {}
            // for (var i = 0; i < this.range.length; i++) {
            //     var ran = this.range[i]
            //     var name = Object.keys(ran)[0]
            //     var val_range = ran[name]
            //     val_range[0] *= 1
            //     val_range[1] *= 1
            //     send2cf_range[name] = val_range
            //     send2table_range[name] = val_range
            // }
            // //if change is not empty, add change to cf
            // if (this.change.change.length != 0) {
            //     send2cf_range.change = this.change.change
            // }
            // // console.log(send2cf_range)

            // var finalcf_range = {}, finaltable_range = {}
            // finalcf_range[day] = send2cf_range
            // finaltable_range[day] = send2table_range

            // console.log(finalcf_range, finaltable_range)

            this.$bus.$emit("template range to group", day);
        },
        async send_range_to_individual() {
            // deal with range, output: finalrange
            var day = this.data.id * 1
            var send2indi_range = {}
            this.range.forEach(ran => {
                var name = Object.keys(ran)[0]
                var val_range = ran[name]
                val_range[0] *= 1
                val_range[1] *= 1
                send2indi_range[name] = val_range
            })
            var finalrange = {}
            finalrange[day] = send2indi_range

            // send table data
            var table_data = await HttpHelper.axiosPost('/table',
                finalrange, 600000)

            this.$bus.$emit("template to table", table_data);

            // send boxplot data
            var boxplot_data = await HttpHelper.axiosPost('/boxplot',
                finalrange, 600000)

            this.$bus.$emit("template to boxplot", boxplot_data);
        },
        wait_data() {
            this.$bus.$on("indi port to template", (msg) => {
                var indi_port

                msg.forEach(port => {
                    var day = port.date * 1
                    if (day == this.data.id) {
                        indi_port = port

                        var port_class = port.group
                        var class_port = this.data.values[port_class].portrait
                        indi_port.portrait = class_port

                        this.indi_port_data = indi_port
                        this.plot_indiport("indi_portSvg" + this.id)
                    }
                })
            });
        },
    },
    created() { // 生命周期中，组件被创建后调用
    },
};