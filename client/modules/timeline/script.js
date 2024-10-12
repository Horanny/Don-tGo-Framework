/* eslint-disable no-undef */
/* eslint-disable no-unused-vars */
import HttpHelper from "common/utils/axios_helper.js";
import { link, scaleBand } from "d3";
import { h } from "vue";
export default {
    components: { // 依赖组件

    },
    data() { // 本页面数据
        return {
            checkedStatus: [],
            all_status: [{
                value: '1',
                label: 'high',
                color: '#666a53',
            }, {
                value: '2',
                label: 'med',
                color: '#d27d1c',
            }, {
                value: '3',
                label: 'low',
                color: '#72a6c1'
            }, {
                value: '4',
                label: 'churn',
                color: '#B64330'
            },
            ],
            //data
            data: [],
            //style
            bar_opacity: 0.8,
            //interaction with backend
            time_range: [], //send to backend
            send_flag: false,
            last_save_day: '',
        };
    },
    mounted() {
        this.wait_data()
    },
    watch: {
        checkedStatus(val, _oldVal) {
            console.log(this.checkedStatus)
            d3.selectAll('#bar')
                .attr('stroke-width', 1)
            this.checkedStatus.forEach(status => {
                console.log(status)
                d3.selectAll('.' + status)
                    .attr('stroke-width', 2)
            })
        },
    },
    methods: { // 这里写本页面自定义方法
        //plot time view
        plot_time() {
            var _this = this

            //#region basic layout config
            const margin = { top: 10, right: 0, bottom: 10, left: 50 }  // focus
            const svg = d3.select('#timeSvg')
            var width = 600, height = 80
            width = width - margin.right - margin.left
            height = height - margin.top - margin.bottom
            //#endregion

            //#region data
            data = _this.data.length == 0 ? data : _this.data

            //#endregion

            //#region params for functions, including range,...
            //get range group of x-axis
            var Xrange = d3.map(data, d => d.date).keys()
            var Xcount = []
            Xrange.forEach((element, i) => {
                Xcount.push(i)
            });

            //get range of y-axis based on min and max of three status of churn
            var churn_range = d3.extent(data, d => [d.high, d.med, d.low])
            var Yrange = [d3.max([0, d3.min(churn_range[0]) / 2]), d3.max(churn_range[1])]

            //param group
            var status_group = ['high', 'med', 'low', 'churn']

            //color
            var color_group = ['#676b54', '#d27d1c', '#4881b3', '#b23722']
            //#endregion

            //#region functions
            var x = d3.scaleBand()
                .domain(Xrange)
                .range([0, width])
                // .padding(0.5)
                .paddingInner(0.5)
                .paddingOuter(0.2)
                .align(0.5)

            var x_refine_day = d3.scaleBand()
                .domain(['D0', 'D1', 'D2', 'D3'])
                .range([0, width])
                // .padding(0.5)
                .paddingInner(0.5)
                .paddingOuter(0.2)
                .align(0.5)

            var group_x = d3.scaleBand()
                .domain(status_group)
                .range([0, x.bandwidth()])
                .padding(0.3)

            function scaleBandInvert(scale) {
                //[d1, d2, d3, d4]
                var domain = scale.domain();
                var paddingOuter = scale(domain[0]);

                var eachBand = scale.step();

                return function (value) {
                    var index = Math.floor(((value - paddingOuter) / eachBand));
                    return index
                    // return domain[Math.max(0, Math.min(index, domain.length - 1))];
                }
            }
            var x_reverse = scaleBandInvert(x);

            var y = d3.scaleLinear()
                .domain(Yrange)
                .range([height, margin.top])


            var xAxis = g => g
                .attr("transform", `translate(${margin.left},${height})`)
                .call(d3.axisBottom(x_refine_day)
                    .ticks(5)
                    .tickSize(5)
                    .tickSizeOuter(0)
                )

            var yAxis = g => g
                .attr("transform", `translate(${margin.left},${0})`)
                .call(d3.axisLeft(y)
                    .ticks(3)
                    .tickSize(2)
                    .tickSizeOuter(0)
                )

            var color = d3.scaleOrdinal()
                .domain(status_group)
                .range(color_group)
            //#endregion

            //#region chart
            var time_line_g = svg.append('g')

            time_line_g.append('g')
                .call(yAxis)

            time_line_g.append('g')
                .call(xAxis)

            var daygroup = time_line_g.append('g')
                .selectAll('g')
                .data(data)
                .enter()
                .append('g')
                .attr("transform", (d, i) => {
                    return `translate(${margin.left + x(d.date)},${0})`
                })
                .attr('class', 'daygroup')

            var bars = daygroup.selectAll('rect')
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
                .attr('class', d => d.key)
                .attr('id', 'bar')


            //#region brush
            var brush = d3.brushX()
                .extent([
                    [margin.left, 0], [width + margin.left, height - 1]
                ])
                .on('brush', brushed)

            // const defaultSelection = [margin.left, width + margin.left];
            const defaultSelection = [margin.left, width / 3];

            function brushed() {
                const selection = d3.event.selection;
                var x_begin_day = x_reverse(selection[0] - margin.left),
                    x_end_day = x_reverse(selection[1] - margin.left) - 1

                // save time range
                _this.time_range = [x_begin_day + 1, x_end_day + 1]
            }

            const gBrush = time_line_g.append('g')
                .attr('class', 'brush')
                .call(brush)
                .call(brush.move, defaultSelection)

            // change brush color
            svg.select('.selection')
                .attr('fill', '#CFDBDC')
                .attr('opacity', 0.6)
                .attr('stroke', '#72A6C1')

            //get brush

            //#endregion

            //#endregion

        },
        plot_time_legend() {
            const svg = d3.select('#timeLegend')

            var data = [
                { name: 'high', color: '#7a7d6c', stroke: '#676b54' },
                { name: 'med', color: '#d28e48', stroke: '#d27d1c' },
                { name: 'low', color: '#6490b8', stroke: '#4881b3' },
                { name: 'churn', color: '#b45947', stroke: '#b23722' },
            ]

            var width = 30

            var each_status = svg.append('g')
                .selectAll('g')
                .data(data)
                .enter()
                .append('g')                    
                .attr("transform", (d, i) => `translate(${400 + i * 50},${15})`)

            var each_rect = each_status.append('rect')
                .attr('x', 0)
                .attr('y', 25)
                .attr('width', width)
                .attr('height', 30)
                .attr('fill', d => d.color)
                .attr('stroke', d => d.stroke)
                .attr('stroke-width', 1)
            
            each_status
                .append('text')
                .attr('x', (d, i) => 15)
                .attr('y', 10)
                .attr('text-anchor', 'middle')
                .text(d => d.name)
                .style('font-size', '16px')
        },
        async get_timeline() {

            // console.log('get_timeline')
            this.data = await HttpHelper.axiosPost('/timeline')
            // console.log(this.data)
        },
        async wait_data() {
            // initialize timeline
            await this.get_timeline();

            // plot time view
            this.plot_time()

            // plot time view legend
            this.plot_time_legend()

            // GET LAST SAVE DAY AND UPDATE 
            this.$bus.$on('send last save day', async (msg) => {
                this.last_save_day = msg
            })
        },
        async send_days() {
            console.log('TIMELINE >> FUNC >> SEND DAYS')

            this.send_flag = true

            this.$bus.$emit("timeline to group", this.time_range);

        },
        async init() {
            console.log('INIT.')
            await HttpHelper.axiosPost('/init', 600000)
        },
        async save() {
            if (this.send_flag == false) {
                // console.log('GLOBAL INIT FINISH.')

                // SAVE GLOBAL
                var ob_se = await HttpHelper.axiosPost('/saveStep',
                    -1, 600000)
                console.log('TIMELINE >> SAVE GLOBAL OB_SE FINISH.')
            }
            else {
                if (this.last_save_day == '') {
                    // SAVE SELECTED DAY
                    var ob_se = await HttpHelper.axiosPost('/saveStep',
                        this.time_range[0], 600000)
                    console.log('TIMELINE >> NOT CHANGED IN ROW: SAVE SPECIFIC OB_SE FINISH.')
                }
                else {
                    // IF NOT CLICK ON THE ROW BUTTON, SAVE THE FIRST DAY
                    // console.log(this.time_range[0])
                    var ob_se = await HttpHelper.axiosPost('/saveStep',
                        this.last_save_day, 600000)
                    console.log('TIMELINE >> CHANGED IN ROW: SAVE SPECIFIC OB_SE FINISH.')
                }
            }

            // send group data and time range to group
            this.$bus.$emit("save to group", [ob_se]);
        },
    },
    created() { // 生命周期中，组件被创建后调用
    },
};