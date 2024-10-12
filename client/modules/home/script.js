/* eslint-disable no-undef */
/* eslint-disable no-unused-vars */
// import selectingCtn from "../selecting/index.vue"
// import timeViewCtn from "../time/index.vue"
// import bubbleCtn from "../bubble/index.vue"
// import styleCtn from "../style/index.vue"
// import rankingCtn from "../ranking/index.vue"

// import exploreCtn from "../explore/index.vue"
// import rankv2Ctn from "../rankingv2/index.vue"

import timeCtn from "../timeline/index.vue"
import groupCtn from "../group/index.vue"
import individualCtn from "../individual/index.vue"

import HttpHelper from "common/utils/axios_helper.js";
import { init } from "echarts";
export default {
    components: { // 依赖组件
        timeCtn,
        groupCtn,
        individualCtn,
    },
    data() { // 本页面数据
        return {
        };
    },
    mounted() {
        this.final_init()
        // this.plot_group_legend()
    },
    methods: { // 这里写本页面自定义方法 
        init_global_cluster_name() {
            this.$clicked_cluster = [];
        },
        async init() {
            console.log('INIT.')
            await HttpHelper.axiosPost('/init', 600000)
        },
        async final_init() {
            await this.init()
            console.log('INIT FINISH.')
            this.init_global_cluster_name()

        },
        plot_group_legend() {
            var svg = d3.select('#groupLegend')

            var data = [
                { name: 'high', color: '#b2b4a9', stroke: '#666A53' },
                { name: 'med', color: '#e8be8d', stroke: '#D27D1C' },
                { name: 'low', color: '#aacada', stroke: '#72A6C1' },
                { name: 'churn', color: '#d38e83', stroke: '#B64330' },
            ]

            var width = 20

            var each_status = svg.append('g')
                .selectAll('g')
                .data(data)
                .enter()
                .append('g')
                .attr("transform", (d, i) => `translate(${i * 80},${0})`)

            var each_rect = each_status.append('rect')
                .attr('x', 0)
                .attr('y', 10)
                .attr('width', width)
                .attr('height', 20)
                .attr('fill', d => d.color)
                .attr('stroke', d => d.stroke)
                .attr('stroke-width', 1)

            each_status
                .append('text')
                .attr('x', (d, i) => 30)
                .attr('y', 25)
                .attr('text-anchor', 'left')
                .text(d => d.name)
        }
    },
    created() { // 生命周期中，组件被创建后调用

    },
};