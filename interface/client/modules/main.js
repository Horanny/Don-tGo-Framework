import Vue from 'vue';
import VueRouter from 'vue-router';
import routes from './routers.js';
import * as d3 from "d3";
import 'assets/css/main.less';
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import axios from 'axios';
import * as echarts from 'echarts';
import locale from '../../node_modules/element-ui/lib/locale/lang/en.js';

// import * as d3lasso from 'd3-lasso'
//lasso
var d3lasso = require('d3-lasso');

Vue.prototype.$axios = axios;
axios.defaults.baseURL = '/cf';
axios.defaults.headers.get['Content-Type'] =
    'application/x-www-form-urlencoded';
Vue.config.productionTip = false;

Vue.use(VueRouter);

Vue.use(ElementUI, { locale });

Vue.prototype.openLoading = function () {
    const loading = this.$loading({
        lock: true,
        customClass: 'create-isLoading',
        text: 'Loading',
        // spinner: 'el-icon-loading',
        spinner: '',
        background: 'rgba(255, 255, 255, 0.5)',
      });
      setTimeout(() => {
        loading.close();
      }, 600000);
    return loading;
}

// window.d3 = d3;
window.d3 = Object.assign(d3, { lasso: d3lasso.lasso });

Vue.prototype.$http = window.$http;
Vue.prototype.$echarts = echarts;
Vue.prototype.$bus = new Vue();

// define global
Vue.prototype.$clicked_cluster = [];
Vue.prototype.$glo_cf_range = {};
Vue.prototype.$glo_table_range = {};

const router = new VueRouter({
    mode: 'history',
    routes: routes.routes
});

new Vue({
    router
}).$mount(`#app-wrapper`); 