const express = require('express');
const path = require('path');
const cookieParser = require('cookie-parser');
const bodyParser = require('body-parser');
const compress = require('compression');
const favicon = require('serve-favicon');
const config = require('./server/config');
const baseRouter = require('./server/router');
const app = express();

//#region MANY MANY USES
// const middlewares = require('./server/middleware');
const helmet = require('helmet');//防注入中间件
// const session = require('express-session');
// let DCacheUtil = null;
// let DCacheSessionStore = null;

//从绝对路径中读取网页图标
app.use(favicon(__dirname + '/client/assets/image/favicon.ico'))

//X-Powered-By是网站响应头信息其中的一个，
//出于安全的考虑，一般会修改或删除掉这个信息。
app.disable('x-powered-by');

//在node中，有全局变量process表示的是当前的node进程。
//NODE_ENV是一个用户自定义的变量，
//在webpack中它的用途是判断生产环境或开发环境。
if (!process.env.NODE_ENV) {
    process.env.NODE_ENV = 'local';
}
console.log("Node 的版本是？" + process.env.NODE_ENV);

app.use(require('cors')());

//express 使用html模板
//模板配置
if(Object.is(process.env.NODE_ENV,'local')){
    //将ejs模板映射到.html文件
    //上面实际上是调用了ejs的.renderFile()方法
    //ejs.__express是该方法在ejs内部的另一个名字。
    //因为加载的模板引擎后调用的是同一个方法.__express，
    //所以如果使用的是ejs模板，不用配置该项。
    app.engine('.html', require('ejs').__express);
    //在.set()方法的参数中，有一项是'view engine'，
    //表示没有指定文件模板格式时，默认使用的引擎插件；
    //如果这里设置为html文件，设置路由指定文件时，
    //只需写文件名，就会找对应的html文件。
    app.set('view engine', 'html');
    // app.set('views', __dirname + '/server/mock/views');
}

//使用helmet模块保证应用安全性
app.use(helmet());
//压缩请求
app.use(compress());
//这个方法返回一个仅仅用来解析json格式的中间件。
//这个中间件能接受任何body中任何Unicode编码的字符。
//支持自动的解析gzip和 zlib。
app.use(bodyParser.json(config.bodyParserJsonOptions));
//这个方法也返回一个中间件，
//这个中间件用来解析body中的urlencoded字符，
//只支持utf-8的编码的字符。同样也支持自动的解析gzip和 zlib
/////////////////////////////////////
//bodyParser.json是用来解析json数据格式的。
//bodyParser.urlencoded则是用来解析我们通常的form表单提交的数据，
//也就是请求头中包含这样的信息：
//Content-Type: application/x-www-form-urlencoded
app.use(bodyParser.urlencoded(config.bodyParserUrlencodedOptions));
//方便操作客户端中的cookie值。
app.use(cookieParser());

//拦截请求,添加参数校验
baseRouter.interceptorHttp(app);
//在进入首页或详情页时session check
app.use(async function (req, res, next) {
    console.log('goto web page' + req.path);
    // TODO
    next();
});

//使用静态文件？
app.use(express.static(path.join(__dirname, './dist')));

// app.use(session({
//     secret: 'foo',
//     cookie: { secure: false, maxAge: 1000 * 60 * 60 * 8 },
//     //cookie: {secure: false, maxAge: 1000 * 60 },
//     resave: false,//是否允许session重新设置，要保证session有操作的时候必须设置这个属性为true
//     rolling: true,//是否按照原设定的maxAge值重设session同步到cookie中，要保证session有操作的时候必须设置这个属性为true
//     saveUninitialized: true,//是否设置session在存储容器中可以给修改
//     store: DCacheSessionStore ? new DCacheSessionStore({
//         client: DCacheUtil
//     }) : undefined
//     //unset:'keep'//设置req.session在什么时候可以设置;destory:请求结束时候session摧毁;keep:session在存储中的值不变，在请求之间的修改将会忽略，而不保存
// }));

// app.use(middlewares());

//#endregion

/////////////我是分割线/////////////
const model = require('./server/model/Data2');
const datafunc = require('./server/myfunc/datafunc');

var SelectingTimeModel = model.SelectingTimeModel
var SelectingCountModel = model.SelectingCountModel
var ExploreRankingModel = model.ExploreRankingModel
var ExploreOveriewModel = model.ExploreOveriewModel

var getdata_byCode = datafunc.getdata_byCode;
var get_alldata = datafunc.get_alldata;
var get_Overview = datafunc.get_Overview;
var get_Ranking = datafunc.get_Ranking;
var get_Highlight_Overview = datafunc.get_Highlight_Overview;
var get_style_byCode = datafunc.get_style_byCode;
var get_Explore = datafunc.get_Explore;
var get_Explore_plot = datafunc.get_Explore_plot;

//get selecting data
app.get('/api/get_selecting', async function(req,res){
    let time = await get_alldata(SelectingTimeModel)
    let count = await get_alldata(SelectingCountModel)

    res.send({time, count})
})


//receive the time range and find suitable fund managers
//return the codes
app.get('/api/findFMs', async function(req, res){
    let ind = req.query.industry_cat;
    let startTime = +req.query.start_time;
    let endTime = +req.query.end_time;
    let ratio = +req.query.ratio;

    console.log('ind',ind)
    console.log('ratio', ratio)
    console.log('startTIME',startTime)
    console.log('endtime',endTime)

    // let Codes = ["070013.OF", "550008.OF","270021.OF"];

    let foundFMs = await get_Highlight_Overview(RankingModel, ind, ratio, startTime, endTime);
    res.send(foundFMs)
})

//get ranking data
app.get('/api/getRanking', async function(req, res){
    let ind = req.query.industry_cat;
    let startTime = +req.query.start_time;
    let endTime = +req.query.end_time;
    let ratio = +req.query.ratio;

    // let ind = ["A"];
    // let ratio = 5;
    // let startTime = 1703;
    // let endTime = 1803;
    let ranking_data = await get_Ranking(RankingModel, ind, ratio, startTime, endTime);
    res.send(ranking_data)
})

//get style data
app.get('/api/getStyle', async function(req, res){
    let code = req.query.code;
    let style_data = await get_style_byCode(RankingModel, code);
    res.send(style_data);
})

//get explore data
app.get('/api/getExplore', async function(req, res){
    // let code = req.query.code;
    let ind = req.query.industry_cat;
    let startTime = req.query.start_time;
    let endTime = req.query.end_time;
    // let ind = ["B"]
    // let startTime = new Date("2018-07-01")
    // startTime = startTime.setHours(startTime.getHours() -8)
    // console.log(startTime)
    // let endTime = new Date("2019-04-01");
    // endTime = endTime.setHours(endTime.getHours() -8)
    let explore_data = await get_Explore(ExploreRankingModel, ind, startTime, endTime);
    let explore_plot_data = await get_Explore_plot(ExploreOveriewModel, ind, startTime, endTime);
    res.send({
        "para" : explore_data,
        "plot" : explore_plot_data
    });
})


app.use(function (err, req, res, next) {
    res.status(err.status || 500);
    console.log(`path : ${req.path}`, err);

    res.json({
        code: -1001,
        msg: err.message
    });

});
app.set('host', process.env.IP || 'localhost');
app.set('port', process.env.PORT || 8050);
const server = app.listen(app.get('port'), app.get('host'), function () {
    console.log('server listening on port', server.address().port);
});
