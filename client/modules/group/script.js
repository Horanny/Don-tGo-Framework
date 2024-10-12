/* eslint-disable no-undef */
/* eslint-disable no-unused-vars */
import HttpHelper from "common/utils/axios_helper.js";
import { cluster } from "d3";
import groupTemplate from "../groupTemplate/index.vue";


const beeswarm = require("d3-beeswarm")

export default {
    components: { // 依赖组件
        groupTemplate,
    },
    data() { // 本页面数据
        return {
            updated_flag: true,
            lineSvgStyle: {
                width: "1750",
                height: "",
            },
            data: [
                {
                    id: 0,
                    values: [
                        {
                            'group': 0,
                            'portrait': [
                                { key: 'school', value: 5.46125 },
                                // { key: 'grade', value: 51.64875 },
                                { key: 'grade', value: 45.64875 },
                                // { key: 'bindcash', value: 366607.3 },
                                { key: 'bindcash', value: 286607.3 },
                                { key: 'deltatime', value: 74317030 },
                                // { key: 'combatscore', value: 123010.2375 },
                                { key: 'combatscore', value: 83010.2375 },
                                // { key: 'sex', value: 1.35125 }
                                { key: 'sex', value: 1.08125 }
                            ],
                            'num': [
                                { key: 'high', value: 3 },
                                { key: 'med', value: 1 },
                                { key: 'low', value: 1 },
                                { key: 'churn', value: 0 },
                            ]
                        },
                        {
                            'group': 1,
                            'portrait': [
                                { key: 'school', value: 5.471707 },
                                { key: 'grade', value: 22.626933 },
                                { key: 'bindcash', value: 45803.957096 },
                                { key: 'deltatime', value: 2655920 },
                                { key: 'combatscore', value: 25414.044836 },
                                { key: 'sex', value: 0.30419 }
                            ],
                            'num': [
                                { key: 'high', value: 6 },
                                { key: 'med', value: 1 },
                                { key: 'low', value: 0 },
                                { key: 'churn', value: 0 },
                            ]
                        },
                        {
                            'group': 2,
                            'portrait': [
                                { key: 'school', value: 5.461431 },
                                { key: 'grade', value: 46.679523 },
                                { key: 'bindcash', value: 241448.515428 },
                                { key: 'deltatime', value: 30262770 },
                                { key: 'combatscore', value: 88852.808555 },
                                { key: 'sex', value: 1.060659 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 3,
                            'portrait': [
                                { key: 'school', value: 5.462725 },
                                { key: 'grade', value: 49.987147 },
                                { key: 'bindcash', value: 311576.8 },
                                { key: 'deltatime', value: 48875210 },
                                { key: 'combatscore', value: 109432.50437 },
                                { key: 'sex', value: 1.262211 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 4,
                            'portrait': [
                                { key: 'school', value: 5.444444 },
                                { key: 'grade', value: 47.388889 },
                                { key: 'bindcash', value: 231500.333333 },
                                { key: 'deltatime', value: 206593000 },
                                { key: 'combatscore', value: 90552.444444 },
                                { key: 'sex', value: 1.222222 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 5,
                            'portrait': [
                                { key: 'school', value: 5.389003 },
                                { key: 'grade', value: 39.981901 },
                                { key: 'bindcash', value: 142614.026346 },
                                { key: 'deltatime', value: 14643270 },
                                { key: 'combatscore', value: 61861.845132 },
                                { key: 'sex', value: 0.764261 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 6,
                            'portrait': [
                                { key: 'school', value: 5.37963 },
                                { key: 'grade', value: 50.990741 },
                                { key: 'bindcash', value: 302601.50463 },
                                { key: 'deltatime', value: 111254600 },
                                { key: 'combatscore', value: 119708.62963 },
                                { key: 'sex', value: 1.263889 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                    ],
                    pred: [{
                        date: 'day2',
                        high: 2,
                        med: 5,
                        low: 7,
                        churn: 0,
                    },]
                },
                {
                    id: 1,
                    values: [
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
                            'num': [
                                { key: 'high', value: 2 },
                                { key: 'med', value: 6 },
                                { key: 'low', value: 2 },
                                { key: 'churn', value: 0 },
                            ]
                        },
                        {
                            'group': 1,
                            'portrait': [
                                { key: 'school', value: 5.411133 },
                                { key: 'grade', value: 23.572045 },
                                { key: 'bindcash', value: 50374.590629 },
                                { key: 'deltatime', value: 2149119 },
                                { key: 'combatscore', value: 26594.809406 },
                                { key: 'sex', value: 0.288004 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 1 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 1 },
                            ]
                        },
                        {
                            'group': 2,
                            'portrait': [
                                { key: 'school', value: 5.457275 },
                                { key: 'grade', value: 51.838337 },
                                { key: 'bindcash', value: 352469.815242 },
                                { key: 'deltatime', value: 84368370 },
                                { key: 'combatscore', value: 125174.972286 },
                                { key: 'sex', value: 1.277136 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 3,
                            'portrait': [
                                { key: 'school', value: 5.491607 },
                                { key: 'grade', value: 49.578897 },
                                { key: 'bindcash', value: 293812.695444 },
                                { key: 'deltatime', value: 38045790 },
                                { key: 'combatscore', value: 107085.941966 },
                                { key: 'sex', value: 1.144365 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 4,
                            'portrait': [
                                { key: 'school', value: 5.594203 },
                                { key: 'grade', value: 46.463768 },
                                { key: 'bindcash', value: 209409.043478 },
                                { key: 'deltatime', value: 145198900 },
                                { key: 'combatscore', value: 95554.507246 },
                                { key: 'sex', value: 1 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 5,
                            'portrait': [
                                { key: 'school', value: 5.453599 },
                                { key: 'grade', value: 51.734605 },
                                { key: 'bindcash', value: 369998.6366 },
                                { key: 'deltatime', value: 56143260 },
                                { key: 'combatscore', value: 121448.388552 },
                                { key: 'sex', value: 1.299219 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 6,
                            'portrait': [
                                { key: 'school', value: 5.403448 },
                                { key: 'grade', value: 37.903775 },
                                { key: 'bindcash', value: 133662.675322 },
                                { key: 'deltatime', value: 10841270 },
                                { key: 'combatscore', value: 56855.363736 },
                                { key: 'sex', value: 0.621863 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                    ],
                    pred: [{
                        id: 0,
                        date: 'day0',
                        high: 2,
                        med: 2,
                        low: 12,
                        churn: 1,
                    },],
                    indi: [{
                        'uid': 0,
                        'group': 0,
                        'portrait': [
                            { key: 'school', value: 2 },
                            { key: 'grade', value: 35 },
                            { key: 'bindcash', value: 2520000 },
                            { key: 'deltatime', value: 1200000 },
                            { key: 'combatscore', value: 2500 },
                            { key: 'sex', value: 2 }
                        ],
                    }
                    ]
                },
                {
                    id: 2,
                    values: [
                        {
                            'group': 0,
                            'portrait': [
                                { key: 'school', value: 5.42806 },
                                { key: 'grade', value: 24.938228 },
                                { key: 'bindcash', value: 54475.31434 },
                                { key: 'deltatime', value: 2442286 },
                                { key: 'combatscore', value: 28808.28471 },
                                { key: 'sex', value: 0.300785 }
                            ],
                            'num': [
                                { key: 'high', value: 0 },
                                { key: 'med', value: 0 },
                                { key: 'low', value: 4 },
                                { key: 'churn', value: 6 },
                            ]
                        },
                        {
                            'group': 1,
                            'portrait': [
                                { key: 'school', value: 5.493614 },
                                { key: 'grade', value: 47.960304 },
                                { key: 'bindcash', value: 240594.5885 },
                                { key: 'deltatime', value: 27901180 },
                                { key: 'combatscore', value: 97961.57957 },
                                { key: 'sex', value: 0.957887 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 1 },
                                { key: 'low', value: 8 },
                                { key: 'churn', value: 0 },
                            ]
                        },
                        {
                            'group': 2,
                            'portrait': [
                                { key: 'school', value: 5.483212 },
                                { key: 'grade', value: 52.007299 },
                                { key: 'bindcash', value: 333275.9328 },
                                { key: 'deltatime', value: 74170010 },
                                { key: 'combatscore', value: 125297.1971 },
                                { key: 'sex', value: 1.224818 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 3,
                            'portrait': [
                                { key: 'school', value: 5.511278 },
                                { key: 'grade', value: 51.819549 },
                                { key: 'bindcash', value: 311157.2556 },
                                { key: 'deltatime', value: 121581400 },
                                { key: 'combatscore', value: 122225.2932 },
                                { key: 'sex', value: 1.255639 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 4,
                            'portrait': [
                                { key: 'school', value: 5.473082 },
                                { key: 'grade', value: 40.814328 },
                                { key: 'bindcash', value: 147042.5551 },
                                { key: 'deltatime', value: 13043180 },
                                { key: 'combatscore', value: 65943.09093 },
                                { key: 'sex', value: 0.671895 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 5,
                            'portrait': [
                                { key: 'school', value: 5.166667 },
                                { key: 'grade', value: 43.333333 },
                                { key: 'bindcash', value: 140403.1667 },
                                { key: 'deltatime', value: 349502000 },
                                { key: 'combatscore', value: 64489.33333 },
                                { key: 'sex', value: 0.833333 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 6,
                            'portrait': [
                                { key: 'school', value: 5.413951 },
                                { key: 'grade', value: 51.294807 },
                                { key: 'bindcash', value: 326830.8366 },
                                { key: 'deltatime', value: 46358610 },
                                { key: 'combatscore', value: 118785.028 },
                                { key: 'sex', value: 1.164969 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                    ],
                    pred: [{
                        date: 'day1',
                        high: 0,
                        med: 0,
                        low: 7,
                        churn: 10,
                    },]
                },
                {
                    id: 3,
                    values: [
                        {
                            'group': 0,
                            'portrait': [
                                { key: 'school', value: 5.433242 },
                                { key: 'grade', value: 23.083058 },
                                { key: 'bindcash', value: 50293.27034 },
                                { key: 'deltatime', value: 1849281 },
                                { key: 'combatscore', value: 27217.77653 },
                                { key: 'sex', value: 0.24322 }
                            ],
                            'num': [
                                { key: 'high', value: 0 },
                                { key: 'med', value: 0 },
                                { key: 'low', value: 0 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 1,
                            'portrait': [
                                { key: 'school', value: 5.496 },
                                { key: 'grade', value: 49.880471 },
                                { key: 'bindcash', value: 282509.208 },
                                { key: 'deltatime', value: 34468640 },
                                { key: 'combatscore', value: 109959.9816 },
                                { key: 'sex', value: 1.025882 }
                            ],
                            'num': [
                                { key: 'high', value: 0 },
                                { key: 'med', value: 0 },
                                { key: 'low', value: 1 },
                                { key: 'churn', value: 9 },
                            ]
                        },
                        {
                            'group': 2,
                            'portrait': [
                                { key: 'school', value: 5.469945 },
                                { key: 'grade', value: 52.32969 },
                                { key: 'bindcash', value: 305528.6521 },
                                { key: 'deltatime', value: 78707990 },
                                { key: 'combatscore', value: 128218.3588 },
                                { key: 'sex', value: 1.169399 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 3,
                            'portrait': [
                                { key: 'school', value: 5.550979 },
                                { key: 'grade', value: 37.390874 },
                                { key: 'bindcash', value: 113494.1948 },
                                { key: 'deltatime', value: 9520312 },
                                { key: 'combatscore', value: 55254.40359 },
                                { key: 'sex', value: 0.480315 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 4,
                            'portrait': [
                                { key: 'school', value: 5.445736 },
                                { key: 'grade', value: 52.096899 },
                                { key: 'bindcash', value: 329119.1364 },
                                { key: 'deltatime', value: 51950550 },
                                { key: 'combatscore', value: 123863.2016 },
                                { key: 'sex', value: 1.154264 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 5,
                            'portrait': [
                                { key: 'school', value: 5.590909 },
                                { key: 'grade', value: 51.212121 },
                                { key: 'bindcash', value: 276306.2273 },
                                { key: 'deltatime', value: 134181700 },
                                { key: 'combatscore', value: 118014.4697 },
                                { key: 'sex', value: 1.212121 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                        {
                            'group': 6,
                            'portrait': [
                                { key: 'school', value: 5.401909 },
                                { key: 'grade', value: 45.738973 },
                                { key: 'bindcash', value: 202073.0349 },
                                { key: 'deltatime', value: 20432950 },
                                { key: 'combatscore', value: 86203.43235 },
                                { key: 'sex', value: 0.796577 }
                            ],
                            'num': [
                                { key: 'high', value: 1 },
                                { key: 'med', value: 3 },
                                { key: 'low', value: 6 },
                                { key: 'churn', value: 10 },
                            ]
                        },
                    ],
                    pred: [{
                        date: 'day2',
                        high: 0,
                        med: 0,
                        low: 2,
                        churn: 15,
                    },]
                },

            ],
            clu_data: [
                [
                    {
                        'group': 0,
                        'portrait': [
                            { key: 'school', value: 5.46125 },
                            { key: 'grade', value: 51.64875 },
                            { key: 'bindcash', value: 366607.3 },
                            { key: 'deltatime', value: 74317030 },
                            { key: 'combatscore', value: 123010.2375 },
                            { key: 'sex', value: 1.35125 }
                        ],
                    },
                    {
                        'group': 1,
                        'portrait': [
                            { key: 'school', value: 5.471707 },
                            { key: 'grade', value: 22.626933 },
                            { key: 'bindcash', value: 45803.957096 },
                            { key: 'deltatime', value: 2655920 },
                            { key: 'combatscore', value: 25414.044836 },
                            { key: 'sex', value: 0.30419 }
                        ],
                    },
                    {
                        'group': 2,
                        'portrait': [
                            { key: 'school', value: 5.461431 },
                            { key: 'grade', value: 46.679523 },
                            { key: 'bindcash', value: 241448.515428 },
                            { key: 'deltatime', value: 30262770 },
                            { key: 'combatscore', value: 88852.808555 },
                            { key: 'sex', value: 1.060659 }
                        ],
                    },
                    {
                        'group': 3,
                        'portrait': [
                            { key: 'school', value: 5.462725 },
                            { key: 'grade', value: 49.987147 },
                            { key: 'bindcash', value: 311576.8 },
                            { key: 'deltatime', value: 48875210 },
                            { key: 'combatscore', value: 109432.50437 },
                            { key: 'sex', value: 1.262211 }
                        ],
                    },
                    {
                        'group': 4,
                        'portrait': [
                            { key: 'school', value: 5.444444 },
                            { key: 'grade', value: 47.388889 },
                            { key: 'bindcash', value: 231500.333333 },
                            { key: 'deltatime', value: 206593000 },
                            { key: 'combatscore', value: 90552.444444 },
                            { key: 'sex', value: 1.222222 }
                        ],
                    },
                    {
                        'group': 5,
                        'portrait': [
                            { key: 'school', value: 5.389003 },
                            { key: 'grade', value: 39.981901 },
                            { key: 'bindcash', value: 142614.026346 },
                            { key: 'deltatime', value: 14643270 },
                            { key: 'combatscore', value: 61861.845132 },
                            { key: 'sex', value: 0.764261 }
                        ],
                    },
                    {
                        'group': 6,
                        'portrait': [
                            { key: 'school', value: 5.37963 },
                            { key: 'grade', value: 50.990741 },
                            { key: 'bindcash', value: 302601.50463 },
                            { key: 'deltatime', value: 111254600 },
                            { key: 'combatscore', value: 119708.62963 },
                            { key: 'sex', value: 1.263889 }
                        ],
                    },
                ],
                [
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
                    },
                    {
                        'group': 1,
                        'portrait': [
                            { key: 'school', value: 5.411133 },
                            { key: 'grade', value: 23.572045 },
                            { key: 'bindcash', value: 50374.590629 },
                            { key: 'deltatime', value: 2149119 },
                            { key: 'combatscore', value: 26594.809406 },
                            { key: 'sex', value: 0.288004 }
                        ],
                    },
                    {
                        'group': 2,
                        'portrait': [
                            { key: 'school', value: 5.457275 },
                            { key: 'grade', value: 51.838337 },
                            { key: 'bindcash', value: 352469.815242 },
                            { key: 'deltatime', value: 84368370 },
                            { key: 'combatscore', value: 125174.972286 },
                            { key: 'sex', value: 1.277136 }
                        ],
                    },
                    {
                        'group': 3,
                        'portrait': [
                            { key: 'school', value: 5.491607 },
                            { key: 'grade', value: 49.578897 },
                            { key: 'bindcash', value: 293812.695444 },
                            { key: 'deltatime', value: 38045790 },
                            { key: 'combatscore', value: 107085.941966 },
                            { key: 'sex', value: 1.144365 }
                        ],
                    },
                    {
                        'group': 4,
                        'portrait': [
                            { key: 'school', value: 5.594203 },
                            { key: 'grade', value: 46.463768 },
                            { key: 'bindcash', value: 209409.043478 },
                            { key: 'deltatime', value: 145198900 },
                            { key: 'combatscore', value: 95554.507246 },
                            { key: 'sex', value: 1 }
                        ],
                    },
                    {
                        'group': 5,
                        'portrait': [
                            { key: 'school', value: 5.453599 },
                            { key: 'grade', value: 51.734605 },
                            { key: 'bindcash', value: 369998.6366 },
                            { key: 'deltatime', value: 56143260 },
                            { key: 'combatscore', value: 121448.388552 },
                            { key: 'sex', value: 1.299219 }
                        ],
                    },
                    {
                        'group': 6,
                        'portrait': [
                            { key: 'school', value: 5.403448 },
                            { key: 'grade', value: 37.903775 },
                            { key: 'bindcash', value: 133662.675322 },
                            { key: 'deltatime', value: 10841270 },
                            { key: 'combatscore', value: 56855.363736 },
                            { key: 'sex', value: 0.621863 }
                        ],
                    },
                ],
                [
                    {
                        'group': 0,
                        'portrait': [
                            { key: 'school', value: 5.42806 },
                            { key: 'grade', value: 24.938228 },
                            { key: 'bindcash', value: 54475.31434 },
                            { key: 'deltatime', value: 2442286 },
                            { key: 'combatscore', value: 28808.28471 },
                            { key: 'sex', value: 0.300785 }
                        ],
                    },
                    {
                        'group': 1,
                        'portrait': [
                            { key: 'school', value: 5.493614 },
                            { key: 'grade', value: 47.960304 },
                            { key: 'bindcash', value: 240594.5885 },
                            { key: 'deltatime', value: 27901180 },
                            { key: 'combatscore', value: 97961.57957 },
                            { key: 'sex', value: 0.957887 }
                        ],
                    },
                    {
                        'group': 2,
                        'portrait': [
                            { key: 'school', value: 5.483212 },
                            { key: 'grade', value: 52.007299 },
                            { key: 'bindcash', value: 333275.9328 },
                            { key: 'deltatime', value: 74170010 },
                            { key: 'combatscore', value: 125297.1971 },
                            { key: 'sex', value: 1.224818 }
                        ],
                    },
                    {
                        'group': 3,
                        'portrait': [
                            { key: 'school', value: 5.511278 },
                            { key: 'grade', value: 51.819549 },
                            { key: 'bindcash', value: 311157.2556 },
                            { key: 'deltatime', value: 121581400 },
                            { key: 'combatscore', value: 122225.2932 },
                            { key: 'sex', value: 1.255639 }
                        ],
                    },
                    {
                        'group': 4,
                        'portrait': [
                            { key: 'school', value: 5.473082 },
                            { key: 'grade', value: 40.814328 },
                            { key: 'bindcash', value: 147042.5551 },
                            { key: 'deltatime', value: 13043180 },
                            { key: 'combatscore', value: 65943.09093 },
                            { key: 'sex', value: 0.671895 }
                        ],
                    },
                    {
                        'group': 5,
                        'portrait': [
                            { key: 'school', value: 5.166667 },
                            { key: 'grade', value: 43.333333 },
                            { key: 'bindcash', value: 140403.1667 },
                            { key: 'deltatime', value: 349502000 },
                            { key: 'combatscore', value: 64489.33333 },
                            { key: 'sex', value: 0.833333 }
                        ],
                    },
                    {
                        'group': 6,
                        'portrait': [
                            { key: 'school', value: 5.413951 },
                            { key: 'grade', value: 51.294807 },
                            { key: 'bindcash', value: 326830.8366 },
                            { key: 'deltatime', value: 46358610 },
                            { key: 'combatscore', value: 118785.028 },
                            { key: 'sex', value: 1.164969 }
                        ],
                    },
                ],
                [
                    {
                        'group': 0,
                        'portrait': [
                            { key: 'school', value: 5.433242 },
                            { key: 'grade', value: 23.083058 },
                            { key: 'bindcash', value: 50293.27034 },
                            { key: 'deltatime', value: 1849281 },
                            { key: 'combatscore', value: 27217.77653 },
                            { key: 'sex', value: 0.24322 }
                        ],
                    },
                    {
                        'group': 1,
                        'portrait': [
                            { key: 'school', value: 5.496 },
                            { key: 'grade', value: 49.880471 },
                            { key: 'bindcash', value: 282509.208 },
                            { key: 'deltatime', value: 34468640 },
                            { key: 'combatscore', value: 109959.9816 },
                            { key: 'sex', value: 1.025882 }
                        ],
                    },
                    {
                        'group': 2,
                        'portrait': [
                            { key: 'school', value: 5.469945 },
                            { key: 'grade', value: 52.32969 },
                            { key: 'bindcash', value: 305528.6521 },
                            { key: 'deltatime', value: 78707990 },
                            { key: 'combatscore', value: 128218.3588 },
                            { key: 'sex', value: 1.169399 }
                        ],
                    },
                    {
                        'group': 3,
                        'portrait': [
                            { key: 'school', value: 5.550979 },
                            { key: 'grade', value: 37.390874 },
                            { key: 'bindcash', value: 113494.1948 },
                            { key: 'deltatime', value: 9520312 },
                            { key: 'combatscore', value: 55254.40359 },
                            { key: 'sex', value: 0.480315 }
                        ],
                    },
                    {
                        'group': 4,
                        'portrait': [
                            { key: 'school', value: 5.445736 },
                            { key: 'grade', value: 52.096899 },
                            { key: 'bindcash', value: 329119.1364 },
                            { key: 'deltatime', value: 51950550 },
                            { key: 'combatscore', value: 123863.2016 },
                            { key: 'sex', value: 1.154264 }
                        ],
                    },
                    {
                        'group': 5,
                        'portrait': [
                            { key: 'school', value: 5.590909 },
                            { key: 'grade', value: 51.212121 },
                            { key: 'bindcash', value: 276306.2273 },
                            { key: 'deltatime', value: 134181700 },
                            { key: 'combatscore', value: 118014.4697 },
                            { key: 'sex', value: 1.212121 }
                        ],
                    },
                    {
                        'group': 6,
                        'portrait': [
                            { key: 'school', value: 5.401909 },
                            { key: 'grade', value: 45.738973 },
                            { key: 'bindcash', value: 202073.0349 },
                            { key: 'deltatime', value: 20432950 },
                            { key: 'combatscore', value: 86203.43235 },
                            { key: 'sex', value: 0.796577 }
                        ],
                    },
                ],
            ],
            test_clu_data: [
                [
                    {
                        'group': 0,
                        'portrait': [
                            { key: 'school', value: 5.41162 },
                            { key: 'grade', value: 45.75241 },
                            { key: 'bindcash', value: 222656.64890 },
                            { key: 'deltatime', value: 20729751.37552 },
                            { key: 'combatscore', value: 86787.54505 },
                            { key: 'sex', value: 0.92400 }
                        ],
                    },
                    {
                        'group': 1,
                        'portrait': [
                            { key: 'school', value: 5.41138 },
                            { key: 'grade', value: 23.57410 },
                            { key: 'bindcash', value: 50388.52391 },
                            { key: 'deltatime', value: 2787520.91221 },
                            { key: 'combatscore', value: 26599.71718 },
                            { key: 'sex', value: 0.28802 }
                        ],
                    },
                    {
                        'group': 2,
                        'portrait': [
                            { key: 'school', value: 5.457275 },
                            { key: 'grade', value: 51.838337 },
                            { key: 'bindcash', value: 352469.815242 },
                            { key: 'deltatime', value: 39535092.37875 },
                            { key: 'combatscore', value: 125174.97229 },
                            { key: 'sex', value: 1.277136 }
                        ],
                    },
                    {
                        'group': 3,
                        'portrait': [
                            { key: 'school', value: 5.491607 },
                            { key: 'grade', value: 49.578897 },
                            { key: 'bindcash', value: 293764.69257 },
                            { key: 'deltatime', value: 29805399.04077 },
                            { key: 'combatscore', value: 107077.72998 },
                            { key: 'sex', value: 1.14388 }
                        ],
                    },
                    {
                        'group': 4,
                        'portrait': [
                            { key: 'school', value: 5.594203 },
                            { key: 'grade', value: 46.463768 },
                            { key: 'bindcash', value: 209409.043478 },
                            { key: 'deltatime', value: 41222623.18841 },
                            { key: 'combatscore', value: 95554.507246 },
                            { key: 'sex', value: 1 }
                        ],
                    },
                    {
                        'group': 5,
                        'portrait': [
                            { key: 'school', value: 5.453599 },
                            { key: 'grade', value: 51.734605 },
                            { key: 'bindcash', value: 370059.20399 },
                            { key: 'deltatime', value: 36822158.85417 },
                            { key: 'combatscore', value: 121429.62847 },
                            { key: 'sex', value: 1.29948 }
                        ],
                    },
                    {
                        'group': 6,
                        'portrait': [
                            { key: 'school', value: 5.40297 },
                            { key: 'grade', value: 37.90766 },
                            { key: 'bindcash', value: 133681.36149 },
                            { key: 'deltatime', value: 10598467.58350 },
                            { key: 'combatscore', value: 56858.40275 },
                            { key: 'sex', value: 0.62213 }
                        ],
                    },
                ],
                [
                    {
                        'group': 0,
                        'portrait': [
                            { key: 'school', value: 5.42852 },
                            { key: 'grade', value: 24.93507 },
                            { key: 'bindcash', value: 54436.24882 },
                            { key: 'deltatime', value: 2835597.62231 },
                            { key: 'combatscore', value: 28801.67307 },
                            { key: 'sex', value: 0.300785 }
                        ],
                    },
                    {
                        'group': 1,
                        'portrait': [
                            { key: 'school', value: 5.49397 },
                            { key: 'grade', value: 47.96036 },
                            { key: 'bindcash', value: 240783.57808 },
                            { key: 'deltatime', value: 23055310.23785 },
                            { key: 'combatscore', value: 97952.07446 },
                            { key: 'sex', value: 0.95726 }
                        ],
                    },
                    {
                        'group': 2,
                        'portrait': [
                            { key: 'school', value: 5.48316 },
                            { key: 'grade', value: 52.00146 },
                            { key: 'bindcash', value: 333662.56808 },
                            { key: 'deltatime', value: 39495534.40703 },
                            { key: 'combatscore', value: 125279.65007 },
                            { key: 'sex', value: 1.22401 }
                        ],
                    },
                    {
                        'group': 3,
                        'portrait': [
                            { key: 'school', value: 5.51128 },
                            { key: 'grade', value: 51.81955 },
                            { key: 'bindcash', value: 311157.25564 },
                            { key: 'deltatime', value: 41612278.19549 },
                            { key: 'combatscore', value: 122225.29323 },
                            { key: 'sex', value: 1.25564 }
                        ],
                    },
                    {
                        'group': 4,
                        'portrait': [
                            { key: 'school', value: 5.47141 },
                            { key: 'grade', value: 40.79585 },
                            { key: 'bindcash', value: 146972.18509 },
                            { key: 'deltatime', value: 11773219.18679 },
                            { key: 'combatscore', value: 65892.57730 },
                            { key: 'sex', value: 0.67154 }
                        ],
                    },
                    {
                        'group': 5,
                        'portrait': [
                            { key: 'school', value: 5.166667 },
                            { key: 'grade', value: 43.333333 },
                            { key: 'bindcash', value: 140403.1667 },
                            { key: 'deltatime', value: 22267833.33333 },
                            { key: 'combatscore', value: 64489.33333 },
                            { key: 'sex', value: 0.833333 }
                        ],
                    },
                    {
                        'group': 6,
                        'portrait': [
                            { key: 'school', value: 5.41425 },
                            { key: 'grade', value: 51.29720 },
                            { key: 'bindcash', value: 326425.27634 },
                            { key: 'deltatime', value: 32899616.79389 },
                            { key: 'combatscore', value: 118806.75369 },
                            { key: 'sex', value: 1.16539 }
                        ],
                    },
                ],
                [
                    {
                        'group': 0,
                        'portrait': [
                            { key: 'school', value: 5.43298 },
                            { key: 'grade', value: 23.083058 },
                            { key: 'bindcash', value: 50327.47284 },
                            { key: 'deltatime', value: 2266922.16870 },
                            { key: 'combatscore', value: 27229.14707 },
                            { key: 'sex', value: 0.24332 }
                        ],
                    },
                    {
                        'group': 1,
                        'portrait': [
                            { key: 'school', value: 5.496 },
                            { key: 'grade', value: 49.84854 },
                            { key: 'bindcash', value: 281328.99530 },
                            { key: 'deltatime', value: 24847796.33114 },
                            { key: 'combatscore', value: 109794.61853 },
                            { key: 'sex', value: 1.02305 }
                        ],
                    },
                    {
                        'group': 2,
                        'portrait': [
                            { key: 'school', value: 5.469945 },
                            { key: 'grade', value: 52.32969 },
                            { key: 'bindcash', value: 305528.6521 },
                            { key: 'deltatime', value: 37313005.46448 },
                            { key: 'combatscore', value: 128218.3588 },
                            { key: 'sex', value: 1.169399 }
                        ],
                    },
                    {
                        'group': 3,
                        'portrait': [
                            { key: 'school', value: 5.55174 },
                            { key: 'grade', value: 37.39046 },
                            { key: 'bindcash', value: 113491.72292 },
                            { key: 'deltatime', value: 7521700.08084 },
                            { key: 'combatscore', value: 55235.32013 },
                            { key: 'sex', value: 0.47999 }
                        ],
                    },
                    {
                        'group': 4,
                        'portrait': [
                            { key: 'school', value: 5.44710 },
                            { key: 'grade', value: 52.10039 },
                            { key: 'bindcash', value: 330076.06486 },
                            { key: 'deltatime', value: 31477488.80309 },
                            { key: 'combatscore', value: 123915.60154 },
                            { key: 'sex', value: 1.15521 }
                        ],
                    },
                    {
                        'group': 5,
                        'portrait': [
                            { key: 'school', value: 5.590909 },
                            { key: 'grade', value: 51.212121 },
                            { key: 'bindcash', value: 276306.2273 },
                            { key: 'deltatime', value: 40226757.57576 },
                            { key: 'combatscore', value: 118014.4697 },
                            { key: 'sex', value: 1.212121 }
                        ],
                    },
                    {
                        'group': 6,
                        'portrait': [
                            { key: 'school', value: 5.401909 },
                            { key: 'grade', value: 45.75173 },
                            { key: 'bindcash', value: 202194.56380 },
                            { key: 'deltatime', value: 16612558.85262 },
                            { key: 'combatscore', value: 86245.26970 },
                            { key: 'sex', value: 0.79789 }
                        ],
                    },
                ],
                [
                    {
                        'group': 0,
                        'portrait': [
                            { key: 'school', value: 5.47815 },
                            { key: 'grade', value: 26.387046 },
                            { key: 'bindcash', value: 61832.21883 },
                            { key: 'deltatime', value: 2916102 },
                            { key: 'combatscore', value: 33084.905440 },
                            { key: 'sex', value: 0.282775 }
                        ],
                    },
                    {
                        'group': 1,
                        'portrait': [
                            { key: 'school', value: 5.494194 },
                            { key: 'grade', value: 52.937419 },
                            { key: 'bindcash', value: 327024.1381 },
                            { key: 'deltatime', value: 47788790 },
                            { key: 'combatscore', value: 128708.7071 },
                            { key: 'sex', value: 1.149677 }
                        ],
                    },
                    {
                        'group': 2,
                        'portrait': [
                            { key: 'school', value: 5.431849 },
                            { key: 'grade', value: 43.049523 },
                            { key: 'bindcash', value: 169325.5913 },
                            { key: 'deltatime', value: 14161270 },
                            { key: 'combatscore', value: 76025.72626 },
                            { key: 'sex', value: 0.643344 }
                        ],
                    },
                    {
                        'group': 3,
                        'portrait': [
                            { key: 'school', value: 5.4 },
                            { key: 'grade', value: 46.35 },
                            { key: 'bindcash', value: 155648.025 },
                            { key: 'deltatime', value: 160005500 },
                            { key: 'combatscore', value: 87840.1 },
                            { key: 'sex', value: 0.825 }
                        ],
                    },
                    {
                        'group': 4,
                        'portrait': [
                            { key: 'school', value: 5.444106 },
                            { key: 'grade', value: 49.517871 },
                            { key: 'bindcash', value: 258850.3654 },
                            { key: 'deltatime', value: 29063350 },
                            { key: 'combatscore', value: 106455.2844 },
                            { key: 'sex', value: 0.925095 }
                        ],
                    },
                    {
                        'group': 5,
                        'portrait': [
                            { key: 'school', value: 5.487696 },
                            { key: 'grade', value: 54.357942 },
                            { key: 'bindcash', value: 360054.7136 },
                            { key: 'deltatime', value: 78763330 },
                            { key: 'combatscore', value: 140946.5257 },
                            { key: 'sex', value: 1.268456 }
                        ],
                    },
                    {
                        'group': 6,
                        'portrait': [
                            { key: 'school', value: 5.333333 },
                            { key: 'grade', value: 38 },
                            { key: 'bindcash', value: 161953.6667 },
                            { key: 'deltatime', value: 414790700 },
                            { key: 'combatscore', value: 70855 },
                            { key: 'sex', value: 0.333333 }
                        ],
                    },
                ],
            ],

            //groupTemplate & link
            show_flag: [],
            rec_show_flag: {},
            radius: 50,
            offset_x: 38,
            offset_y: 0,
            delta_x: 14,
            day_height: 205,
            hidden_height: 416,
            cluster_num: 7,
            pos_list: [],
            line_list: [],
            true_num: 0,
            chosen_days: '',
            // line_color: [
            //     '#666a53',
            //     '#d27d1c',
            //     '#222014',
            //     '#72a6c1',
            //     '#b23722',
            //     '#a7805d',
            //     '#d5bf67',
            // ],
            line_color:[
                '#555775',//1
                '#528d6a',//2
                '#c0d06b',//3
                '#ebd168',//4
                '#edc0ba',//5
                "#aa5d7b",//6
                '#30a3b1',//7
            ],
            pos_flag: false,

            //link rec
            rec_link: [],
            rec_link_up: [],
            link_group: 0,

            //cluster data
            cluster: '',

            //overall range
            overall_range: [],

            //test range    
            test_range: '',
            cf_range: '',
            table_range: '',

            //time range
            time_range: [0, 3],
            // time_range: '',

            //obse data
            obse_box_data: [],
            obse_chn_data: [],

            //record step range
            step_range: '',
            step_cf_range: '',
            step_table_range: '',
            step_change: '',

            link_info: [
                // [{
                //     clu: 0, link: [
                //         { tar: 0, value: 10 },
                //         // { tar: 1, value: 20 },
                //     ]
                // },
                // {
                //     clu: 1, link: [
                //         { tar: 0, value: 20 },
                //     ]
                // },],
                // [],
                [{
                    clu: 0, link: [
                        { tar: 1, value: 20 },
                    ]
                },
                {
                    clu: 1, link: [
                        { tar: 0, value: 10 },
                    ]
                },
                ],
                [{
                    clu: 1, link: [
                        { tar: 1, value: 30 },
                        { tar: 3, value: 20 },
                    ]
                },
                ],
            ],
            copy_link_info: [],
            filter_link_info: [],
            //slider value
            slider_value: '',
            link_per_range: [],
            glo_link_num_range: [],
            slider_min: 0,
            slider_max: 100,
            filter_flag: false,
            filter_finish_flag: false,

            // counter_module
            counter_modules: [
                {
                    value: '0',
                    label: 'Exchange'
                },
                {
                    value: '1',
                    label: 'Increase'
                },
                {
                    value: '2',
                    label: 'Decrease'
                }
            ],
            module_value: 'Exchange',
            selected_module: 0,
        };
    },
    mounted() {
        this.plot_clus_legend()

        // show template
        this.init_showflag()
        this.$bus.$on("template to group", (msg) => {
            // save msg to show_flag, including idx and show flag
            var id = msg[0], flag = msg[1]
            this.show_flag[id] = flag

            // save rec to rec_show_flag
            this.rec_show_flag[id] = flag

            // initialize true_num
            this.true_num = 0

            // count true numbers
            for (var key in this.rec_show_flag) {
                if (this.rec_show_flag[key] == true)
                    this.true_num += 1;
            }

            // wait some time
            if (flag == false) {
                setTimeout(() => this.plotlines(), 170)
            }
            // else
            setTimeout(() => this.plotlines(), 50)

        })
        if (this.pos_flag == false) {
            this.get_linepos()
            this.pos_flag = true
        }
        this.plotlines()

        this.wait_data()

        this.plot_ob_sequence()
        this.plot_obse_legend()
    },
    watch: {
        link_info(val, _oldVal) {
            this.init_showflag()
            this.$bus.$on("template to group", (msg) => {
                // save msg to show_flag, including idx and show flag
                var id = msg[0], flag = msg[1]
                this.show_flag[id] = flag

                // save rec to rec_show_flag
                this.rec_show_flag[id] = flag

                // initialize true_num
                this.true_num = 0

                // count true numbers
                for (var key in this.rec_show_flag) {
                    if (this.rec_show_flag[key] == true)
                        this.true_num += 1;
                }

                // wait some time
                if (flag == false) {
                    setTimeout(() => this.plotlines(), 170)
                }
                // else
                setTimeout(() => this.plotlines(), 50)

            })
            if (this.pos_flag == false) {
                this.get_linepos()
                this.pos_flag = true
            }

            this.plotlines()
        },
        async step_range(val, _oldVal) {
            console.log('GROUP >> STEP RANGE >> step range', this.step_range)

            // GET CF RANGE, TABLE RANGE & CHANGE RANGE
            this.step_table_range = this.deepCopy(this.step_range)
            for (var day in this.step_table_range) {
                delete this.step_table_range[day].change
            }
            // console.log('updated step table range', this.step_table_range)

            this.step_change = {}
            for (var day in this.step_range) {
                this.step_change[day] = {}
                if (typeof (this.step_range[day].change) == 'undefined')
                    this.step_change[day]['change'] = []
                else
                    this.step_change[day]['change'] = this.step_range[day].change
            }
            console.log('GROUP >> STEP RANGE >> step change', this.step_change)

            //LOADING  
            const rLoading = this.openLoading()

            ///// UPDATE ALL DATA
            //1. TABLE & BOX -- FLITER
            var table_data = await HttpHelper.axiosPost('/table',
                this.step_table_range, 600000)
            this.$bus.$emit("template to table", table_data);

            var boxplot_data = await HttpHelper.axiosPost('/boxplot',
                this.step_table_range, 600000)
            this.$bus.$emit("template to boxplot", boxplot_data);
            console.log('GROUP >> STEP RANGE >> TABLE UPDATED.')

            //2. GROUP & LINK
            var group_data = await HttpHelper.axiosPost('/group', this.time_range)
            console.log('GROUP >> STEP RANGE >> GROUP UPDATED.')

            // if only choose one day, no link
            if (this.time_range[0] == this.time_range[1]) {
                // do nothing
            }
            // if choose multiple days, send back to backend
            else {
                var link_time_range = [this.time_range[0], this.time_range[1] - 1]
                var link_data = await HttpHelper.axiosPost('/link', link_time_range)
                this.link_info = link_data
                this.copy_link() // copy
                console.log('GROUP >> STEP RANGE >> LINK UPDATED.')
            }

            //3. CF
            var cf_data = await HttpHelper.axiosPost('/cf',
                { 'setting': this.step_range, 'split_num': 5, 'target': this.selected_module}, 600000)
            console.log('GROUP >> STEP RANGE >> CF UPDATED.')

            cf_data = d3.nest()
                .key(d => d.id)
                .entries(cf_data)

            group_data.forEach((da, i) => {
                var clu_group = da.id
                // add cf data
                da.cf = cf_data[i].values
                da.cf_change = this.step_change[da.id].change

                // add local portrait data
                da.values.forEach((val, j) => {
                    val.portrait = this.clu_data[clu_group][j].portrait
                })
            })

            console.log('GROUP >> STEP RANGE >> group_data', group_data)

            this.data = group_data
            this.updated_flag = true

            rLoading.close()
        },
        filter_finish_flag(val, _oldVal) {
            if (this.filter_finish_flag == true) {
                this.plotlines()
                this.filter_finish_flag = false
            }
        }
    },
    methods: {
        plot_clus_legend() {
            var svg = d3.select('#clusLegend')

            var data = [
                { name: 'clus1', color: '#c4c6d5', stroke: '#9799b6',},
                { name: 'clus2', color: '#b7dbc5', stroke: '#78c497',},
                { name: 'clus3', color: '#dce2b0', stroke: '#c0d068',},
                { name: 'clus4', color: '#f3e4ae', stroke: '#ebd168',},
                { name: 'clus5', color: '#f3dbd8', stroke: '#edc0ba',},
                { name: 'clus6', color: '#e5cbf1', stroke: "#d8a0eb",},
                { name: 'clus7', color: '#badce2', stroke: '#7fc4cd',},
            ]
            var width = 20

            var each_status = svg.append('g')
                .selectAll('g')
                .data(data)
                .enter()
                .append('g')
                .attr("transform", (d, i) => `translate(${15 + i * 90},${20})`)

            var each_rect = each_status.append('circle')
                .attr('r', 10)
                .attr('fill', d => d.color)
                .attr('stroke', d => d.stroke)
                .attr('stroke-width', 1)

            each_status
                .append('text')
                .attr('x', (d, i) => 37)
                .attr('y', 5)
                .attr('text-anchor', 'middle')
                .text(d => d.name)
        },
        // get pos of every clusters and save them in _this.pos_list
        get_linepos() {
            let _this = this;
            _this.chosen_days = _this.data.length
            console.log(_this.chosen_days)

            for (var i = _this.time_range[0]; i < _this.time_range[0] + _this.chosen_days; i++) {
                var day_pos_list = []
                for (var j = 0; j < _this.cluster_num; j++) {
                    var cir = d3.select('.id' + i + 'portrait' + j).node().getBoundingClientRect()
                    var x = cir.x + _this.radius + _this.offset_x
                    var y = cir.y
                    var pos = [x, y]
                    day_pos_list.push(pos)
                }
                _this.pos_list.push(day_pos_list)
            }
        },
        // save source and target based on link info
        get_line() {
            let _this = this;

            // for each day
            for (var i = 0; i < _this.copy_link_info.length; i++) {
                //link
                var day_line = []
                var day_cluster = _this.copy_link_info[i]

                //rec
                var day_rec = {}

                // for each clusters
                for (var j = 0; j < day_cluster.length; j++) {
                    var clu = day_cluster[j]
                    var source_clu = clu.clu

                    //rec
                    // var clu_link = clu.link
                    var clu_rec = []

                    // for each target
                    for (var k = 0; k < clu.link.length; k++) {
                        var target_info = clu.link[k]
                        var clu_info = {}
                        clu_info.source = source_clu
                        clu_info.target = target_info.tar
                        clu_info.value = target_info.value
                        // deal with delta
                        if (clu.link.length % 2 == 0) {
                            var clu_info_delta = (k + 1) <= clu.link.length / 2 ? (k + 1 - clu.link.length / 2 - 0.5) : (k + 1 - clu.link.length / 2 - 0.5)
                        }
                        else if (clu.link.length % 2 == 1) {
                            var clu_info_delta = k - Math.floor(clu.link.length / 2)
                        }
                        clu_info.delta = clu_info_delta
                        day_line.push(clu_info)

                        //rec
                        var link = clu.link[k]
                        var rec = [source_clu, link.tar, { group: [] }]
                        clu_rec.push(rec)
                    }
                    day_rec[source_clu] = clu_rec
                }
                _this.line_list.push(day_line)
                _this.rec_link.push(day_rec)
            }
        },
        get_line_up() {
            let _this = this
            // for link info
            var rec_link_up = []
            for (var i = 0; i < _this.copy_link_info.length; i++) {
                var day_link_info = _this.copy_link_info[i]
                var rec_day_link = {}

                // for clusters every day
                for (var j = 0; j < day_link_info.length; j++) {
                    var cluster_link = day_link_info[j].link
                    var cluster_name = day_link_info[j].clu
                    // for links in cluster
                    for (var k = 0; k < cluster_link.length; k++) {
                        var link = cluster_link[k]
                        var tar = link.tar
                        if (typeof (rec_day_link[tar]) == 'undefined') {
                            rec_day_link[tar] = []
                        }
                        rec_day_link[tar].push([tar, cluster_name, { group: [] }])
                    }
                }
                rec_link_up.push(rec_day_link)
            }
            _this.rec_link_up = rec_link_up
        },
        rec_lines() {
            let _this = this
            // console.log(_this.rec_link)

            // down
            for (var i = 0; i < _this.rec_link.length; i++) {
                var day_lines = _this.rec_link[i]
                // for clusters
                for (let clu in day_lines) {
                    var cluster_lines = day_lines[clu]
                    // for lines
                    for (var k = 0; k < cluster_lines.length; k++) {
                        var line = cluster_lines[k]
                        _this.link_group = i + _this.time_range[0] + "" + clu
                        _this.find_tar_down(i + _this.time_range[0], line)
                    }
                }
            }

            // up
            for (var i = 0; i < _this.rec_link_up.length; i++) {
                var day_lines = _this.rec_link_up[i]
                //for clusters
                for (let clu in day_lines) {
                    var cluster_lines = day_lines[clu]
                    // for lines
                    for (var k = 0; k < cluster_lines.length; k++) {
                        var line = cluster_lines[k]
                        _this.link_group = (i + _this.time_range[0] + 1) + '' + clu
                        _this.find_tar_up(i + _this.time_range[0], line)
                    }
                }
            }
        },
        find_tar_down(day, line) {
            let _this = this

            var line_tar = line[1]
            // add group
            line[2].group.push(_this.link_group)

            day = day + 1
            if (day == _this.time_range[0] + _this.rec_link.length) {
                return
            }

            var next_clu = _this.rec_link[day - _this.time_range[0]][line_tar]
            if (typeof (next_clu) == "undefined") {
                return
            }

            for (var i = 0; i < next_clu.length; i++) {
                var next_line = next_clu[i]
                this.find_tar_down(day, next_line)
            }

            // console.log(next_clu)
        },
        find_tar_up(day, line) {
            let _this = this

            var line_tar = line[1]

            // add group
            line[2].group.push(_this.link_group)

            day = day - 1
            if (day == _this.time_range[0] - 1) {
                return
            }

            var next_clu = _this.rec_link_up[day - _this.time_range[0]][line_tar]
            if (typeof (next_clu) == "undefined") {
                return
            }

            for (var i = 0; i < next_clu.length; i++) {
                var next_line = next_clu[i]
                _this.find_tar_up(day, next_line)
            }


        },
        change_pos() {
            var true_count = [], false_count = []
            //initialize true_count and false_count
            for (var i = 0; i < this.data.length; i++) {
                true_count.push(0)
                false_count.push(0)
            }

            for (var i = 0; i < this.show_flag.length; i++) {
                if (this.show_flag[i] == true) {
                    for (var j = i + 1; j < true_count.length; j++) {
                        true_count[j] += 1
                    }
                }
                if (this.show_flag[i] == false) {
                    for (var j = i + 1; j < false_count.length; j++) {
                        false_count[j] += 1
                    }
                }
            }

            var true_idx = this.show_flag.indexOf(true)
            var false_idx = this.show_flag.indexOf(false)

            // change pos
            // if true status exists
            if (true_idx > -1) {
                for (var i = 0; i < true_count.length; i++) {
                    var add_num = true_count[i]
                    this.show_flag[true_idx] = 'defalut'
                    if (add_num != 0) {
                        for (var j = 0; j < this.cluster_num; j++) {
                            this.pos_list[i][j][1] += add_num * this.hidden_height
                        }
                    }
                }
            }
            if (false_idx > -1) {
                for (var i = 0; i < true_count.length; i++) {
                    var sub_num = false_count[i]
                    this.show_flag[false_idx] = 'defalut'
                    if (sub_num != 0) {
                        for (var j = 0; j < this.cluster_num; j++) {
                            this.pos_list[i][j][1] -= sub_num * this.hidden_height
                        }
                    }
                }
            }

            this.lineSvgStyle.height = this.data.length * this.day_height + this.true_num * this.hidden_height

        },
        init_showflag() {
            for (var i = 0; i < this.data.length; i++) {
                this.show_flag.push('defalut')
            }
        },
        plotlines() {
            let _this = this;

            var svg = d3.select('#lineSvg')
            svg.selectAll('*').remove()
            _this.line_list = []
            _this.rec_link = []

            _this.get_line()
            _this.change_pos()

            _this.get_line_up()

            _this.rec_lines()
            // console.log('plot rec', _this.rec_link)

            var color = d3.scaleOrdinal()
                .domain([0, 1, 2, 3, 4, 5, 6])
                .range(_this.line_color)

            //scale value to width
            function scale_width(range) {
                return d3.scaleLinear()
                    .domain(range)
                    .range([0.25, 2])
            }

            //scale value to opacity
            function scale_opacity(range) {
                return d3.scaleLinear()
                    .domain(range)
                    .range([0.5, 0.5])
            }

            // personalize link function
            function link(sour, targ, width, clu) {
                var curvature = 0.8
                var offset = 2

                //deal with delta
                clu = clu - 3
                var x0 = sour[0] - width / 2,
                    y0 = sour[1],
                    // y0 = sour[1] + _this.offset_y,
                    x1 = targ[0] - width / 2 + clu * 15,
                    y1 = targ[1] - _this.radius * 2 - 105,
                    // y1 = targ[1] - _this.radius * 2 - 60,
                    yi = d3.interpolateNumber(y0, y1),
                    y2 = yi(curvature),
                    y3 = yi(1 - curvature)

                if (x0 <= x1) {
                    return "M" + (x0) + "," + y0
                        + "C" + (x0) + "," + (y2 + offset)
                        + " " + (x1) + "," + (y3 + offset)
                        + " " + (x1) + "," + y1
                        + "L" + (x1 + width) + "," + (y1)
                        + "C" + (x1 + width) + "," + (y3 - offset)
                        + " " + (x0 + width) + "," + (y2 - offset)
                        + " " + (x0 + width) + "," + (y0)
                        + "L" + (x0) + "," + y0;
                }
                else {
                    return "M" + (x0) + "," + y0
                        + "C" + (x0) + "," + (y2 - offset)
                        + " " + (x1) + "," + (y3 - offset)
                        + " " + (x1) + "," + y1
                        + "L" + (x1 + width) + "," + (y1)
                        + "C" + (x1 + width) + "," + (y3)
                        + " " + (x0 + width) + "," + (y2)
                        + " " + (x0 + width) + "," + (y0)
                        + "L" + (x0) + "," + y0;
                }
            }

            // mouse functions
            function clicked() {
                var mom = d3.select(this)
                var mom_id = mom.attr('id')
                svg.select('#' + mom_id)
                    .attr('opacity', 1)
            }

            // get value range for every days
            var day_value_range = []
            _this.line_list.map(day => {
                // get min and max of everyday value
                var val_range = d3.map(day, d => d.value).keys()
                val_range = val_range.map(Number)
                var min = d3.min(val_range), max = d3.max(val_range)

                day_value_range.push([min, max])
            })

            if (_this.filter_flag == false) {
                // get total min & max
                var glo_min = d3.map(day_value_range, d => d[0]).keys()
                glo_min = d3.min(glo_min.map(min => min * 1))
                var glo_max = d3.map(day_value_range, d => d[1]).keys()
                glo_max = d3.max(glo_max.map(max => max * 1))
                var glo_val_range = [glo_min, glo_max]
                _this.glo_link_num_range = glo_val_range
            }

            // for each day
            for (var i = 0; i < _this.line_list.length; i++) {
                var day_line = _this.line_list[i]

                // for cluster info in each day
                var count = 0, up_count = 0
                for (var j = 0; j < day_line.length; j++) {
                    var source = day_line[j].source, target = day_line[j].target
                    day_line[j].source_pos = _this.pos_list[i][source]
                    day_line[j].target_pos = _this.pos_list[i + 1][target]

                    // rec down
                    var per_rec_len = _this.rec_link[i][source].length
                    if (count > per_rec_len - 1) {
                        count = 0
                    }
                    var group = _this.rec_link[i][source][count][2].group
                    // rec up
                    var per_rec_up_len = _this.rec_link_up[i][target].length
                    if (up_count > per_rec_up_len - 1) {
                        up_count = 0
                    }
                    var up_group = _this.rec_link_up[i][target][up_count][2].group

                    // plot lines
                    var line = svg.append('path')
                        .attr('d', function () {
                            return link(day_line[j].source_pos,
                                day_line[j].target_pos,
                                15, day_line[j].source)
                        })
                        .attr('fill', color(day_line[j].source))
                        .attr('fill-opacity', scale_opacity(_this.glo_link_num_range)(day_line[j].value))
                        // .attr('fill-opacity', scale_opacity(day_value_range[i])(day_line[j].value))
                        .attr('stroke', color(day_line[j].source))
                        // .attr('stroke-width', scale_width(day_value_range[i])(day_line[j].value))
                        .attr('stroke-width', scale_width(_this.glo_link_num_range)(day_line[j].value))
                        .attr('stroke-opacity', 1)
                        .attr('opacity', function () {
                            return 0
                        })
                        .attr('class', function () {
                            var final_class = '';
                            for (var i = 0; i < group.length; i++) {
                                if (i == 0)
                                    final_class += '_group_' + group[i]
                                else
                                    final_class += " " + '_group_' + group[i]
                            }
                            for (var i = 0; i < up_group.length; i++) {
                                if (final_class == '')
                                    final_class += '_group_' + up_group[i]
                                else
                                    final_class += " " + '_group_' + up_group[i]
                            }
                            return final_class
                        })
                        .attr('id', 'id_' + i + '_sour_' + day_line[j].source)
                        .on('click', clicked)

                    count += 1

                }
            }

            // global opacity
            _this.$clicked_cluster.forEach(clu => {
                d3.selectAll(clu)
                    .attr('opacity', 1)
            })

        },
        async plot_ob_sequence() {
            let _this = this;

            //#region data
            data = []
            chnrate_data = []

            data = _this.obse_box_data.length == 0 ? data : _this.obse_box_data
            chnrate_data = _this.obse_chn_data.length == 0 ? chnrate_data : _this.obse_chn_data

            console.log('OBSE >> DATA', data)
            //#endregion

            //#region layout and default configs
            var svg = d3.select('#comparisonSvg')
            svg.selectAll('*').remove()

            var margin = { top: 15, right: 20, bottom: 20, left: 20 }

            // basic params
            var width = 1998, height = 305

            // box params
            var box_height = 180
            var text_height = 10
            const boxWidth = 10;
            const feature_group = data.map(d => d.key)
            var date_group = d3.map(data[0].values, d => d.id).keys()
            // box ranges for each feature
            var box_range = {}
            data.forEach(feat => {
                var val_min = 9999999999999, val_max = -9999999999999
                feat.values.forEach(ele =>{
                    var ran = ele.range
                    var ran_min = d3.min(ran)
                    var ran_max = d3.max(ran)
                    val_min = ran_min < val_min ? ran_min : val_min
                    val_max = ran_max > val_max ? ran_max : val_max
                })
                var cf_min = 9999999999999, cf_max = -9999999999999
                feat.counter_values.forEach(ele =>{
                    var ran = ele.range
                    var ran_min = d3.min(ran)
                    var ran_max = d3.max(ran)
                    cf_min = ran_min < cf_min ? ran_min : cf_min
                    cf_max = ran_max > cf_max ? ran_max : cf_max
                })
                var min = val_min < cf_min ? val_min : cf_min
                var max = val_max > cf_max ? val_max : cf_max
                var box_ran = [min, max]

                box_range[feat.key] = box_ran
            })
            console.log('OBSE >> BOX RANGE', box_range)

            // step params
            const step_width = (width - margin.left - margin.right) / chnrate_data.length,
                step_height = 150 //ori -> 130
            const padding = 25
            var pie_r = 25, pie_outer_r = 30
            // step y --> churn rate range
            var min = 2, max = -1, num_min = 9999999, num_max = -9999999
            var met = []
            chnrate_data.forEach(day => {
                // y range
                var day_min = d3.min(day.values, d => d.churn_rate)
                var day_max = d3.max(day.values, d => d.churn_rate)
                min = day_min < min ? day_min : min
                max = day_max > max ? day_max : max

                // num range
                var tmp_num_min = d3.min(day.values, d => d.num)
                var tmp_num_max = d3.max(day.values, d => d.num)
                num_min = tmp_num_min < num_min ? tmp_num_min : num_min
                num_max = tmp_num_max > num_max ? tmp_num_max : num_max

                // metric
                met.push(day.metric)
            })
            var step_y_range = [min, max]
            var num_range = [num_min, num_max]
            // step metrics
            var met_range = [d3.min(met), d3.max(met)]

            //color
            var color_group = ['#5E1675',
                '#EE4266',
                '#FFD23F',
                '#337357',]
            var pie_color = [
                '#858976',
                '#db9749',
                '#6d9ac2',
                '#c15f4e',
            ]
            var pie_stroke_color = [
                '#676B54',
                '#D27D1C',
                '#4881B3',
                '#B23722',
            ]

            //#endregion

            //#region functions

            // #region box plot functions
            var x = d3.scaleBand()
                .range([0, width])
                .domain(feature_group)
                .padding(0.05)

            var group_x = d3.scaleBand()
                .domain(date_group)
                .range([0, x.bandwidth()])
                .padding(1)

            function y(feat, da) {
                var idx = feature_group.indexOf(feat)
                return d3.scaleLinear()
                    .domain(box_range[feat])
                    .range([height - margin.bottom, step_height + padding])
            }

            var xAxis = g => g
                .attr("transform", `translate(${0}, ${height - margin.bottom + text_height})`)
                .call(d3.axisBottom(x).ticks(null, "s").tickSize(0))
                .attr("class", "axis")

            //change text size
            svg.selectAll('.axis')
                .selectAll('text')
                .style('font-size', 16)
                .style('opacity', 1)

            function yAxis(feat) {
                return g => g
                    .attr("transform", `translate(${0},${height - margin.bottom})`)
                    .call(d3.axisBottom(y(feat)))
            }
            // #endregion

            // #region step functions
            function step_day_x(domain) {
                return d3.scaleBand()
                    .domain(domain)
                    .range([0, step_width])
            }

            var step_churn_y = d3.scaleLinear()
                .domain(step_y_range)
                .range([step_height - pie_outer_r, pie_outer_r + 10])

            var pie = d3.pie()
                // .padAngle(0.05)
                .sort(null)
                .value(d => d.value)

            var arc = d3.arc()
                .innerRadius(1.6)
                .outerRadius(pie_r)
                .padAngle(0.8)
                .padRadius(4)
                .cornerRadius(1)

            var num_arc = d3.arc()
                .innerRadius(pie_r + 3)
                .outerRadius(pie_outer_r)
                .padAngle(0.8)
                .padRadius(4)
                .cornerRadius(1)

            var scale_metrics = d3.scaleLinear()
                .domain(met_range)
                .range([0.05, 0.3])

            var scale_nums = d3.scaleLinear()
                .domain(num_range)
                .range([0.2, 1])

            var yAxis = g => g
                .attr("transform", `translate(${margin.left + 20},${0})`)
                .call(d3.axisLeft(step_churn_y)
                    .ticks(3)
                    .tickSize(2)
                    .tickSizeOuter(0)
                )

            // #endregion

            // #region color
            var color = d3.scaleOrdinal()
                .domain(date_group)
                .range(color_group)
            // #endregion

            // #region mouse functions
            function pie_overed(event, d) {
                var mom = d3.select(this)
                var mom_class = mom.attr("class")
                //pie + day
                var day = mom_class.substring(8)

                // pie
                svg.selectAll('.pie' + day)
                    .attr('stroke-width', 1.75)
            }

            function pie_outed(event, d) {
                var mom = d3.select(this)
                var mom_class = mom.attr("class")
                //pie + day
                var day = mom_class.substring(8)

                // pie
                svg.selectAll('.pie' + day)
                    .attr('stroke-width', 1)
            }

            async function button_clicked(event, d) {
                var mom = d3.select(this)
                var mom_class = mom.attr("class")
                console.log(mom_class)
                // get step
                var step = mom_class.substring(3)

                // get step_range
                _this.step_range = await HttpHelper.axiosPost('/getStep',
                    0, 600000)
            }

            // #endregion

            //#endregion

            //#region plot real boxplots on the left
            const groups = svg.selectAll("g")
                .data(data)
                .enter()
                .append('g')
                .attr("transform", d => `translate(${x(d.key)}, ${0})`)
                .attr("class", d => d.key);

            var day_groups = groups.append('g')
                .selectAll('g')
                .data(d => {
                    return d.values
                })
                .enter()
                .append('g')
                .attr("transform", (d, i) => `translate(${0}, ${0})`)

            day_groups
                .selectAll("vertLine")
                .data(d => [d])
                .enter()
                .append('line')
                .attr("class", "vertLine")
                .attr("stroke", "#C0C0C0")
                .attr('stroke-width', '1px')
                .attr("x1", d => group_x(d.id))
                .attr("x2", d => group_x(d.id))
                .attr("y1", d => {
                    return y(d.key, d)(d.range[0])
                })
                .attr("y2", d => y(d.key, d)(d.range[1]))
                .attr("class", d => d.id)

            day_groups
                .selectAll("box")
                .data(d => [d])
                .enter()
                .append('rect')
                .attr("class", "box")
                .attr("y", d => {
                    return y(d.key, d)(d.quartiles[2])
                })
                // .attr("x", d => group_x(d.id) - boxWidth / 2)
                .attr("x", d => group_x(d.id) - boxWidth / 2)
                .attr("width", boxWidth / 2)
                .attr("height", d => y(d.key, d)(d.quartiles[0]) - y(d.key, d)(d.quartiles[2]))
                .attr("stroke", "#808080")
                .style("fill", d => color(d.id))
                .style("fill-opacity", 0.3)
                .attr("class", d => d.id)

            day_groups
                .selectAll("verticalLine")
                .data(d => {
                    return [
                        { id: d.id, key: d.key, value: d.range[0], range: d.range },
                        { id: d.id, key: d.key, value: d.quartiles[1], range: d.range },
                        { id: d.id, key: d.key, value: d.range[1], range: d.range }
                    ]
                })
                .enter()
                .append('line')
                .attr("class", "verticalLine")
                .attr("stroke", "#808080")
                .attr('stroke-width', '1px')
                .attr("y1", d => {
                    return y(d.key, d)(d.value)
                })
                .attr("y2", d => y(d.key, d)(d.value))
                .attr("x1", d => group_x(d.id) - boxWidth / 2)
                .attr("x2", d => group_x(d.id))
                // .attr("x2", d => group_x(d.id) + boxWidth / 2)
                .attr("class", d => d.id)

            svg.append("g")
                .call(xAxis);

            svg.select('.domain').remove()

            //#endregion

            //#region plot counter boxplots on the right
            var counter_day_groups = groups.append('g')
                .selectAll('g')
                .data(d => {
                    return d.counter_values
                })
                .enter()
                .append('g')
                .attr("transform", (d, i) => `translate(${0}, ${0})`)

            counter_day_groups
                .selectAll("vertLine")
                .data(d => [d])
                .enter()
                .append('line')
                .attr("class", "vertLine")
                .attr("stroke", "#C0C0C0")
                .attr('stroke-width', '1px')
                .attr("x1", d => group_x(d.id))
                .attr("x2", d => group_x(d.id))
                .attr("y1", d => {
                    return y(d.key, d)(d.range[0])
                })
                .attr("y2", d => y(d.key, d)(d.range[1]))
                .attr("class", d => d.id)

            counter_day_groups
                .selectAll("box")
                .data(d => [d])
                .enter()
                .append('rect')
                .attr("class", "box")
                .attr("y", d => {
                    return y(d.key, d)(d.quartiles[2])
                })
                .attr("x", d => group_x(d.id))
                .attr("width", boxWidth / 2)
                .attr("height", d => y(d.key, d)(d.quartiles[0]) - y(d.key, d)(d.quartiles[2]))
                .attr("stroke", "#808080")
                .style("fill", d => color(d.id))
                .style("fill-opacity", 0.3)
                .attr("class", d => d.id)

            counter_day_groups
                .selectAll("verticalLine")
                .data(d => {
                    return [
                        { id: d.id, key: d.key, value: d.range[0], range: d.range, change: d.change },
                        { id: d.id, key: d.key, value: d.quartiles[1], range: d.range, change: d.change },
                        { id: d.id, key: d.key, value: d.range[1], range: d.range, change: d.change }
                    ]
                })
                .enter()
                .append('line')
                .attr("class", "verticalLine")
                .attr("stroke", "#808080")
                .attr('stroke-width', '1px')
                .attr("y1", d => {
                    return y(d.key, d)(d.value)
                })
                .attr("y2", d => y(d.key, d)(d.value))
                .attr("x1", d => group_x(d.id))
                .attr("x2", d => group_x(d.id) + boxWidth / 2)
                .attr("class", d => d.id)

            // add extra background color
            counter_day_groups
                .selectAll("counter_bgbox")
                .data(d => [d])
                .enter()
                .append('rect')
                .attr("class", "box")
                .attr("y", d => {
                    // return y(d.key, d)(d.quartiles[2])
                    return y(d.key, d)(d.range[1])
                })
                .attr("x", d => group_x(d.id))
                .attr("width", boxWidth / 2)
                .attr("height", d => {
                    var bg_height = y(d.key, d)(d.range[0]) - y(d.key, d)(d.range[1])
                    return bg_height
                })
                .style("fill", d => {
                    //NEED CHANGE
                    if (d.change == 1) {
                        return color(d.id)
                    }
                    else {
                        return ''
                    }
                })
                .style("fill-opacity", d => {
                    if (d.change == 1) {
                        return 0.2
                    }
                    else {
                        return 0
                    }
                })
                .attr("class", d => d.id)

            //#endregion

            //#region plot churn rate!

            // #region plot area
            var area_pos = []
            for (var i = 0; i < chnrate_data.length; i++) {
                for (var j = 0; j < chnrate_data[i].values.length; j++) {
                    var per_d = chnrate_data[i].values[j]

                    var start_date = chnrate_data[i].values[0].date

                    // x pos
                    var day_domain = [start_date - 1]
                    for (var k = start_date; k < start_date + per_d.day_len; k++)
                        day_domain.push(k)
                    var x_pos = step_day_x(day_domain)(per_d.date) + margin.left +20 + per_d.step * step_width

                    // y pos
                    var y_pos = step_churn_y(per_d.churn_rate)

                    if (j == 0 && i != 0) {
                        var prev = area_pos.slice(-1)
                        var prev_x = prev[0].x, prev_y = prev[0].y
                        var split_x = i * step_width + margin.left
                        var split_y = (split_x - prev_x) * (y_pos - prev_y) / (x_pos - prev_x) + prev_y
                        area_pos.push({ x: split_x, y: split_y, step: i - 1 })
                        area_pos.push({ x: split_x, y: split_y, step: i })
                    }

                    // add first and last pos
                    if (i == 0 && j == 0)
                        area_pos.push({ x: margin.left + 20, y: y_pos, step: i })

                    area_pos.push({ x: x_pos, y: y_pos, step: i })

                    if (i == chnrate_data.length - 1 && j == chnrate_data[i].values.length - 1)
                        area_pos.push({ x: (i + 1) * step_width, y: y_pos, step: i })
                }
            }

            var line = d3.line()
                .defined(function (d) { return d; })
                .x(function (d) { return d.x; })
                .y(function (d) { return margin.top + d.y; });

            var area = d3.area()
                .defined(line.defined())
                .x(line.x())
                .y1(line.y())
                .y0(step_height + margin.top);

            area_pos = d3.nest()
                .key(d => d.step)
                .entries(area_pos)
            
            console.log('met', met)

            area_pos.forEach(step => {
                var vals = step.values
                var area_g = svg.append('g')
                    .datum(vals)
                    console.log('step',step)

                area_g.append("path")
                    .attr("class", "area")
                    .attr("d", area)
                    .attr('fill', '#72A6C1')
                    .attr('fill-opacity', d => {
                        return scale_metrics(met[step.key])
                    })

                area_g.append("path")
                    .attr("class", "line")
                    .attr("d", line)
                    .attr('fill', 'none')
                    .attr('stroke', '#72A6C1')
            })
            // #endregion

            var step = svg.append('g')
                .selectAll('g')
                .data(chnrate_data)
                .enter()
                .append('g')
                .attr('transform', (d, i) => `translate(${margin.left + 20+ i * step_width},${margin.top})`)

            // #region plot pie
            var each_day = step.append('g')
                .selectAll('g')
                .data(d => {
                    var day_len = d.values.length
                    d.values.forEach(val => {
                        val.day_len = day_len
                    })
                    return d.values
                })
                .enter()
                .append('g')
                .attr('transform', (d, i) => {
                    // x pos
                    var day_domain = [-1]
                    for (var j = 0; j < d.day_len; j++)
                        day_domain.push(j)
                    var x_pos = step_day_x(day_domain)(i)

                    // y pos
                    var y_pos = step_churn_y(d.churn_rate)

                    return `translate(${x_pos},${y_pos})`
                })

            var arcs = each_day.selectAll('arc')
                .data(d => {
                    var da = pie(d.status)
                    var step = d.step
                    var date = d.date
                    var num = d.num
                    // console.log(da)
                    da.forEach(data => {
                        data.step = step
                        data.date = date
                        data.num = num
                    })
                    return da
                })
                .enter()
                .append('g')


            arcs.append('path')
                .attr('fill', (d, i) => pie_color[i])
                .attr('d', arc)
                .attr('stroke', (d, i) => {
                    if (d.value == 0)
                        return
                    else
                        return pie_stroke_color[i]
                })
                .attr('stroke-width', 1)
                .attr('id', d => {
                    return 'step' + d.step + 'day' + d.date
                })
                .attr('class', d => {
                    return 'pie' + d.date
                })

            arcs.append('path')
                .attr('fill', '#666a53')
                .attr('d', num_arc)
                .attr('fill-opacity', d => {
                    return scale_nums(d.num)
                })
                .attr('stroke', (d, i) => {
                    if (d.value == 0)
                        return
                    else
                        return '#666a53'
                })
                .attr('stroke-opacity', d => {
                    return scale_nums(d.num)
                })
                .attr('stroke-width', 1)
                .attr('class', d => {
                    return 'pienum' + d.date
                })

            // draw interaction circle
            arcs.append('circle')
                .attr('r', pie_outer_r)
                .attr('fill', 'steelblue')
                .attr('fill-opacity', 0)
                .attr('class', d => {
                    return 'interpie' + d.date
                })
                .on("mouseover", pie_overed)
                .on("mouseout", pie_outed)

            // #endregion

            //#endregion

            //#region plot dashed lines
            var dashed_line = svg.append('g')
                .attr("transform", d => `translate(${margin.left}, ${margin.top})`)
            for (var i = 0; i < chnrate_data.length; i++) {
                if (i != 0) {
                    dashed_line.append('line')
                        .style("stroke-dasharray", ("3, 3"))
                        .attr('x1', i * step_width)
                        .attr('x2', i * step_width)
                        .attr('y1', 0)
                        .attr('y2', step_height)
                        .attr('stroke', '#222014')
                        .attr('stroke-width', 2)
                        .attr('stroke-opacity', 0.5)
                }
                dashed_line.append('text')
                    .style("font-size", "14px")
                    .attr("text-anchor", "middle")
                    .attr('x', step_width / 2.1 + i * step_width)
                    .attr('y', 0)
                    .text('step' + i)

                dashed_line.append('g')
                    .append('rect')
                    .attr('x', step_width / 1.9 + i * step_width)
                    .attr('y', -13)
                    .attr('width', 50)
                    .attr('height', 15)
                    .attr('stroke', '#808080')
                    .attr('fill', 'steelblue')
                    .attr('fill-opacity', 0.2)
                    .attr('rx', 3)
                    .attr('class', 'btn' + chnrate_data[i].step)
                    .on('click', button_clicked)
            }

            svg.append('g')
                .call(yAxis)
            //#endregion
        },
        plot_obse_legend() {
            var svg = d3.select('#obseLegend')

            var data = [
                { name: 'high', color: '#b2b4a9', stroke: '#666A53' },
                { name: 'med', color: '#e8be8d', stroke: '#D27D1C' },
                { name: 'low', color: '#aacada', stroke: '#72A6C1' },
                { name: 'churn', color: '#d38e83', stroke: '#B64330' },
            ]

            var step_data = [
                { name: 'step0', color: '#cfb9d6', stroke: '#5e1675' },
                { name: 'step1', color: '#fac6d1', stroke: '#ee4266' },
                { name: 'step2', color: '#fff2c5', stroke: '#ffd23f' },
                { name: 'step3', color: '#c2d5cd', stroke: '#337357' },
            ]

            var num_arc = d3.arc()
                .innerRadius(10)
                .outerRadius(12)

            var each_status = svg.append('g')
                .selectAll('g')
                .data(data)
                .enter()
                .append('g')
                .attr("transform", (d, i) => `translate(${23 + i * 80},${20})`)

            var arc = svg.append('g')
                .attr("transform", "translate(" + 358 + ", " + 20 + ")")

            arc
                .append('path')
                .attr('d', d => {
                    var endangle = 0;
                    var startangle = 2 * Math.PI;
                    return num_arc({ startAngle: startangle, endAngle: endangle })
                })
                .attr('fill', '#666a53')
                .attr('opacity', 0.5)

            arc.append('text')
                .attr('x', (d, i) => 20)
                .attr('y', 5)
                .attr('text-anchor', 'left')
                .text('player num')

            var step = svg.append('g')
                .selectAll('g')
                .data(step_data)
                .enter()
                .append('g')
                .attr("transform", (d, i) => `translate(${480 + i * 85},${10})`)

            step
                .append('rect')
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', 20)
                .attr('height', 20)
                .attr('fill', d => d.color)
                .attr('stroke', d => d.stroke)
                .attr('stroke-width', 1)

            step
                .append('text')
                .attr('x', (d, i) => 30)
                .attr('y', 15)
                .attr('text-anchor', 'left')
                .text(d => d.name)
        },
        read_cluster_csv() {
            return new Promise((resolve, reject) => {
                d3.csv("static/cluster.csv", function (err, data) {
                    resolve(data);
                    return data
                })
            })
        },
        async deal_with_cluster() {
            let cluster_data = await this.read_cluster_csv()
            var clu = d3.nest()
                .key(d => d.id)
                .entries(cluster_data)
            clu.forEach(ele => {
                ele.values.forEach(val => {
                    val.portrait = [
                        { key: 'school', value: val.school },
                        { key: 'grade', value: val.grade },
                        { key: 'bindcash', value: val.bindcash },
                        { key: 'deltatime', value: val.deltatime },
                        { key: 'combatscore', value: val.combatscore },
                        { key: 'sex', value: val.sex }
                    ]
                })
            })
            this.cluster = clu
        },
        link() {
            console.log('link_info', this.copy_link_info)
            this.init_showflag()
            this.$bus.$on("template to group", (msg) => {
                // save msg to show_flag, including idx and show flag
                var id = msg[0], flag = msg[1]
                this.show_flag[id] = flag

                // save rec to rec_show_flag
                this.rec_show_flag[id] = flag

                // initialize true_num
                this.true_num = 0

                // count true numbers
                for (var key in this.rec_show_flag) {
                    if (this.rec_show_flag[key] == true)
                        this.true_num += 1;
                }

                // wait some time
                if (flag == false) {
                    setTimeout(() => this.plotlines(), 170)
                }
                // else
                setTimeout(() => this.plotlines(), 50)

            })
            if (this.pos_flag == false) {
                this.get_linepos()
                this.pos_flag = true
            }

            this.plotlines()
        },
        formatTooltip(val) {
            let _this = this

            // GET PERCENTAGE, RETURN STRING
            if (val != null) {
                var slider1_per = _this.$refs.slider.$refs.button1.wrapperStyle.left,
                    slider2_per = _this.$refs.slider.$refs.button2.wrapperStyle.left

                // TURN STRING TO NUMBER
                slider1_per = slider1_per.slice(0, slider1_per.length - 1) * 1
                slider2_per = slider2_per.slice(0, slider2_per.length - 1) * 1

                _this.link_per_range = [slider1_per, slider2_per]
            }
            return val + '%';
        },
        async filter_link() {
            console.log('FILTER LINK')
            this.filter_flag = true

            // TEST VERSION, PLEASE COMMENT IT IN BUILD
            // this.copy_link()

            var per_min = this.link_per_range[0], per_max = this.link_per_range[1]
            var scale_per = d3.scaleLinear()
                .domain([0, 100])
                .range(this.glo_link_num_range)

            var num_min = scale_per(per_min), num_max = scale_per(per_max)

            this.copy_link_info = []
            this.link_info.forEach(day => {
                var save_day = []
                day.forEach((clu, i) => {
                    var save_clu = {}
                    var save_link = []
                    var link = clu.link
                    link.forEach((line, j) => {
                        if (line.value >= num_min && line.value <= num_max) {
                            var save_line = {}
                            save_line['tar'] = line.tar
                            save_line['value'] = line.value
                            save_link.push(save_line)
                        }
                    })
                    // if no lines, do not save
                    if (save_link.length != 0) {
                        save_clu['clu'] = clu.clu
                        save_clu['link'] = save_link
                        save_day.push(save_clu)
                    }
                })
                this.copy_link_info.push(save_day)
            })
            // console.log('link info', this.link_info)

            console.log('FILTER FINISH')
            this.filter_finish_flag = true
        },
        deepCopy(data) {
            if (typeof data !== 'object' || data === null) {
                throw new TypeError('传入参数不是对象')
            }
            let newData = {};
            const dataKeys = Object.keys(data);
            dataKeys.forEach(value => {
                const currentDataValue = data[value];
                if (typeof currentDataValue !== "object" || currentDataValue === null) {
                    newData[value] = currentDataValue;
                } else if (Array.isArray(currentDataValue)) {
                    newData[value] = [...currentDataValue];
                } else if (currentDataValue instanceof Set) {
                    newData[value] = new Set([...currentDataValue]);
                } else if (currentDataValue instanceof Map) {
                    newData[value] = new Map([...currentDataValue]);
                } else {
                    newData[value] = this.deepCopy(currentDataValue);
                }
            });
            return newData;
        },
        copy_link() {
            // copy link info
            this.copy_link_info = this.deepCopy(this.link_info)
            var arr_copy = []
            for (var key in this.copy_link_info) {
                arr_copy.push(this.copy_link_info[key])
            }
            this.copy_link_info = arr_copy

            //update flag
            this.filter_flag = false
        },
        cf_module_change(value){
            this.selected_module = value
            console.log(this.selected_module*1)
        },
        async wait_data() {
            // receive group data
            this.updated_flag = false

            this.$bus.$on("timeline to group", async (msg) => {
                // LOADING
                const rLoading = this.openLoading()

                // GET TIME RANGE
                this.time_range = msg

                // GET CF INITIALIZATION RANGE
                var cf_init_range = {}
                // only choose 1 day
                if (this.time_range[0] == this.time_range[1])
                    cf_init_range[this.time_range[0]] = {}
                // choose multiple days
                else {
                    var deltaday = this.time_range[1] - this.time_range[0] + 1
                    for (var i = this.time_range[0]; i < this.time_range[0] + deltaday; i++) {
                        cf_init_range[i] = {}
                    }
                }

                this.cf_range = cf_init_range
                this.table_range = this.deepCopy(cf_init_range)


                // filter users, --> table api
                var table_data = await HttpHelper.axiosPost('/table',
                    this.table_range, 600000)
                console.log('GROUP >> TIMELINE -> GROUP >> GROUP FILTER FINISH.')

                // group api
                var data = await HttpHelper.axiosPost('/group', this.time_range)
                console.log('GROUP >> TIMELINE -> GROUP >> GROUP FINISH.')

                // link api
                // if only choose one day, no link
                if (this.time_range[0] == this.time_range[1]) {
                    // do nothing
                }
                // if choose multiple days, send back to backend
                else {
                    var link_time_range = [this.time_range[0], this.time_range[1] - 1]
                    var link_data = await HttpHelper.axiosPost('/link', link_time_range)
                    this.link_info = link_data
                    this.copy_link()
                    console.log('GROUP >> TIMELINE -> GROUP >> LINK FINISH.')
                }

                // cf raw api
                var cf_raw_data = await HttpHelper.axiosPost('/raw',
                    { 'split_num': 5 }, 600000)
                console.log('GROUP >> TIMELINE -> GROUP >> CF RAW FINISH.')


                cf_raw_data = d3.nest()
                    .key(d => d.id)
                    .entries(cf_raw_data)

                data.forEach((da, i) => {
                    var clu_group = da.id
                    // add cf data
                    da.cf_raw = cf_raw_data[i].values
                    // da.cf = cf_data[i].values

                    // add local portrait data
                    da.values.forEach((val, j) => {
                        val.portrait = this.clu_data[clu_group][j].portrait
                    })
                })

                // console.log('GROUP >> TIMELINE -> GROUP >> data', data)
                this.data = data

                // loading close
                rLoading.close()

                this.updated_flag = true

            })

            // receive range from templates
            this.$bus.$on("template range to group", async (msg) => {
                // LOADING
                const rLoading = this.openLoading()

                console.log('template range to group', msg)
                var day = msg// send to save

                // SEND LAST SAVE DAY & CHANGE TO TIMELINE_SAVE
                this.$bus.$emit('send last save day', day)

                // DELETE NEXT DAY
                delete this.$glo_cf_range[this.time_range[1] + 1]
                delete this.$glo_table_range[this.time_range[1] + 1]

                // DEAL WITH STATUS IN GLO_TABLE
                for (var day in this.$glo_table_range) {
                    if (typeof (this.$glo_table_range[day].pred) != 'undefined') {
                        // pred
                        var pred_arr = this.$glo_table_range[day].pred
                        var min = d3.min(pred_arr)
                        var max = d3.max(pred_arr)
                        this.$glo_table_range[day].pred = [min, max]
                    }
                    if (typeof (this.$glo_table_range[day].class) != 'undefined') {
                        // class
                        var class_arr = this.$glo_table_range[day].class
                        var min = d3.min(class_arr)
                        var max = d3.max(class_arr)
                        this.$glo_table_range[day].class = [min, max]
                    }
                }
                console.log('GROUP >> TEMPLATE RANGE -> GROUP >> glo cf', this.$glo_cf_range)
                console.log('GROUP >> TEMPLATE RANGE -> GROUP >> glo table', this.$glo_table_range)

                // filter data first -- table api
                // send table data
                var table_data = await HttpHelper.axiosPost('/table',
                    this.$glo_table_range, 600000)
                this.$bus.$emit("template to table", table_data);

                // send boxplot data
                var boxplot_data = await HttpHelper.axiosPost('/boxplot',
                    this.$glo_table_range, 600000)
                this.$bus.$emit("template to boxplot", boxplot_data);
                console.log('GROUP >> TEMPLATE -> GROUP >> TABLE UPDATED.')

                // update group data and link data
                var updated_data = await HttpHelper.axiosPost('/group', this.time_range)
                console.log('GROUP >> TEMPLATE -> GROUP >> GROUP UPDATED.')

                // if only choose one day, no link
                if (this.time_range[0] == this.time_range[1]) {
                    // do nothing
                }
                // if choose multiple days, send back to backend
                else {
                    var link_time_range = [this.time_range[0], this.time_range[1] - 1]
                    var link_data = await HttpHelper.axiosPost('/link', link_time_range)
                    this.link_info = link_data
                    this.copy_link() //copy link info
                    console.log('GROUP >> TEMPLATE -> GROUP >> LINK UPDATED.')
                }

                console.log('cf module', this.selected_module)
                // update cf data
                var cf_data = await HttpHelper.axiosPost('/cf',
                    { 'setting': this.$glo_cf_range, 'split_num': 5 , 'target': this.selected_module}, 600000)
                console.log('GROUP >> TEMPLATE -> GROUP >> CF UPDATED.')

                cf_data = d3.nest()
                    .key(d => d.id)
                    .entries(cf_data)

                updated_data.forEach((da, i) => {
                    var clu_group = da.id
                    // add cf data
                    da.cf = cf_data[i].values

                    // add local portrait data
                    da.values.forEach((val, j) => {
                        val.portrait = this.clu_data[clu_group][j].portrait
                    })
                })

                this.data = updated_data
                this.updated_flag = true

                // loading close
                rLoading.close()
            });

            // receive save ob_sw from timeline
            this.$bus.$on("save to group", async (msg) => {
                var ob_se = msg[0]
                this.obse_box_data = ob_se.boxplot
                this.obse_chn_data = ob_se.churnrate
                this.plot_ob_sequence()
            });
        },
    },
    created() {
    },
};