# -*- encoding: utf-8 -*-

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import csv
import json
import logging

from train_model import counter_factual_example, api_timeline, api_timeline_mongo, api_group, api_group_mongo, api_link_mongo, api_boxplot_mongo, \
                        api_individual, api_table, api_shap, api_cf, api_comparison, api_init_current_mongo, api_savestep, api_getstep, api_raw, api_next_mongo

app = Flask(__name__)
cors = CORS(app, resources={r"/cf/*": {"origins": "*"}}, supports_credential=True)


@app.route('/cf/timeline', methods=['GET', 'POST'])
def getTimeline():
    data = api_timeline()
    response = {
        'code': 0,
        'data': data
    }
    return jsonify(response)


@app.route('/cf/next', methods=['GET', 'POST'])
def getNextGroup():
    data = api_next_mongo()
    response = {
        'code': 0,
        'data': data
    }
    return jsonify(response)


@app.route('/cf/group', methods=['GET', 'POST'])
def getGroup():
    if request.method == 'GET':
        req = request.args
        req = json.loads(req.get('data'))
    elif request.method == 'POST':
        req = request.get_json()
    data = api_group_mongo(req)
    response = {
        'code': 0,
        'data': data
    }
    return jsonify(response)


@app.route('/cf/link', methods=['GET', 'POST'])
def getLink():
    if request.method == 'GET':
        req = request.args
        req = json.loads(req.get('data'))
    elif request.method == 'POST':
        req = request.get_json()
    data = api_link_mongo(req)
    response = {
        'code': 0,
        'data': data
    }
    return jsonify(response)


@app.route('/cf/boxplot', methods=['GET', 'POST'])
def getBoxPlot():
    data = api_boxplot_mongo()
    response = {
        'code': 0,
        'data': data
    }
    return jsonify(response)


@app.route('/cf/individual', methods=['GET', 'POST'])
def getIndividual():
    if request.method == 'GET':
        req = request.args
        req = json.loads(req.get('data'))
    elif request.method == 'POST':
        req = request.get_json()
    data = api_individual(req)
    response = {
        'code': 0,
        'data': data
    }
    return jsonify(response)


@app.route('/cf/table', methods=['GET', 'POST'])
def getTable():
    if request.method == 'GET':
        req = request.args
        req = json.loads(req.get('data'))
    elif request.method == 'POST':
        req = request.get_json()
    data = api_table(req)
    response = {
        'code': 0,
        'data': data
    }
    return jsonify(response)


@app.route('/cf/shap', methods=['GET', 'POST'])
def getShap():
    req = request.get_json()
    data = api_shap(req)
    response = {
        'code': 0,
        'data': data
    }
    return jsonify(response)


@app.route('/cf/cf', methods=['GET', 'POST'])
def getCF():
    if request.method == 'GET':
        req = request.args
        req = json.loads(req.get('data'))
    elif request.method == 'POST':
        req = request.get_json()
    data = api_cf(req)
    response = {
        'code': 0,
        'data': data
    }
    return jsonify(response)


@app.route('/cf/raw', methods=['GET', 'POST'])
def getRaw():
    if request.method == 'GET':
        req = request.args
        req = json.loads(req.get('data'))
    elif request.method == 'POST':
        req = request.get_json()
    data = api_raw(req)
    response = {
        'code': 0,
        'data': data
    }
    return jsonify(response)


@app.route('/cf/comparison', methods=['GET', 'POST'])
def getComparison():
    data = api_comparison()
    response = {
        'code': 0,
        'data': data
    }
    return jsonify(response)


@app.route('/cf/init', methods=['GET', 'POST'])
def getInit():
    api_init_current_mongo()
    response = {
        'code': 0,
        'data': ''
    }
    return jsonify(response)


@app.route('/cf/saveStep', methods=['GET', 'POST'])
def getSaveStep():
    if request.method == 'GET':
        req = request.args
        req = json.loads(req.get('data'))
    elif request.method == 'POST':
        req = request.get_json()
    data = api_savestep(req)
    response = {
        'code': 0,
        'data': data
    }
    return jsonify(response)

@app.route('/cf/getStep', methods=['GET', 'POST'])
def getGetStep():
    if request.method == 'GET':
        req = request.args
        req = json.loads(req.get('data'))
    elif request.method == 'POST':
        req = request.get_json()
    data = api_getstep(req)
    response = {
        'code': 0,
        'data': data
    }
    return jsonify(response)

# @app.route('/euro/getLasso', methods=['GET', 'POST'])
# def getLasso():
#     req = request.get_json()
#     data = root_getLasso(req)
#     response = {
#         'code': 0,
#         'data': data
#     }
#     return jsonify(response)
#
#
# @app.route('/euro/getInference', methods=['GET', 'POST'])
# def getInference():
#     data = root_inference()
#     data = json.loads(json.dumps(data).replace('NaN', '\"NaN\"'))
#     response = {
#         'code': 0,
#         'data': data
#     }
#     return jsonify(response)


if __name__ == '__main__':
    app.debug = True
    app.run()
