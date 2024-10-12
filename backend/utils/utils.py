#-*- coding: utf-8 -*-
import re
from datetime import datetime, timedelta
import os

abspath = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

def get_config_param():
    config_path = '../config/config.txt'
    abspath = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), config_path))
    with open(abspath,'r') as f:
        param_str = f.readlines()
        delta_time = int(param_str[0].split()[1])
        k = int(param_str[1].split()[1])
        delta_minute = int(param_str[2].split()[1])
    return delta_time, k, delta_minute

delta_time, k, delta_minute = get_config_param()

def date_cmp(dat1, dat2):
    t1 = datetime.strptime(dat1, '%Y-%m-%d %H:%M:%S')+timedelta(hours=delta_time)
    t2 = datetime.strptime(dat2, '%Y-%m-%d %H:%M:%S')
    return t1<=t2

def read_raw_data(f, st, ed, cnt):
    '''
    input
    f: file reader
    st: start line
    ed: end line
    cnt: current line
    output
    cnt: line tag
    all_data: [cur_time, gbid, tgbid, evt_type, intimacy]
    '''
    all_data = []
    rule = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\t(\d{5})\t(\d{20})\t(\d+)\t(\d{19})' \
           r'\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t(\d{8})'
    while cnt < st:
        f.readline()
        cnt += 1
    while(True):
        line = f.readline()
        if line == '' or cnt >= ed:
            break
        line_match = re.match(rule, line)
        if line_match == None:
            continue
        line_split = line_match.groups()
        cur_time = line_split[0]
        gbid = line_split[2][1:]
        tgbid = line_split[4]
        evt_type = int(line_split[5])
        intimacy = int(line_split[7])
        line_split = [cur_time, gbid, tgbid, evt_type, intimacy]
        all_data.append(line_split)
        cnt += 1
    return cnt, all_data

def minute_cmp(dat1, dat2):
    t1 = datetime.strptime(dat1, '%Y-%m-%d %H:%M:%S')+timedelta(minutes=delta_minute)
    t2 = datetime.strptime(dat2, '%Y-%m-%d %H:%M:%S')
    return t1<=t2

def read_raw_data_span(f, st_date=None):
    '''
    input
    f: file reader
    st_date: starting date. If None, the first read date is st_date.
    output
    last_date: last read date
    all_data: [cur_time, gbid, tgbid, evt_type, intimacy]
    '''
    last_date = None
    all_data = []
    rule = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\t(\d{5})\t(\d{20})\t(\d+)\t(\d{19})' \
           r'\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t(\d{8})'
    while(True):
        last_p = f.tell()
        line = f.readline()
        if line == '':
            break
        line_match = re.match(rule, line)
        if line_match == None:
            continue
        line_split = line_match.groups()
        cur_time = line_split[0]
        gbid = line_split[2][1:]
        tgbid = line_split[4]
        evt_type = int(line_split[5])
        intimacy = int(line_split[7])
        line_split = [cur_time, gbid, tgbid, evt_type, intimacy]
        if st_date == None:
            st_date = cur_time
            continue
        if minute_cmp(st_date, cur_time):
            f.seek(last_p)
            break
        last_date = cur_time
        all_data.append(line_split)

    assert last_date != None
    return last_date, all_data

def read_raw_login_span(f, st_date=None):
    '''
    input
    f: file reader
    st_date: starting date. If None, the first read date is st_date.
    output
    last_date: last read date
    all_data: [cur_time, gbid, tgbid, evt_type, intimacy]
    '''
    last_date = None
    all_data = []
    rule = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(.*)(\d{20})(.*)(\d+)\t(\d+)\t(\d+\.\d+\.\d+\.\d+)(.*)' \
           r'(\d+)\t(\d+)\t(\d+)\t(\d+)\t(\[(.*), (.*), (.*), (.*), (.*)\])\t(.*)\t(\d+)\t(.*)\t(.*)\t(\d+)'
    while(True):
        line = f.readline()
        if line == '':
            break
        line_match = re.match(rule, line)
        if line_match == None:
            continue
        line_split = line_match.groups()
        # line_split = line.split('\t')
        # if line_split[3] == '02841794259865042945':
        #     continue
        # line_split = list(filter(len, line_split))
        cur_time = line_split[0]
        gbid = line_split[2][1:]
        grade = line_split[4]
        experience = line_split[5]
        cash = line_split[8]
        # radarchart = line_split[13:18]
        combatscore = line_split[19]
        if gbid == '2841794259865042945':
            continue
        line_split = [cur_time, gbid, grade, experience, cash, combatscore]
        all_data.append(line_split)
        if st_date == None:
            st_date = cur_time
            continue
        if date_cmp(st_date, cur_time):
            last_date = cur_time
            break
    assert last_date != None
    return last_date, all_data

def read_raw_logout_span(f, st_date=None):
    '''
    input
    f: file reader
    st_date: starting date. If None, the first read date is st_date.
    output
    last_date: last read date
    all_data: [cur_time, gbid, tgbid, evt_type, intimacy]
    '''
    last_date = None
    all_data = []
    rule = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(.*)(\d{20})(.*)(\d+)\t(\d+)\t(\d+\.\d+\.\d+\.\d+)(.*)' \
           r'((\t+(\d+)){14}\t)(.*)(\d+)\t(\{.*\})\t(\d+)'
    while(True):
        line = f.readline()
        if line == '':
            break
        line_match = re.match(rule, line)
        if line_match == None:
            continue
        line_split = line_match.groups()
        # line_split = line.split('\t')
        # if line_split[3] == '02841794259865042945':
        #     continue
        # line_split = list(filter(len, line_split))
        cur_time = line_split[0]
        gbid = line_split[2][1:]
        grade = line_split[4]
        experience = line_split[5]
        cash = line_split[8].split('\t')[5]
        combatscore = line_split[10]
        if gbid == '2841794259865042945':
            continue
        line_split = [cur_time, gbid, grade, experience, cash, combatscore]
        all_data.append(line_split)
        if st_date == None:
            st_date = cur_time
            continue
        if date_cmp(st_date, cur_time):
            last_date = cur_time
            break
    assert last_date != None
    return last_date, all_data

def get_line_tag_path():
    return os.path.join(abspath, 'line_tag/line_tag.txt')

def get_pg_path():
    return os.path.join(abspath, 'raw_data/pg_add_intimacy.txt')

def get_login_path():
    return os.path.join(abspath, 'raw_data/pg_logon_day.txt')

def get_logout_path():
    return os.path.join(abspath, 'raw_data/pg_logout_day.txt')

def get_graph_dir():
    return os.path.join(abspath, 'model_data/undirected_graph')

def get_dist_dir():
    return os.path.join(abspath, 'model_data/dist_graph')

def get_between_dir():
    return os.path.join(abspath, 'model_data/between_graph')

def get_embedding_model_dir():
    return os.path.join(abspath, 'model_data/embedding_model')

def get_embedding_weight_dir():
    return os.path.join(abspath, 'model_data/embedding_weight')

def get_dynamic_graph_dir():
    return os.path.join(abspath, 'model_data/dynamic_graph')

def get_all_nodes_path():
    return os.path.join(abspath, 'model_data/all_nodes.json')

def get_aernn_result_dir():
    return os.path.join(abspath, 'model_data/aernn_result/')

def get_struc2vec_dir():
    return os.path.join(abspath, 'model_data/struc2vec_emb')

def get_alignment_dir():
    return os.path.join(abspath, 'model_data/alignment/')

def get_decoder_dir():
    return os.path.join(abspath, 'model_data/decoder/')

def get_tsne_of_struc2vec_dir():
    return os.path.join(abspath, 'model_data/tsne_struc2vec')

def get_tsne_of_dyn_dir():
    return os.path.join(abspath, 'model_data/tsne_dynAERNN')

def get_reproduction_dir():
    return os.path.join(abspath, 'model_data/reproduction')

def get_reproduction_alignment_dir():
    return os.path.join(abspath, 'model_data/reproduction_alignment')

def get_reproduction_decoder_dir():
    return os.path.join(abspath, 'model_data/reproduction_decoder')

def get_reproduction_tsne_dir():
    return os.path.join(abspath, 'model_data/reproduction_tsne')

def get_event_list_dir():
    return os.path.join(abspath, 'raw_data/event_list')

def get_full_event_list_dir():
    return os.path.join(abspath, 'raw_data/full_event_list')

def get_event_model_dir():
    return os.path.join(abspath,'model_data/event_model_dyn')

def get_log_dir():
    return os.path.join(abspath, 'log')

def get_metrics_dynAERNN_dir():
    return os.path.join(abspath, 'model_data/metrics_dynAERNN')

def get_metrics_reproduction_dir():
    return os.path.join(abspath, 'model_data/metrics_reproduction')

def get_metrics_struc2vec_dir():
    return os.path.join(abspath, 'model_data/metrics_struc2vec')

def get_event_emb_dyn_dir():
    return os.path.join(abspath, 'model_data/event_emb_dyn')

def get_tsne_event_dir():
    return os.path.join(abspath, 'model_data/tsne_event')

def get_doc2vec_dir():
    return os.path.join(abspath, 'raw_data/doc2vec')

def get_churn_d1():
    return os.path.join(abspath, 'churn_data/d1')

def get_churn_pretrain():
    return os.path.join(abspath, 'churn_data/pretrain')