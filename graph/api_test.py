from train_model import api_link, api_group, api_timeline, api_shap, save_shap, api_boxplot, \
    push_to_DB, api_individual, api_table, add_class, api_init_current_mongo, \
    api_link_mongo, add_shap, update_shap, api_cf, api_comparison, inference, api_timeline_mongo, \
    api_group_mongo, api_savestep, api_getstep, api_next_mongo, api_raw, train, highest_shap, sp_train

if __name__ == '__main__':
    # api_group([0,2])
    # api_group_mongo([0,2])
    # api_timeline()
    # api_timeline_mongo()
    # api_link([0,3])
    # api_link_mongo([0,1])
    # save_shap()
    # api_boxplot()
    # push_to_DB('test_pred.csv', 'shap_high')
    # api_individual({'uid': '2841796236690903101', 'day':0})
    # api_table({0: {'sex': [1, 1]}, 1: {'grade': [42, 43]}})
    # add_class_mongo()
    # add_class()
    # init_current_mongo()
    # add_shap()
    # update_shap()
    # api_table({0: {'sex': [1, 1]}, 1: {'grade': [42, 43]}, 2: {'grade': [43, 44]}})
    # api_shap(2)
    api_cf({'setting': {0: {'change': ['deltatime']}}, 'split_num': 5, 'target': 1})
    # api_comparison()
    # api_savestep(0)
    # api_getstep(0)
    # api_next_mongo()
    # api_raw({'split_num': 5})

    '''train'''
    # train(is_norm=0, loss_name='focal')
    '''check'''
    # highest_shap()
    # sp_train(is_norm=1)

    '''final'''
    # inference(is_norm=1, ckpt_name='norm_pretrain_epo_80_0.1494.ckpt')
    '''only focal'''
    # inference(is_norm=0, ckpt_name='focal_pretrain_epo_60_0.2076.ckpt')
    '''only autoint'''
    # inference(is_norm=0, ckpt_name='pretrain_epo_20_1.2975.ckpt')
    '''only norm'''
    # inference(is_norm=1, ckpt_name='norm__pretrain_epo_40_0.9971.ckpt')