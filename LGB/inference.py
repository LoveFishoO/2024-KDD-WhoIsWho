import os
import json
import argparse

import pickle as pk
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help="the name of embedding model", default = "e5_instruct")
parser.add_argument('--test_path', type=str, help="the path of test feature file", default = "../out_data/e5_instruct_lgb_test.csv")
parser.add_argument('--test_author_path', type=str, help="the path of test file", default = "../IND-WhoIsWho/ind_test_author_submit.json")
parser.add_argument('--result_path', type=str, help="the path of result file", default = "../output/e5_instruct_lgb.json")
parser.add_argument('--model_dir', type=str, help="the dir of lgb model", default = "./lgb_model/")

args = parser.parse_args() 

test_data = pd.read_csv(args.test_path)

model_dir = args.model_dir + args.model + '/'
file_names = os.listdir(model_dir)

models = []
for file_name in file_names:
    with open(model_dir + file_name, 'rb') as file:
        model = pk.load(file)
        models.append(model)



choose_cols=[col for col in test_data.columns if col not in ['id', 'name','text_id' , 'title', 'abstract', 'venue','title_len', 'title_split_len', 'paper_num'] + 
             ['year_mean', 'year_std', 'year_skew'] 
             + ['coauthor_paper_count' + 'max_coauthor_max', 'keywords_len_max'] 
             + ['max_venue_sim_abs_year', 'max_title_sim_abs_year', 'max_abstract_sim_abs_year']
             + [
                 'near_max_cokeyword_cnt', 'near_mean_cokeyword_cnt', 'near_cokeyword_cnt_over_this_keyword', 'near_cokeyword_cnt_over_keyword2',
                'max_cnt_aut_over_paper_author', 'max_cnt_org_over_paper_orgs',

                    # 'before_match_org_cnt', 'is_match_org',
                    # 'before_match_venue_cnt',

                    # 'all_match_org_cnt', 'all_match_venue_cnt',

                    # 'max_venue_sim_abs_year', 'max_title_sim_abs_year', 'max_abstract_sim_abs_year',
                    'year_gap_mean', 'year_gap_median',
                    'year_gap_min_max_ave',
                    
                    # 'before_match_org_cnt_over_all_cnt','before_match_venue_cnt_over_all_cnt'
                    'max_title_dist', 'mean_title_dist', 'std_title_dist', 'skew_title_dist', 'max_title_sim_add_dist'
                    ]
            #  + [ 'coaut_paper_count_over_paper_num']
            # + ['max_cnt_aut_over_paper_author', 'max_cnt_coaut_jaccard_sim', 'max_cnt_org_over_paper_orgs', 'max_cnt_org_jaccard_sim',
            #         'max_aut_jaccard_sim', 'mean_aut_jaccard_sim',]
             
            #  ['max_coauthor_skew', 'keywords_len_skew']
            #  ['near_max_coorg_cnt', 'near_mean_coorg_cnt']
             ]
        
test_pred_pro=np.zeros((len(models),len(test_data),2))
for i, model in enumerate(models):
    test_pred_pro[i] = model.predict_proba(test_data[choose_cols])
    
test_preds=test_pred_pro.mean(axis=0)[:,1]


with open(args.test_author_path) as f:
    submission=json.load(f)

cnt=0
for id,names in submission.items():
    for name in names:
        submission[id][name]=test_preds[cnt]
        cnt+=1
with open(args.result_path, 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii=False, indent=4)
    
print(args.result_path)



