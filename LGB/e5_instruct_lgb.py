
import re
import gc
import swifter
import optuna
import math
import random
import pandas as pd
import numpy as np
import json
import torch
import copy
import pickle as pl
import lightgbm as lgb
from tqdm import tqdm
import multiprocessing as mp
from jellyfish import jaro_winkler_similarity
from lightgbm import LGBMClassifier,log_evaluation,early_stopping
from unidecode import unidecode
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA, TruncatedSVD
from collections import Counter
from sklearn.model_selection import StratifiedKFold

device = torch.device("cuda:0")


with open('../out_data/e5_instruct_title_data.pkl', 'rb') as f:
    TITLE_EMBEDDINGS = pl.load(f)

with open('../out_data/e5_instruct_abstract_data.pkl', 'rb') as f:
    ABSTRACT_EMBEDDINGS = pl.load(f)

with open('../out_data/e5_instruct_venue_data.pkl', 'rb') as f:
    VENUE_EMBEDDINGS = pl.load(f)

with open('../out_data/bge_orgs_data.pkl', 'rb') as f:
    ORGS_EMBEDDINGS = pl.load(f)
    
def cosine_similarity(v1, v2):
    if type(v1) == list:
        v1 = np.array(v1)

    if type(v2) == list:
        v2 = np.array(v2)
    
    return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

with open("../IND-WhoIsWho/pid_to_info_all.json") as f:
    pid_to_info=json.load(f)


class Config():
    seed=42
    num_folds=10
    TARGET_NAME ='label'


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
seed_everything(Config.seed)

def get_embedding_data(pid, key):
    if key == 'title':
        # return TITLE_EMBEDDINGS[pid][key]
        return TITLE_EMBEDDINGS[pid]
    elif key =='abstract':
        # return ABSTRACT_EMBEDDINGS[pid][key]
        return ABSTRACT_EMBEDDINGS[pid]
    elif key =='venue':
        # return VENUE_EMBEDDINGS[pid][key]
        return VENUE_EMBEDDINGS[pid]

def base_feat_engineer(feat):

    title_embedding = get_embedding_data(feat['id'], 'title')
    title_embedding = vec2list(title_embedding)


    abstract_embedding = get_embedding_data(feat['id'], 'abstract')
    abstract_embedding = vec2list(abstract_embedding)
    # abstract_embedding_features = [f'abstract_embed_{i}' for i in range(len(title_embedding))]
    
    venue_embedding = get_embedding_data(feat['id'], 'venue')
    venue_embedding = vec2list(venue_embedding)

    column_names = [
                    # 'title_len', 
                    # 'title_split_len', 
                    # 'abstract_len', 
                    'text_id',
                    'keywords_len', 
                    'authors_len'] + TITLE_EMBEDDING_FEATURES + ABSTRACT_EMBEDDING_FEATURES + VENUE_EMBEDDING_FEATURES

    org_feats, org_columns = org_feat(feat)
    feat_list = [
                # len(feat['title']),len(feat['title'].split()), 
                # len(feat['abstract']),
                feat['id'],
                len(feat['keywords']),
                len(feat['authors']),
                ]  + title_embedding + abstract_embedding + venue_embedding + org_feats
    try:
        feat_list = feat_list + [len(feat['venue']),int(feat['year'])]
    except:
        feat_list = feat_list + [0, 0]
    
    return feat_list, column_names + org_columns + ['venue_len', 'year']

def clean_name(name):
        # print(name)
        name = unidecode(name)
        name = name.lower()
        new_name = ""
        for a in name:
            if a.isalpha():
                new_name += a
            else:
                new_name = new_name.strip()
                new_name += " "
        return new_name.strip()

def simple_name_match(n1,n2):
    n1_set = set(n1.split())
    n2_set = set(n2.split())

    if(len(n1_set) != len(n2_set)):
        return False
    com_set = n1_set & n2_set
    if(len(com_set) == len(n1_set)):
        return True
    return False

def LCS(str_a, str_b):
    str_a=str_a.lower()
    str_b=str_b.lower()
    if len(str_a) == 0 or len(str_b) == 0:
        return 0
    dp = [0 for _ in range(len(str_b) + 1)]
    for i in range(1, len(str_a) + 1):
        left_up = 0
        dp[0] = 0
        for j in range(1, len(str_b) + 1):
            left = dp[j-1]
            up = dp[j]
            if str_a[i-1] == str_b[j-1]:
                dp[j] = left_up + 1
            else:
                dp[j] = max([left, up])
            left_up = up
    return dp[len(str_b)]

def LCS_Score(str_a, str_b):
    return np.round(LCS(str_a, str_b)*2/(len(str_a)+len(str_b)),2)


def str_jaccard_similarity(a, b):
    a = a.lower()
    b = b.lower()

    a = set(a.split())
    b = set(b.split())
    try:
        return len(a & b) / len(a | b)
    except ZeroDivisionError:
        return -1e-4

def list_jaccard_similarity(list1, list2):
    if not list1 or not list2:
        return 0
    intersection = len(set(list1) & set(list2))
    union = len(set(list1)) + len(set(list2)) - intersection
    return intersection / union if union != 0 else 0


Min_year = 20000
for i in pid_to_info:
    c = pid_to_info[i]
    if c['year'] == '' or c['year']==0:
        continue
    if c['year'] < Min_year:
        Min_year = c['year']


def vec2list(vec):
    # return vec.to('cpu').detach().numpy().reshape(-1).tolist()
    if type(vec) != list:
        return vec.reshape(-1).tolist()
    return vec


def org_feat(feat):
    orgs = []
    authors = feat['authors']
    for a in authors:
        orgs.append(a['org'].lower())
    
    od = Counter(orgs)
    try:
        od.pop('')
        od.pop(' ')
    except:
        pass
    
    cnt =0
    for i in od:
        if od[i]!=1:
            
            cnt += od[i]
    try:
        v = [len(od), len(od)/len(orgs), cnt, cnt/len(orgs)]
        
    except:
        v = [0,0,0,0]
    
    c = ['nunique_org', 'nunique_org_over_orgs_len', 'same_orgs_num', 'same_orgs_num_over_orgs_len']
    return v, c

TITLE_EMBEDDING_FEATURES = [f'title_embed_{i}' for i in range(1024)]
# TITLE_EMBEDDING_FEATURES = [f'title_embed_{i}' for i in range(4096)]

ABSTRACT_EMBEDDING_FEATURES = [f'abstract_embed_{i}' for i in range(1024)]
VENUE_EMBEDDING_FEATURES = [f'venue_embed_{i}' for i in range(1024)]
# VENUE_EMBEDDING_FEATURES = [f'venue_embed_{i}' for i in range(4096)]





def corss_feature_enginner(user, p_feat, paper_list):
        user = clean_name(user)

        max_cnt_aut = 0
        max_cnt_org = 0
        max_cnt_keyword = 0
        max_cnt_aut_org_cnt = 0
        
        max_cnt_venue_word = 0

        max_this_user_org_in_list_num = 0
        max_this_user_org_in_list_ratio = 0


        if p_feat['venue'] != '' and p_feat['venue'] != ' ' and p_feat['venue'] != None:
            this_venue_word_set = set(p_feat['venue'].lower().split())

        else:
            this_venue_word_set = set()


        coaut_papar_count = 0
        # same_venue_paper_count = 0

        if p_feat['year'] != '' and p_feat['year'] != ' ':
            this_year = p_feat['year']
        else:
            this_year = 0
        if p_feat['venue'] != '' and p_feat['venue'] != ' ' and p_feat['venue'] != None:
            this_venue = p_feat['venue'].lower()
        else:
            this_venue = ''
        
        authors1 = []
        orgs1 = set()
        keyword1 = []
        authors1_orgs1 = []
        this_user = ''
        this_user_org = ''
        
        for i in p_feat['authors']:
            # nl = i['name'].lower()
            nl = clean_name(i['name'])

            ol = i['org'].lower()
            authors1.append(nl)
            if i['org'] != '':
                orgs1.add(ol)
            
            authors1_orgs1.append([nl, ol])

            # if nl == user.lower():
            if simple_name_match(nl, user):
                this_user = nl
                this_user_org = i['org']

        keyword1 += [i.lower() for i in p_feat['keywords']]
        
        max_venue_dist = 0
        max_venue_sim = 0
        max_title_sim = 0
        max_abstract_sim = 0

        all_authors = set()
        all_keywords = set()
        all_orgs = set()

        max_orgs_year = 0
        max_aut_year = 0
        max_keyword_year = 0
        
        before_years = []
        after_years = []
        
        title_sims = []
        abstract_sims = []
        venue_sims = []
        org_sim_list = []
        
        
        org_dist_list = []
        venue_dist_list = []
        title_dist_list = []
        
        nearest_year_diff = 10000
        # near_coauthor_cnts = []
        
        near_coauthor_dict = {}
        near_coorgs_dict = {}
        
        near_cokeyword_dict = {}

        near_venue_sim = []
        near_title_sim = []
        near_abstract_sim = []
        near_org_sim = []

        near_org_dist = []

        max_cnt_cokeyword_keyword2 = set()
        max_cnt_aut_authors2 = set()
        max_cnt_coorgs_orgs2 = set()
        
        keyword2_list = []
        authors2_list = []
        orgs2_list =[]

        max_paper_venue_word_set = set()
        
        this_user_orgs_in_paper_list_num = 0
        this_user_in_paper_list_num = 0
        this_venue_in_paper_list_num = 0

        keyword_in_paper_list_num = 0

        is_match_venue = 0
        before_match_venue_cnt = 0
        all_match_venue_cnt = 0

        is_match_org = 0
        before_match_org_cnt = 0
        all_match_org_cnt = 0

        for paper in paper_list:
            feat = pid_to_info[paper]
            
            if feat['year'] != '' and feat['year'] != ' ':
                paper_year = int(feat['year'])
            else:
                paper_year = 0
            
            # 新增
                
            paper_venue = feat['venue']

            if  paper_venue != '' and paper_venue != ' ' and paper_venue != None and paper_venue.lower() == this_venue:
                
                if paper_year !=0 and paper_year < this_year:
                    is_match_venue = 1
                    before_match_venue_cnt += 1
                
                all_match_venue_cnt += 1

                this_venue_in_paper_list_num += 1

            # else:
            #     is_match_venue = 0

            if  paper_venue != '' and paper_venue != ' ' and paper_venue != None:
                paper_venue_word_set = set(paper_venue.lower().split())
            else:
                paper_venue_word_set = set()

            if len(list(paper_venue_word_set.intersection(this_venue_word_set))) > max_cnt_venue_word:
                max_cnt_venue_word = len(list(paper_venue_word_set.intersection(this_venue_word_set)))
                max_paper_venue_word_set = paper_venue_word_set

            aut_cnt=0
            org_cnt=0
            keyword_cnt = 0
            aut_org_cnt = 0

            # 过滤重复的orgs
            orgs2_record = []

            near_authors2 = set()
            near_orgs2 = set()
            near_keywords2 = set()
            
            tmp_keyword2 = set()
            tmp_authors = set()
            tmp_orgs = set()

            paper_org_sim_list = []
            paper_org_dist_list =[]

            this_user_orgs_in_list_num = 0
            
            for i in feat['authors']:
                # name_low = i['name'].lower()
                name_low = clean_name(i['name'])
                
                all_authors.add(name_low)
                tmp_authors.add(name_low)

                org_low = i['org'].lower()
                
                all_orgs.add(org_low)
                tmp_orgs.add(org_low)

                near_authors2.add(name_low)
                if org_low != '' and  org_low != ' ':
                    near_orgs2.add(org_low)

                    if org_low == this_user_org:
                        this_user_orgs_in_list_num += 1
                        this_user_orgs_in_paper_list_num += 1


                if  name_low in authors1:
                    aut_cnt += 1
                if  org_low in orgs1 and org_low not in orgs2_record:
                    org_cnt += 1
                    orgs2_record.append(org_low)

                if [name_low, org_low] in authors1_orgs1:
                    aut_org_cnt += 1
                    
                    # 新增
                    if paper_year !=0 and paper_year < this_year:
                        is_match_org = 1
                        before_match_org_cnt += 1
                    all_match_org_cnt += 1

                # if  name_low == this_user:
                if simple_name_match(name_low, this_user):
                    
                    this_user_in_paper_list_num += 1

                    try:
                        org_sim = cosine_similarity(ORGS_EMBEDDINGS[this_user_org], ORGS_EMBEDDINGS[i['org']])
                    except:
                        org_sim = 0
                    org_sim_list.append(org_sim)
                    paper_org_sim_list.append(org_sim)
                    
                    try:
                        org_dist = jaro_winkler_similarity(this_user_org, org_low)
                    except:
                        org_dist = 0
                    org_dist_list.append(org_dist)
                    paper_org_dist_list.append(org_dist)

            for i in feat['keywords']:
                kl = i.lower()
                all_keywords.add(kl)
                
                tmp_keyword2.add(kl)

                near_keywords2.add(kl)
                if  kl in keyword1:
                    keyword_cnt += 1
                    
                    keyword_in_paper_list_num += 1

            keyword2_list.append(list(tmp_keyword2))
            authors2_list.append(list(tmp_authors))
            orgs2_list.append(list(tmp_orgs))

            if aut_cnt>0:
                coaut_papar_count+=1     
            
            # max_cnt_aut = max(max_cnt_aut, aut_cnt)
            # max_cnt_org = max(max_cnt_org, org_cnt)
            
            if aut_cnt > max_cnt_aut:
                max_cnt_aut = aut_cnt
                max_aut_year = paper_year

                max_cnt_aut_authors2 = tmp_authors
            
            if org_cnt > max_cnt_org:
                max_cnt_org = org_cnt
                max_orgs_year = paper_year

                max_cnt_coorgs_orgs2 = tmp_orgs
                
            if keyword_cnt > max_cnt_keyword:
                max_cnt_keyword = keyword_cnt
                max_keyword_year = paper_year
                
                max_cnt_cokeyword_keyword2 = tmp_keyword2
            
            # max_cnt_keyword =max(max_cnt_keyword, keyword_cnt)
            max_cnt_aut_org_cnt = max(max_cnt_aut_org_cnt, aut_org_cnt)
            
            # 新增
            max_this_user_org_in_list_num = max(max_this_user_org_in_list_num, this_user_orgs_in_list_num)
            
            try:
                this_user_org_in_list_ratio = this_user_orgs_in_list_num / len(feat['authors'])
            except:
                this_user_org_in_list_ratio = 0
                
            max_this_user_org_in_list_ratio = max(max_this_user_org_in_list_ratio, this_user_org_in_list_ratio)
            
            if paper_year < this_year:
                before_years.append(paper_year)
            
            if paper_year > this_year:
                after_years.append(paper_year)
            
            try:
                venue_dist = jaro_winkler_similarity(p_feat['venue'], feat['venue'])
                
            except:
                venue_dist = 0
                
            venue_dist_list.append(venue_dist)

            try:
                venue_sim = cosine_similarity(VENUE_EMBEDDINGS[p_feat['id']], VENUE_EMBEDDINGS[feat['id']])
            except Exception as e:
                venue_sim = 0
            venue_sims.append(venue_sim)
            # max_venue_sim = max(max_venue_sim, venue_sim)
            # max_venue_dist = max(max_venue_dist, venue_dist)

            if venue_sim > max_venue_sim:
                max_venue_sim = venue_sim
                max_venue_sim_year = paper_year

            # 新增
            try:
                title_dist = jaro_winkler_similarity(p_feat['title'], feat['title'])
                
            except:
                title_dist = 0
            
            title_dist_list.append(title_dist)
            
            try:
                
                title_sim = cosine_similarity(TITLE_EMBEDDINGS[p_feat['id']], TITLE_EMBEDDINGS[feat['id']])
            except:
                title_sim = 0
            
            
            title_sims.append(title_sim)
            # max_title_sim = max(max_title_sim, title_sim)
            if title_sim > max_title_sim:
                max_title_sim = title_sim
                max_title_sim_year = paper_year


            try:
                abstract_sim = cosine_similarity(ABSTRACT_EMBEDDINGS[p_feat['id']], ABSTRACT_EMBEDDINGS[feat['id']])
            except:
                abstract_sim = 0

            abstract_sims.append(abstract_sim)
            # max_abstract_sim = max(max_abstract_sim, abstract_sim)
            if abstract_sim > max_abstract_sim:
                max_abstract_sim = abstract_sim
                max_abstract_sim_year = paper_year

            # 新增
            if abs(paper_year-this_year) < nearest_year_diff:
                nearest_year_diff = abs(paper_year-this_year)
                near_coauthor_dict = {
                    aut_cnt: near_authors2
                }

                near_coorgs_dict = {
                    org_cnt: near_orgs2
                }

                near_cokeyword_dict ={
                    keyword_cnt: near_keywords2
                }

                near_venue_sim = [venue_sim]
                near_title_sim = [title_sim]
                near_abstract_sim = [abstract_sim]

                near_org_dist = paper_org_dist_list
                near_org_sim = paper_org_sim_list

            elif abs(paper_year-this_year) == nearest_year_diff:

                if aut_cnt in near_coauthor_dict:
                    near_coauthor_dict[aut_cnt].union(near_authors2)
                else:
                    near_coauthor_dict[aut_cnt] = near_authors2

                if org_cnt in near_coorgs_dict:
                    near_coorgs_dict[org_cnt].union(near_orgs2)
                else:
                    near_coorgs_dict[org_cnt] = near_orgs2

                if keyword_cnt in near_cokeyword_dict:
                    near_cokeyword_dict[keyword_cnt].union(near_keywords2)
                else:
                    near_cokeyword_dict[keyword_cnt] = near_keywords2

                near_venue_sim.append(venue_sim)
                near_title_sim.append(title_sim)
                near_abstract_sim.append(abstract_sim)

                near_org_dist += paper_org_dist_list
                near_org_sim += paper_org_sim_list

        # 新增
        try:
            max_cnt_aut_over_all_author = max_cnt_aut / len(all_authors)
        except:
            max_cnt_aut_over_all_author = 0
        
        try:
            max_cnt_aut_over_this_author = max_cnt_aut / len(authors1)
        except:
            max_cnt_aut_over_this_author = 0
        
        # 新增
        try:
            max_cnt_aut_over_paper_author = max_cnt_aut / len(max_cnt_aut_authors2)
        except:
            max_cnt_aut_over_paper_author = 0

        # max_cnt_coaut_jaccard_sim = list_jaccard_similarity(authors1, max_cnt_aut_authors2)
        # aut_jaccard_sims = [list_jaccard_similarity(authors1, a2) for a2 in authors2_list]

        
        # try:
        #     max_aut_jaccard_sim = max(aut_jaccard_sims)
        # except:
        #     max_aut_jaccard_sim = 0
        
        # try:
        #     mean_aut_jaccard_sim = sum(aut_jaccard_sims) / len(aut_jaccard_sims)
        # except:
        #     mean_aut_jaccard_sim = 0


        try:
            max_cokeyword_over_all_keyword = max_cnt_keyword / len(all_keywords)
        except:
            max_cokeyword_over_all_keyword = 0

        
        try:
            max_cokeyword_over_this_keyword = max_cnt_keyword / len(keyword1)
        except:
            max_cokeyword_over_this_keyword = 0
        

        # 新增
        try:
            max_cokeyword_over_paper_keyword = max_cnt_keyword / len(max_cnt_cokeyword_keyword2)
        except:
            max_cokeyword_over_paper_keyword = 0

        
        max_cnt_cokeyword_jaccard_sim = list_jaccard_similarity(keyword1, max_cnt_cokeyword_keyword2)
        keyword_jaccard_sims = [list_jaccard_similarity(keyword1, k2) for k2 in keyword2_list]
        
        
        try:
            max_keyword_jaccard_sim = max(keyword_jaccard_sims)
        except:
            max_keyword_jaccard_sim = 0

        try:
            mean_keyword_jaccard_sim = sum(keyword_jaccard_sims) / len(keyword_jaccard_sims)
        except:
            mean_keyword_jaccard_sim = 0

        # 新增
        try:
            std_keyword_jaccard_sim = np.std(keyword_jaccard_sims)
        except:
            std_keyword_jaccard_sim = 0

        try:
            skew_keyword_jaccard_sim = np.skew(keyword_jaccard_sims)
        except:
            skew_keyword_jaccard_sim = 0

        try:
            max_cnt_org_over_this_orgs = max_cnt_org / len(orgs1)
        except:
            max_cnt_org_over_this_orgs = 0
        
        try:
            max_cnt_org_over_all_orgs = max_cnt_org / len(all_orgs)
        except:
            max_cnt_org_over_all_orgs = 0

        # 新增
        try:
            max_cnt_org_over_paper_orgs = max_cnt_org / len(max_cnt_coorgs_orgs2)
        except:
            max_cnt_org_over_paper_orgs = 0
        
        # max_cnt_org_jaccard_sim = list_jaccard_similarity(orgs1, max_cnt_coorgs_orgs2)
        # org_jaccard_sims = [list_jaccard_similarity(orgs1, o2) for o2 in orgs2_list]

        # try:
        #     max_org_jaccard_sim = max(org_jaccard_sims)
        # except:
        #     max_org_jaccard_sim = 0
        
        # try:
        #     mean_org_jaccard_sim = sum(org_jaccard_sims) / len(org_jaccard_sims)
        # except:
        #     mean_org_jaccard_sim = 0


        max_aut_year_abs_diff = abs(this_year-max_aut_year)
        max_orgs_year_abs_diff = abs(this_year-max_orgs_year)
        max_keyword_year_abs_diff = abs(this_year-max_keyword_year)
        
        # 新增
        max_venue_sim_abs_year = abs(this_year-max_venue_sim_year)
        max_title_sim_abs_year = abs(this_year-max_title_sim_year)
        max_abstract_sim_abs_year = abs(this_year-max_abstract_sim_year)


        before_years.sort()
        after_years.sort()
        
        if len(before_years) >=2:
            before_one = this_year - before_years[-1]
            before_two = this_year - before_years[-2]
        elif len(before_years) == 1:
            before_one = this_year - before_years[-1]
            before_two = 0
        else:
            before_one = 0
            before_two = 0
        
        if len(after_years) >=2:
            after_one = after_years[0] - this_year
            after_two = after_years[1] - this_year
        elif len(after_years) == 1:
            after_one = after_years[0] - this_year
            after_two = 0
        else:
            after_one = 0
            after_two = 0
        
        
        try:
            mean_title_sim = sum(title_sims) / len(title_sims)
        except:
            mean_title_sim = 0
        
        try:
            std_title_sim = np.std(title_sims)
        except:
            std_title_sim = np.inf
        
        try:
            skew_title_sim = np.skew(title_sims)
        except:
            skew_title_sim = np.inf
        
        # try:
        #     min_title_sim = min(title_sims)
        # except:
        #     min_title_sim = 0

        try:
            mean_abstract_sim = sum(abstract_sims) / len(abstract_sims)
        except:
            mean_abstract_sim = 0
        

        try:
            std_abstract_sim = np.std(abstract_sims)
        except:
            std_abstract_sim = np.inf
        
        try:
            skew_abstract_sim = np.skew(abstract_sims)
        except:
            skew_abstract_sim = np.inf

        
        # try:
        #     min_abstract_sim = min(abstract_sims)
        # except:
        #     min_abstract_sim = 0

        
        try:
            mean_venue_sim = sum(venue_sims) / len(venue_sims)
        except:
            mean_venue_sim = 0
        
        try:
            std_venue_sim = np.std(venue_sims)
        except:
            std_venue_sim = np.inf

        
        try:
            skew_venue_sim = np.skew(venue_sims)
        except:
            skew_venue_sim = np.inf
        
        # try:
        #     min_venue_sim = min(venue_sims)
        # except:
        #     min_venue_sim = 0
        

        try:
            max_org_sim = max(org_sim_list)
        except:
            max_org_sim = 0
            
        try:
            mean_org_sim = sum(org_sim_list) / len(org_sim_list)
        except:
            mean_org_sim = 0

        try:
            std_org_sim = np.std(org_sim_list)
        except:
            std_org_sim = np.inf

        try:
            skew_org_sim = np.skew(org_sim_list)
        except:
            skew_org_sim = np.inf

    
        near_max_coauthor_cnt = max(near_coauthor_dict.keys())
        
        # 新增
        try:
            near_mean_coauthor_cnt = sum(near_coauthor_dict.keys()) / len(near_coauthor_dict.keys())
        except:
            near_mean_coauthor_cnt = 0
        
        try:
            near_coauthor_cnt_over_this_aut = near_max_coauthor_cnt / len(authors1)
        except:
            near_coauthor_cnt_over_this_aut = 0
        try:
            near_coauthor_cnt_over_aut2 = near_max_coauthor_cnt / len(near_coauthor_dict[near_max_coauthor_cnt])
        except:
            near_coauthor_cnt_over_aut2 = 0
        
        
        near_max_coorg_cnt = max(near_coorgs_dict.keys())
        near_mean_coorg_cnt = sum(near_coorgs_dict.keys()) / len(near_coorgs_dict.keys())
        
        # 新增
        # near_min_coorg_cnt = min(near_coorgs_dict.keys())
        # try:
        #     near_std_coorg_cnt = np.std(near_coorgs_dict.keys())
        # except:
        #     near_std_coorg_cnt = np.inf

        try:
            near_coorg_cnt_over_this_orgs = near_max_coorg_cnt / len(orgs1)
        except:
            near_coorg_cnt_over_this_orgs = 0
        try:
            near_coorg_cnt_over_orgs2 = near_max_coorg_cnt / len(near_coorgs_dict[near_max_coorg_cnt])
        except:
            near_coorg_cnt_over_orgs2 = 0

        
        # 新增
        near_max_cokeyword_cnt = max(near_cokeyword_dict.keys())
        near_mean_cokeyword_cnt = sum(near_cokeyword_dict.keys()) / len(near_cokeyword_dict.keys())
        try:
            near_cokeyword_cnt_over_this_keyword = near_max_cokeyword_cnt / len(keyword1)
        except:
            near_cokeyword_cnt_over_this_keyword = 0
        try:
            near_cokeyword_cnt_over_keyword2 = near_max_cokeyword_cnt / len(near_cokeyword_dict[near_max_cokeyword_cnt])
        except:
            near_cokeyword_cnt_over_keyword2 = 0

        
        near_max_venue_sim  = max(near_venue_sim)
        near_max_title_sim = max(near_title_sim)
        near_max_abstract_sim = max(near_abstract_sim)

        try:
            near_mean_venue_sim  = sum(near_venue_sim) / len(near_venue_sim)
        except:
            near_mean_venue_sim = 0 
        try:
            near_mean_title_sim = sum(near_title_sim) / len(near_title_sim)
        except:
            near_mean_title_sim = 0
        try:
            near_mean_abstract_sim = sum(near_abstract_sim) / len(near_abstract_sim)
        except:
            near_mean_abstract_sim = 0

        # 新增
        # try:
        #     near_std_venue_sim  = np.std(near_venue_sim) 
        # except:
        #     near_std_venue_sim = np.inf
        # try:
        #     near_std_title_sim = np.std(near_title_sim) 
        # except:
        #     near_std_title_sim = np.inf
        # try:
        #     near_std_abstract_sim = np.std(near_abstract_sim) 
        # except:
        #     near_std_abstract_sim = np.inf
        
        # near_min_venue_sim  = min(near_venue_sim)
        # near_min_title_sim = min(near_title_sim)
        # near_min_abstract_sim = min(near_abstract_sim)


        try:
            max_org_dist = max(org_dist_list)
        except:
            max_org_dist = 0

        try:
            mean_org_dist = sum(org_dist_list) / len(org_dist_list)
        except:
            mean_org_dist = 0

        # 新增
        try:
            std_org_dist = np.std(org_dist_list)
        except:
            std_org_dist = np.inf

        try:
            skew_org_dist = np.skew(org_dist_list)
        except:
            skew_org_dist = np.inf

        try:
            max_venue_dist = max(venue_dist_list)
        except:
            max_venue_dist = 0
        
        try:
            mean_venue_dist = sum(venue_dist_list) / len(venue_dist_list)
        except:
            mean_venue_dist = 0

        # 新增
        try:
            std_venue_dist = np.std(venue_dist_list)
        except:
            std_venue_dist = np.inf

        try:
            skew_venue_dist = np.skew(venue_dist_list)
        except:
            skew_venue_dist = np.inf

        # -------------
        try:
            max_title_dist = max(title_dist_list)
        except:
            max_title_dist = 0
        
        try:
            mean_title_dist = sum(title_dist_list) / len(title_dist_list)
        except:
            mean_title_dist = 0

        # 新增
        try:
            std_title_dist = np.std(title_dist_list)
        except:
            std_title_dist = np.inf

        try:
            skew_title_dist = np.skew(title_dist_list)
        except:
            skew_title_dist = np.inf


        try:
            near_mean_org_dist = sum(near_org_dist) / len(near_org_dist)
        except:
            near_mean_org_dist = 0
        
        try:
            near_max_org_dist = max(near_org_dist)
        except:
            near_max_org_dist = 0

        try:
            near_mean_org_sim = sum(near_org_sim) / len(near_org_sim)
        except:
            near_mean_org_sim = 0
        
        try:
            near_max_org_sim = max(near_org_sim)
        except:
            near_max_org_sim = 0
        
        # try:
        #     max_cnt_venue_word_over_this_venue = max_cnt_venue_word / len(this_venue_word_set)
        # except:
        #     max_cnt_venue_word_over_this_venue = 0
        
        # try:
        #     max_cnt_venue_word_over_paper_venue = max_cnt_venue_word / len(max_paper_venue_word_set)
        # except:
        #     max_cnt_venue_word_over_paper_venue = 0

        # 新增
        try:
            max_this_user_org_in_list_num_over_paper_num = max_this_user_org_in_list_num / len(paper_list)
        except:
            max_this_user_org_in_list_num_over_paper_num = 0
        
        paper_num = len(paper_list)
        
        return [max_cnt_aut, max_cnt_org, max_cnt_aut_org_cnt, 
                
                max_venue_sim, max_title_sim,max_abstract_sim,  
                
                max_cnt_keyword, max_cokeyword_over_all_keyword,max_cokeyword_over_this_keyword, coaut_papar_count,
                
                max_cnt_org_over_this_orgs, max_cnt_org_over_all_orgs,
                
                max_aut_year_abs_diff, max_orgs_year_abs_diff, max_keyword_year_abs_diff,
                
                before_one, before_two, after_one, after_two,
                
                mean_title_sim, mean_abstract_sim, mean_venue_sim,
                is_match_venue,
                
                near_max_coauthor_cnt, near_coauthor_cnt_over_this_aut, near_coauthor_cnt_over_aut2,near_mean_coauthor_cnt,
                near_max_coorg_cnt, near_coorg_cnt_over_this_orgs, near_coorg_cnt_over_orgs2,near_mean_coorg_cnt,
                
                near_max_venue_sim, near_max_title_sim, near_max_abstract_sim, near_mean_venue_sim, near_mean_title_sim, near_mean_abstract_sim,
                
                max_venue_dist,

                max_org_dist, mean_org_dist,max_org_sim,mean_org_sim,
                near_mean_org_dist, near_max_org_dist, near_mean_org_sim, near_max_org_sim,
                # max_cnt_venue_word_over_this_venue, max_cnt_venue_word_over_paper_venue, max_cnt_venue_word
                max_this_user_org_in_list_num, max_this_user_org_in_list_num_over_paper_num, max_this_user_org_in_list_ratio, paper_num,
                
                # 新增
                this_user_orgs_in_paper_list_num, this_user_in_paper_list_num, this_venue_in_paper_list_num, keyword_in_paper_list_num,
                
                std_title_sim, std_abstract_sim, std_venue_sim,
                
                # 新增
                # min_title_sim, min_abstract_sim, min_venue_sim,
                # near_min_coorg_cnt, near_std_coorg_cnt,
                # near_std_venue_sim, near_std_title_sim, near_std_abstract_sim, near_min_venue_sim, near_min_title_sim, near_min_abstract_sim,
                
                
                std_org_sim, mean_venue_dist,
                
                max_cokeyword_over_paper_keyword, 
                
                max_cnt_cokeyword_jaccard_sim, max_keyword_jaccard_sim, mean_keyword_jaccard_sim,
                
                max_cnt_aut_over_all_author, max_cnt_aut_over_this_author,
                
                # 新增
                # max_cnt_aut_over_paper_author, 
                # max_cnt_coaut_jaccard_sim, 
                # max_cnt_org_over_paper_orgs, 
                # max_cnt_org_jaccard_sim,
            #    max_aut_jaccard_sim, mean_aut_jaccard_sim, max_org_jaccard_sim, mean_org_jaccard_sim,
                
                # 新增
                std_venue_dist, std_org_dist, 
                # skew_venue_dist, skew_org_dist,
                # skew_org_sim, skew_abstract_sim, skew_title_sim, skew_venue_sim,

                # 新增
                # std_keyword_jaccard_sim, skew_keyword_jaccard_sim, 

                near_max_cokeyword_cnt, near_mean_cokeyword_cnt, near_cokeyword_cnt_over_this_keyword, near_cokeyword_cnt_over_keyword2,
                max_cnt_aut_over_paper_author, max_cnt_org_over_paper_orgs,

                # 新增
                before_match_org_cnt, is_match_org,
                before_match_venue_cnt,

                all_match_org_cnt, all_match_venue_cnt,

                max_venue_sim_abs_year, max_title_sim_abs_year, max_abstract_sim_abs_year,
                
                # 新增
                max_title_dist, mean_title_dist, std_title_dist, skew_title_dist
                ]

    # def max_coorg(p_feat, paper_list):
    #     max_cnt = 0

    #     authors1 = []
    #     for i in p_feat['authors']:
    #         authors1.append(i['org'].lower())
        
    #     for paper in paper_list:
    #         feat = pid_to_info[paper]
    #         authors2 = []

    #         for i in feat['authors']:
    #             authors2.append(i['org'].lower())
            
    #         cnt = coauthor(authors1, authors2)

    #         if cnt > max_cnt:
    #             max_cnt += 1
    #     return max_cnt


def user_cross_feat_engineer(user, p_feat, person_info):
    try:
        normal_p = person_info['normal_data']
        out_p = person_info['outliers']

        all_p = normal_p + out_p
    except:
        all_p = person_info['papers']
    
    # 删除自身
    id = p_feat['id']
    tp = copy.deepcopy(all_p)
    tp.remove(id)
    
    cross_features = corss_feature_enginner(user, p_feat, tp)
    cross_columns = ['max_coauthor', 'max_coorg','max_cnt_aut_org_cnt', 'max_venue_sim', 'max_title_sim','max_abstract_sim',
                    'max_cokeyword', 'max_cokeyword_over_all_keyword', 'max_cokeyword_over_this_keyword', 'coauthor_paper_count',
                    'max_cnt_org_over_this_orgs', 'max_cnt_org_over_all_orgs',
                    'max_aut_year_abs_diff', 'max_orgs_year_abs_diff', 'max_keyword_year_abs_diff',
                    'before_one', 'before_two', 'after_one', 'after_two',
                    'mean_title_sim', 'mean_abstract_sim', 'mean_venue_sim',
                    'is_match_venue',
                    'near_coauthor_cnt', 'near_coauthor_cnt_over_this_aut', 'near_coauthor_cnt_over_aut2','near_mean_coauthor_cnt',
                    'near_max_coorg_cnt', 'near_coorg_cnt_over_this_orgs', 'near_coorg_cnt_over_orgs2','near_mean_coorg_cnt',
                    'near_max_venue_sim', 'near_max_title_sim', 'near_max_abstract_sim', 'near_mean_venue_sim', 'near_mean_title_sim', 'near_mean_abstract_sim',
                    'max_venue_dist',
                    'max_org_dist', 'mean_org_dist','max_org_sim','mean_org_sim',
                    
                    'near_mean_org_dist', 'near_max_org_dist', 'near_mean_org_sim', 'near_max_org_sim',
                    # 'max_cnt_venue_word_over_this_venue', 'max_cnt_venue_word_over_paper_venue', 'max_cnt_venue_word',
                    'max_this_user_org_in_list_num', 'max_this_user_org_in_list_num_over_paper_num', 'max_this_user_org_in_list_ratio', 'paper_num',
                    
                    'this_user_orgs_in_paper_list_num', 'this_user_in_paper_list_num', 'this_venue_in_paper_list_num', 'keyword_in_paper_list_num',
                    'std_title_sim', 'std_abstract_sim', 'std_venue_sim',

                    # 'min_title_sim', 'min_abstract_sim', 'min_venue_sim',
                    # 'near_min_coorg_cnt', 'near_std_coorg_cnt',
                    # 'near_std_venue_sim', 'near_std_title_sim', 'near_std_abstract_sim', 'near_min_venue_sim', 'near_min_title_sim', 'near_min_abstract_sim',
                    'std_org_sim', 'mean_venue_dist',
                    'max_cokeyword_over_paper_keyword', 'max_cnt_cokeyword_jaccard_sim', 'max_keyword_jaccard_sim', 'mean_keyword_jaccard_sim',
                    'max_cnt_aut_over_all_author', 'max_cnt_aut_over_this_author',
                    
                    # 'max_cnt_aut_over_paper_author', 
                    # 'max_cnt_coaut_jaccard_sim', 
                    # 'max_cnt_org_over_paper_orgs', 
                    # 'max_cnt_org_jaccard_sim',
                    # 'max_aut_jaccard_sim', 'mean_aut_jaccard_sim', 'max_org_jaccard_sim', 'mean_org_jaccard_sim',

                    'std_venue_dist', 'std_org_dist', 
                    # 'skew_venue_dist', 'skew_org_dist',
                    # 'skew_org_sim', 'skew_abstract_sim', 'skew_title_sim', 'skew_venue_sim',

                    # 'std_keyword_jaccard_sim', 'skew_keyword_jaccard_sim', 
                    'near_max_cokeyword_cnt', 'near_mean_cokeyword_cnt', 'near_cokeyword_cnt_over_this_keyword', 'near_cokeyword_cnt_over_keyword2',
                    'max_cnt_aut_over_paper_author', 'max_cnt_org_over_paper_orgs',

                    'before_match_org_cnt', 'is_match_org',
                    'before_match_venue_cnt',

                    'all_match_org_cnt', 'all_match_venue_cnt',

                    'max_venue_sim_abs_year', 'max_title_sim_abs_year', 'max_abstract_sim_abs_year',
                    
                    'max_title_dist', 'mean_title_dist', 'std_title_dist', 'skew_title_dist'
                    ]
    return cross_features, cross_columns



def dataframe_cross_feat_engineer(df:pd.DataFrame):
    df['max_coauthor_over_authors_len'] = df['max_coauthor'] / df['authors_len']
    # df.groupby('name')['max_coauthor'].transform(lambda x: x/x.sum())
    
    
    for i in ['min','max', 'mean', 'median', 'std','skew']:
        
        df[f'year_{i}'] = df.groupby(['name'])['year'].transform(i)
        
        df[f'max_coauthor_{i}'] = df.groupby(['name'])['max_coauthor'].transform(i)
        df[f'authors_len_{i}'] = df.groupby(['name'])['authors_len'].transform(i)
    
        df[f'keywords_len_{i}'] = df.groupby(['name'])['keywords_len'].transform(i)
        
        # 新增
        # df[f'max_coorg_{i}'] = df.groupby(['name'])['max_coorg'].transform(i)
        
        # df[f'title_max_sim_{i}'] = df.groupby(['name'])['title_max_sim'].transform(i)
        # df[f'abstract_max_sim_{i}'] = df.groupby(['name'])['abstract_max_sim'].transform(i)
        # df[f'venue_max_dista nce_{i}'] = df.groupby(['name'])['venue_max_distance'].transform(i)
        
        # df[f'max_cnt_aut_org_cnt_{i}'] = df.groupby(['name'])['max_cnt_aut_org_cnt'].transform(i)
        # df[f'coauthor_paper_count_{i}'] = df.groupby(['name'])['coauthor_paper_count'].transform(i)
        
    # df['year_gap_min'] = df['year'] - Min_year
    df['year_gap_min'] = df['year'] - 1835
    
    df['year_gap_max'] = df['year'].max() - df['year']
    df['user_year_gap_min'] = df.groupby('name')['year'].transform(lambda x: x-x.min())
    df['user_year_gap_max'] = df.groupby('name')['year'].transform(lambda x: x.max()-x)
    
    df['this_year_cnt'] = df.groupby(['name', 'year'])['year'].transform('count')
    df['this_year_cnt_over_all_year'] = df['this_year_cnt'] / df.groupby(['name'])['year'].transform('count')
    # df['this_year_is_appear'] = df['this_year_cnt'].apply(lambda x: 1 if x > 1 else 0)

    # 新增
    # for i in ['mean', 'median']:
    #     df[f'user_year_gap_{i}'] = df['year'] - df[f'year_{i}']
    
    # df['is_in_range'] = df['year'].apply(lambda x: 1 if Min_year <= x <= 2024 else 0)
    df['is_in_range'] = df['year'].apply(lambda x: 1 if 1835 <= x <= 2024 else 0)
    
    
    df['year_is_appear'] = df['this_year_cnt'].apply(lambda x: 1 if x > 1 else 0)
    df[f'user_year_gap_min_max_avg'] = df['year'] - (df['year_min'] + df['year_max']) / 2
    
    # 新增
    def _near_year(x, df):
        name = x['name']
        ys = list(df[df['name'] == name]['year'])
        
        min_diff = 10000
        result = 0
        for y in ys:
            diff = abs(y-x['year'])
        
            if diff < min_diff and diff > 0:
                min_diff  = diff
                result = y
        return result
    df['near_year'] = df.swifter.apply(lambda x: _near_year(x, df), axis=1)
    df['near_year_diff'] = df['year'] - df['near_year']
    
    df['max_venue_sim_add_dist'] = df['max_venue_dist'] + df['max_venue_sim']
    df['max_org_sum_add_dist'] = df['max_org_sim'] + df['max_org_dist']
    
    # 新增
    df['near_max_org_dist_add_sim'] = df['near_max_org_dist'] + df['near_max_org_sim']
    # df['near_max_title_sim'] = df['near_max_title_sim'] + df['near_max_title_dist']
    # df['near_max_abstract_sim'] = df['near_max_abstract_sim'] + df['near_max_abstract_dist']
    
    df['max_coauthor_over_paper_num'] = df['max_coauthor'] / df['paper_num']
    df['max_coorg_over_paper_num'] = df['max_coorg'] / df['paper_num']
    df['max_cnt_aut_org_cnt_over_paper_num'] = df['max_cnt_aut_org_cnt'] / df['paper_num']
    # 新增
    df['max_cnt_keyword_over_paper_num'] = df['max_cokeyword'] / df['paper_num']
    
    df['this_user_orgs_in_paper_list_num_over_paper_num'] = df['this_user_orgs_in_paper_list_num'] / df['paper_num']
    df['this_user_in_paper_list_num_over_paper_num'] = df['this_user_in_paper_list_num'] / df['paper_num']
    df['this_venue_in_paper_list_num_over_paper_num'] = df['this_venue_in_paper_list_num'] / df['paper_num']
    df['keyword_in_paper_list_num_over_paper_num'] = df['keyword_in_paper_list_num'] / df['paper_num']
    
    # 新增
    # df['coaut_paper_count_over_paper_num'] = df['coauthor_paper_count'] / df['paper_num']
    df['year_gap_mean'] = df['year'] - df['year'].mean()
    df['year_gap_median'] = df['year'] - df['year'].median()

    # df['year_gap_min_max_ave'] = df['year'] - (Min_year + df['year'].max()) / 2
    df['year_gap_min_max_ave'] = df['year'] - (1835 + df['year'].max()) / 2

    df['before_match_org_cnt_over_all_cnt'] = df['before_match_org_cnt'] / df['all_match_org_cnt']
    df['before_match_venue_cnt_over_all_cnt'] = df['before_match_venue_cnt'] / df['all_match_venue_cnt']
    
    # 新增
    df['max_title_sim_add_dist'] = df['max_title_dist'] + df['max_title_sim']
    return df

def main(train_author,valid_author, pid_to_info, submission, task):
    
    

    # +title_embedding_features  

    # def co(l1, l2):
    #     cnt = 0
    #     for i in l1:
    #         if i in l2:
    #             cnt +=1
    #     return cnt

    
    train_feats=[]
    labels=[]
    for id,person_info in tqdm(train_author.items()):
        user = person_info['name']
        for text_id in person_info['normal_data']:#正样本
            feat=pid_to_info[text_id]
            #['title', 'abstract', 'keywords', 'authors', 'venue', 'year']
            title = feat['title']
            abstract = feat['abstract']
            venue = feat['venue']

            features, column_names = base_feat_engineer(feat)
            c_features, cross_columns = user_cross_feat_engineer(user, feat, person_info)

            train_feats.append(
                    [id, user, title, abstract, venue] 
                    + features 
                    + c_features
                    )
        
            labels.append(1)
        for text_id in person_info['outliers']:#负样本
            feat=pid_to_info[text_id]
            #['title', 'abstract', 'keywords', 'authors', 'venue', 'year']
            title = feat['title']
            abstract = feat['abstract']
            venue = feat['venue']
            
            features, column_names = base_feat_engineer(feat)
            c_features, cross_columns = user_cross_feat_engineer(user, feat, person_info)

            train_feats.append(
                    [id, user, title, abstract, venue] 
                    + features
                    + c_features
                    )
            labels.append(0)   

    train_feats=pd.DataFrame(train_feats, columns=['id','name', 'title', 'abstract', 'venue'] + column_names + cross_columns)
    train_feats['label']=labels
    # train_feats.head()
    train_feats = dataframe_cross_feat_engineer(train_feats)

    valid_feats=[]
    for id,person_info in tqdm(valid_author.items()):
        user = person_info['name']
        for text_id in person_info['papers']:
            feat=pid_to_info[text_id]
            #['title', 'abstract', 'keywords', 'authors', 'venue', 'year']
            
            title = feat['title']
            abstract = feat['abstract']
            venue = feat['venue']

            features, column_names = base_feat_engineer(feat)
            c_features, cross_columns = user_cross_feat_engineer(user, feat, person_info)

            valid_feats.append(
                [id, user, title, abstract, venue] + features + c_features
                    )
            
    # valid_feats=np.array(valid_feats)
    # print(f"valid_feats.shape:{valid_feats.shape}")
    valid_feats=pd.DataFrame(valid_feats,columns=['id', 'name', 'title', 'abstract', 'venue']+ column_names +cross_columns)
    valid_feats = dataframe_cross_feat_engineer(valid_feats)

    # del ABSTRACT_EMBEDDINGS
    # del TITLE_EMBEDDINGS
    # del VENUE_EMBEDDINGS
    # del ORGS_EMBEDDINGS
    # gc.collect()

    all_feats = pd.concat([train_feats, valid_feats])
    all_feats.reset_index(drop=True, inplace=True)

    n_components = 32
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    title_svd = svd.fit_transform(all_feats[TITLE_EMBEDDING_FEATURES])
    title_svd_df = pd.DataFrame(title_svd, columns=[f'title_embed_svd_{i}' for i in range(title_svd.shape[-1])])
    all_feats = pd.concat([all_feats, title_svd_df], axis=1)
    all_feats = all_feats.drop(TITLE_EMBEDDING_FEATURES, axis=1)

    del title_svd_df
    gc.collect()

    abstract_svd = svd.fit_transform(all_feats[ABSTRACT_EMBEDDING_FEATURES])
    abstract_svd_df = pd.DataFrame(abstract_svd, columns=[f'abstract_embed_svd_{i}' for i in range(abstract_svd.shape[-1])])
    all_feats = pd.concat([all_feats, abstract_svd_df], axis=1)
    all_feats = all_feats.drop(ABSTRACT_EMBEDDING_FEATURES, axis=1)

    del abstract_svd_df
    gc.collect()

    venue_svd = svd.fit_transform(all_feats[VENUE_EMBEDDING_FEATURES])
    venue_svd_df = pd.DataFrame(venue_svd, columns=[f'venue_embed_svd_{i}' for i in range(venue_svd.shape[-1])])
    all_feats = pd.concat([all_feats, venue_svd_df], axis=1)
    all_feats = all_feats.drop(VENUE_EMBEDDING_FEATURES, axis=1)

    del venue_svd_df
    gc.collect()

    train_feats = all_feats[~all_feats[Config.TARGET_NAME].isna()].reset_index(drop=True)
    valid_feats = all_feats[all_feats[Config.TARGET_NAME].isna()].reset_index(drop=True)

    valid_feats = valid_feats.drop('label',axis=1)

    choose_cols=[col for col in valid_feats.columns if col not in ['id','name','text_id' , 'title', 'abstract', 'venue','title_len', 'title_split_len', 'paper_num'] + 
                ['year_mean', 'year_std', 'year_skew'] 
                + ['coauthor_paper_count' + 'max_coauthor_max', 'keywords_len_max'] 
                + ['max_venue_sim_abs_year', 'max_title_sim_abs_year', 'max_abstract_sim_abs_year']
                + [
                    'near_max_cokeyword_cnt', 'near_mean_cokeyword_cnt', 'near_cokeyword_cnt_over_this_keyword', 'near_cokeyword_cnt_over_keyword2',
                    'max_cnt_aut_over_paper_author', 'max_cnt_org_over_paper_orgs',

                        # 'before_match_org_cnt', 'is_match_org',
                        # 'before_match_venue_cnt',

                        # 'all_match_org_cnt', 'all_match_venue_cnt',

                        'max_venue_sim_abs_year', 'max_title_sim_abs_year', 'max_abstract_sim_abs_year',
                        'year_gap_mean', 'year_gap_median', 'year_gap_min_max_ave',
                        
                        # 'before_match_org_cnt_over_all_cnt','before_match_venue_cnt_over_all_cnt'
                        'max_title_dist', 'mean_title_dist', 'std_title_dist', 'skew_title_dist', 'max_title_sim_add_dist'
                        ]
                #  + [ 'coaut_paper_count_over_paper_num']
                # + ['max_cnt_aut_over_paper_author', 'max_cnt_coaut_jaccard_sim', 'max_cnt_org_over_paper_orgs', 'max_cnt_org_jaccard_sim',
                #         'max_aut_jaccard_sim', 'mean_aut_jaccard_sim',]
                
                #  ['max_coauthor_skew', 'keywords_len_skew']
                #  ['near_max_coorg_cnt', 'near_mean_coorg_cnt']
                ]

    # choose_cols=[col for col in valid_feats.columns if col not in ['name', 'title_len', 'title_split_len'] + 
    #              toxic_feats]
    def fit_and_predict(model,train_feats=train_feats,test_feats=valid_feats,name=0):
        X=train_feats[choose_cols].copy()
        y=train_feats[Config.TARGET_NAME].copy()
        test_X=test_feats[choose_cols].copy()
        oof_pred_pro=np.zeros((len(X),2))
        test_pred_pro=np.zeros((Config.num_folds,len(test_X),2))

        #10折交叉验证
        skf = StratifiedKFold(n_splits=Config.num_folds,random_state=Config.seed, shuffle=True)
        models = [] 
        for fold, (train_index, valid_index) in (enumerate(skf.split(X, y.astype(str)))):
            print(f"name:{name},fold:{fold}")

            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
            model.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],
                    callbacks=[log_evaluation(500),early_stopping(100)]
                    )
            
            oof_pred_pro[valid_index]=model.predict_proba(X_valid)
            #将数据分批次进行预测.
            test_pred_pro[fold]=model.predict_proba(test_X)

            models.append(model)
        roc_auc = roc_auc_score(y.values,oof_pred_pro[:,1])
        
        return oof_pred_pro,test_pred_pro, roc_auc, models

        
    #参数来源:https://www.kaggle.com/code/daviddirethucus/home-credit-risk-lightgbm
    lgb_params={
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        # "max_depth": 12,
        "max_depth": 9,
        "learning_rate": 0.05,
        "n_estimators":3072,
        "colsample_bytree": 0.9,
        "colsample_bynode": 0.9,
        "verbose": -1,
        "random_state": Config.seed,
        # "reg_alpha": 0.1,
        "reg_alpha": 0.0011426364370069847,
        # "reg_lambda": 10,
        "reg_lambda":3.4006859695041123,
        "extra_trees":True,
        'num_leaves':64,
        "max_bin":255,
        }
    # xgb_params={
    #     "boosting_type": "gbdt",
    #     "objective": "binary",
    #     "metric": "auc",
    #     "max_depth": 7,
    #     "learning_rate": 0.1,
    #     "n_estimators":1500,
    #     "verbose": -1,
    #     "random_state": 42,
    #     "extra_trees":True,
    #     "subsample": 0.8,
    #     "min_child_weight": 2
    #     }

    lgb_oof_pred_pro,lgb_test_pred_pro, socre, models =fit_and_predict(model= LGBMClassifier(**lgb_params),
                                                                    train_feats=train_feats,test_feats=valid_feats, name='lgb'
                                                    )
    # xgb = XGBClassifier(
    #                 max_depth=7, learning_rate=0.1, n_estimators=1500, subsample=0.8,
    #                 n_jobs=-1, min_child_weight=2, random_state=2024
    #             )
    # xgb_oof_pred_pro,xgb_test_pred_pro, socre, models =fit_and_predict(model= XGBClassifier(**xgb_params),
    #                                                                    train_feats=train_feats,test_feats=valid_feats, name='lgb'
    #                                                   )

    print(f'AUC:{socre}')
    test_preds=lgb_test_pred_pro.mean(axis=0)[:,1]

    cnt=0
    for id,names in submission.items():
        for name in names:
            submission[id][name]=test_preds[cnt]
            cnt+=1
    
    res_path = f'../output/e5_instruct_lgb.json'
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)

    print(res_path)
    
    
    out_cols=[col for col in valid_feats.columns if col not in ['title', 'abstract', 'venue','title_len', 'title_split_len', 'paper_num'] + 
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
                    ]]

    out_train = train_feats[out_cols].fillna(0)
    out_test = valid_feats[out_cols].fillna(0)
    out_train.to_csv('../out_data/e5_instruct_train.csv', index=False)
    out_test.to_csv('../out_data/e5_instruct_test.csv', index=False)
    return res_path


if __name__ == '__main__':

    path='../'
    #sample: Iki037dt dict_keys(['name', 'normal_data', 'outliers'])
    with open(path+"IND-WhoIsWho/train_author.json") as f:
        train_author=json.load(f)
    #sample : 6IsfnuWU dict_keys(['id', 'title', 'authors', 'abstract', 'keywords', 'venue', 'year'])   

    #efQ8FQ1i dict_keys(['name', 'papers'])

    # with open(path+"IND-WhoIsWho/ind_test_author_filter_public.json") as f:
    #     valid_author=json.load(f)

    with open(path+"IND-WhoIsWho/ind_test_author_filter_public.json") as f:
        valid_author=json.load(f)

    with open(path+"IND-WhoIsWho/ind_test_author_submit.json") as f:
        submission=json.load(f)

    rp = main(train_author,valid_author, pid_to_info, submission, 'e5_instruct_pred')
