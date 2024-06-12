import json as js
import pandas as pd
import numpy as np
import pickle as pk
from unidecode import unidecode
import torch
from torch_geometric.data.batch import Batch 
import multiprocessing as mp
import re
import argparse
import os
from collections import Counter
from jellyfish import jaro_winkler_similarity

RECORDS = []

puncs = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
stopwords = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with',
            'the', 'by', 'we', 'be', 'is', 'are', 'can']
stopwords_extend = ['university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab',
                    'school', 'al', 'et', 'institute', 'inst', 'college', 'chinese', 'beijing',
                    'journal', 'science', 'international', 'key', 'sciences', 'research',
                    'academy', 'state', 'center']
stopwords_check = ['a', 'was', 'were', 'that', '2', 'key', '1', 'technology', '0', 'sciences', 'as',
                    'from', 'r', '3', 'academy', 'this', 'nanjing', 'shanghai', 'state', 's', 'research',
                    'p', 'results', 'peoples', '4', 'which', '5', 'high', 'materials', 'study', 'control',
                    'method', 'group', 'c', 'between', 'or', 'it', 'than', 'analysis', 'system',  'sci',
                    'two', '6', 'has', 'h', 'after', 'different', 'n', 'national', 'japan', 'have', 'cell',
                    'time', 'zhejiang', 'used', 'data', 'these']

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

def cosine_similarity(v1, v2):
    return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def org_venue_features(n1_attr, n2_attr, score_dict, default_value):
    n1_attr_set = set(n1_attr.split())
    n2_attr_set = set(n2_attr.split())

    inter_words = set(n1_attr_set) & set(n2_attr_set)
    scores = 0.0
    
    if(len(inter_words) > 0):
        for inter in inter_words:
            scores += score_dict.get(inter, default_value)
        if len(n1_attr_set) > len(n2_attr_set):
            divide = n1_attr_set
        else:
            divide = n2_attr_set

        divide_score = 0.0
        for each in divide:
            divide_score += score_dict.get(each, default_value)
        
        if(divide_score * 1 <= scores):
            scores = scores / divide_score
        else:
            scores = 0.0
    return scores
    
def co_occurance(core_name, paper1, paper2):

    core_name = clean_name(core_name)
    coauthor_weight = 0
    coorg_weight = 0
    covenue_weight = 0
    ori_n1_authors = [clean_name(paper1["authors"][ins_index]["name"]).strip() for ins_index in range(min(len(paper1["authors"]), 50))]
    ori_n2_authors = [clean_name(paper2["authors"][ins_index]["name"]).strip() for ins_index in range(min(len(paper2["authors"]), 50))]
    
    #remove disambiguate author
    for name in ori_n1_authors:
        if simple_name_match(core_name,name):
            ori_n1_authors.remove(name)

    for name in ori_n2_authors:
        if simple_name_match(core_name,name):
            ori_n2_authors.remove(name)

    whole_authors = max(len(set(ori_n1_authors+ori_n2_authors)),1)

    matched = []
    for per_n1 in ori_n1_authors:
        for per_n2 in ori_n2_authors:
            if(simple_name_match(per_n1, per_n2)):
                matched.append((per_n1, per_n2))
                coauthor_weight += 1
                break

    coauthor_weight = coauthor_weight/whole_authors
    

    def jaccard_similarity(list1, list2):
        if not list1 or not list2:
            return 0
        intersection = len(set(list1) & set(list2))
        union = len(set(list1)) + len(set(list2)) - intersection
        return intersection / union if union != 0 else 0

    def over1(list1, list2):
        if not list1 or not list2:
            return 0
        intersection = len(set(list1) & set(list2))
        union = len(set(list1))
        return intersection / union if union != 0 else 0
    
    def over2(list1, list2):
        if not list1 or not list2:
            return 0
        intersection = len(set(list1) & set(list2))
        union = len(set(list2))
        return intersection / union if union != 0 else 0

    n1_org = ' '.join([i['org'] for i in paper1['authors'] if i['org'] != '']).split()
    n2_org = ' '.join([i['org'] for i in paper2['authors'] if i['org'] != '']).split()

    n1_venue = paper1['venue'].split()
    n2_venue = paper2['venue'].split()

    n1_keywords = [k.lower().strip() for k in paper1['keywords']]
    n2_keywords = [k.lower().strip() for k in paper2['keywords']]

    coorg_jacd_weight = jaccard_similarity(n1_org,n2_org)
    covenue_jacd_weight = jaccard_similarity(n1_venue,n2_venue)
    cokeyword_jacd_weight = jaccard_similarity(n1_keywords,n2_keywords)

    # coorg_over_orgs1 = over1(n1_org,n2_org)
    # coorg_over_orgs2 = over2(n1_org,n2_org)

    # covenue_over_venue1 = over1(n1_venue,n2_venue)
    # covenue_over_venue2 = over2(n1_venue,n2_venue)

    # cokeyword_over_keyword1 = over1(n1_keywords,n2_keywords)
    # cokeyword_over_keyword2 = over2(n1_keywords,n2_keywords)

    # covenue_cos_sim = cosine_similarity(dic_venue_embedding[paper1['id']]['venue'], dic_venue_embedding[paper2['id']]['venue'])
    # title_cos_sim = cosine_similarity(dic_title_embedding[paper1['id']]['title'], dic_title_embedding[paper2['id']]['title'])
    # abstract_cos_dim = cosine_similarity(dic_abstract_embedding[paper1['id']]['abstract'], dic_abstract_embedding[paper2['id']]['abstract'])

    # year_diff = abs(paper1['year'] - paper2['year'])

    return matched, coauthor_weight, coorg_jacd_weight, covenue_jacd_weight, [cokeyword_jacd_weight, 
    # covenue_cos_sim, title_cos_sim, abstract_cos_dim, year_diff,
    # coorg_over_orgs1, coorg_over_orgs2, covenue_over_venue1, covenue_over_venue2, cokeyword_over_keyword1, cokeyword_over_keyword2,
    ]

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

def getdata(orcid):
    trainset = True 
    if "normal_data" in author_names[orcid]:
        normal_papers_id = author_names[orcid]["normal_data"]
        outliers_id = author_names[orcid]["outliers"]
        all_pappers_id = normal_papers_id + outliers_id
    elif "papers" in author_names[orcid]:
        all_pappers_id = author_names[orcid]["papers"]
        trainset = False
    total_matrix, total_weight = [], []
    
    for ii in range(len(all_pappers_id)):
        paper1_id = all_pappers_id[ii]
        for jj in range(len(all_pappers_id)):
            paper2_id = all_pappers_id[jj]
            if paper1_id == paper2_id:
                continue
            
            paper1_inf = papers_info[paper1_id]
            paper2_inf = papers_info[paper2_id]
            
            _, w_coauthor, jac_w_coorg, jac_w_covenue, add_features = co_occurance(author_names[orcid]['name'], paper1_inf, paper2_inf)
            if w_coauthor + jac_w_coorg + jac_w_covenue == 0:
                continue
            else:
                total_matrix.append([paper1_id, paper2_id])
                total_weight.append([w_coauthor , jac_w_coorg , jac_w_covenue] + add_features)

    num_papers = len(all_pappers_id)

    # re-numbering
    re_num = dict(zip(all_pappers_id, list(range(num_papers))))
    # edge_index
    if trainset:
        set_norm = set(normal_papers_id)
        set_out = set(outliers_id)
        list_edge_y = [0 if (i in set_out) or (j in set_out) else 1 for i,j in total_matrix]
        
    else:
        list_edge_y = [1] * len(total_matrix)

    total_matrix = [[re_num[i],re_num[j]] for i,j in total_matrix]
    edge_index = np.array(total_matrix, dtype=np.int64).T
    

    # features
    # list_x = []
    # for x in all_pappers_id:
    #     tmp = dic_paper_embedding[x]
    #     for i in range(len(tmp)):
    #         if tmp[i] == np.inf or tmp[i] == -np.inf:
    #             tmp[i] = 0
        
    #     list_x.append(tmp)

    # list_x = [dic_title_embedding[x]['title'] + dic_abstract_embedding[x]['abstract'] for x in all_pappers_id]
    # list_x = [np.concatenate((dic_title_embedding[x]['title'], dic_abstract_embedding[x]['abstract'], dic_venue_embedding[x]['venue']))  for x in all_pappers_id]
    

    list_x = []
    for x in all_pappers_id:
    #     name_list = papers_info[x]['authors']
    #     orgs = [i['org'] for i in name_list]
    #     orgs_cnt_dict = Counter(orgs)
    #     if len(orgs_cnt_dict) > 0:
    #         max_cnt_org_str = sorted(orgs_cnt_dict.items())[0][0]
    #     else:
    #         max_cnt_org_str = ''

        pi = papers_info[x]

        year = pi['year']
        if year == '':
            year =0

        keywords_cnt = len(pi['keywords'])
        authors_cnt = len(pi['authors'])
        
        base_org_features, _ = org_feat(pi)
        
        feats_array = np.array([keywords_cnt, authors_cnt, year] + base_org_features)
        
        lgb_feats = feats_data[(feats_data['text_id'] == x) & (feats_data['name']==author_names[orcid]['name'])]
        lgb_feats = lgb_feats.drop('text_id', axis=1)
        lgb_feats = lgb_feats.drop('name', axis=1)
        lgb_feats = lgb_feats.drop('id', axis=1)
        idx = lgb_feats.index[0]
        lgb_feats_array = lgb_feats.loc[idx, :].values.reshape(-1,)
        
        feats_data.drop(index=idx, inplace=True)
        list_x.append(np.concatenate(
            (dic_title_embedding[x], dic_abstract_embedding[x], dic_venue_embedding[x], feats_array, lgb_feats_array)
            ))
    
    try:
        features = np.stack(list_x)
    except:
        print('e')
        pass

    
    # node labels
    if trainset:
        list_y = len(normal_papers_id) * [1] + len(outliers_id) * [0]
    else: 
        list_y = None

    # build batch
    batch = [0] * num_papers

    if edge_index.size == 0:  #if no edge, for rare cases, add default self loop with low weight
        e = [[],[]]
        for i in range(len(all_pappers_id)):
            for j in range(len(all_pappers_id)):
                if i != j:
                    e[0].append(i)
                    e[1].append(j)
        edge_index = e
        total_weight = [[0.0001,0.0001,0.0001]+[0.0001] * len(add_features)] * len(e[0])
        if trainset:
            list_edge_y =[]
            for i in range(len(edge_index[0])):
                if list_y[edge_index[0][i]] == 1 and list_y[edge_index[1][i]] == 1:
                    list_edge_y.append(1)
                else:
                    list_edge_y.append(0)
    #build data

    data = Batch(x=torch.tensor(features, dtype=torch.float32), 
                edge_index=torch.tensor(edge_index), 
                edge_attr=torch.tensor(total_weight, dtype = torch.float32),
                y=torch.tensor(list_y) if list_y is not None else None,
                batch=torch.tensor(batch))

    assert torch.any(torch.isnan(data.x)) == False
    edge_label = torch.tensor(list_edge_y) if trainset else None

    return (data,edge_label,orcid,all_pappers_id)

def build_dataset(path):
    
    keys_list = list(author_names.keys())
    results = []

    with mp.Pool(processes=20) as pool:
        results = pool.map(getdata,keys_list)
    
    # for k in keys_list:
    #     results.append(getdata(k, feats_data))
    
    with open(path, "wb") as f:
        pk.dump(results, f)
    print('finish')
    
def norm(data):
    """
    normalize venue, name and org, for build cleaned graph
    {
        id: str
        title: str
        authors:[{
            name
            org
        }]
        "abstract"
        "keywords"
        "venue"
        "year"
    }
    """
    venue = ''
    if data['venue']:
        venue = data["venue"].strip()
        venue = venue.lower()
        venue = re.sub(puncs, ' ', venue)
        venue = re.sub(r'\s{2,}', ' ', venue).strip()
        venue = venue.split(' ')
        venue = [word for word in venue if len(word) > 1]
        venue = [word for word in venue if word not in stopwords]
        venue = [word for word in venue if word not in stopwords_extend]
        venue = [word for word in venue if word not in stopwords_check]
        venue = ' '.join(venue)
    authors = []
    if data['authors']:
        for i in data['authors'][:50]:
            org = i['org']
            if org:
                org = org.strip()
                org = org.lower()
                org = re.sub(puncs, ' ', org)
                org = re.sub(r'\s{2,}', ' ', org).strip()
                org = org.split(' ')[:50]
                org = [word for word in org if len(word) > 1]
                org = [word for word in org if word not in stopwords]
                org = [word for word in org if word not in stopwords_extend]
                org = " ".join(org)
            authors.append({
                "name": i['name'],
                "org": org
            })
    if data['year']:
        pass
    else:
        data['year'] = 0
    return {
        'id': data['id'],
        'title': data['title'],
        'venue': venue,
        'year': data['year'],
        'authors': authors,
        'keywords': data['keywords'],
        'abstract': data['abstract']
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_author_dir', type=str, default='../IND-WhoIsWho/train_author.json')
    parser.add_argument('--test_author_dir', type=str, default='../IND-WhoIsWho/ind_test_author_filter_public.json')

    parser.add_argument('--pub_dir', type=str, default='../IND-WhoIsWho/pid_to_info_all.json')
    
    parser.add_argument('--title_embeddings_dir', type=str, default='../out_data/e5_instruct_title_data.pkl')
    parser.add_argument('--abstract_embeddings_dir', type=str, default='../out_data/e5_instruct_abstract_data.pkl')
    parser.add_argument('--venue_embeddings_dir', type=str, default='../out_data/e5_instruct_venue_data.pkl')
    
    parser.add_argument('--train_feats_dir', type=str, default='../out_data/e5_instruct_train.csv')
    parser.add_argument('--test_feats_dir', type=str, default='../out_data/e5_instruct_test.csv')
    
    parser.add_argument('--save_train_dir', type=str, default='../out_data/e5_instruct_graph_train.pkl')
    parser.add_argument('--save_test_dir', type=str, default='../out_data/e5_instruct_graph_test.pkl')
    
    args = parser.parse_args()  
    with open(args.pub_dir, "r", encoding = "utf-8") as f:
        papers_info = js.load(f)

    # clean pub, if needed
    with mp.Pool(processes=20) as pool:
        results = pool.map(norm,[value for _,value  in papers_info.items()])
    papers_info = {k:v for k,v in zip(papers_info.keys(),results)}
    print('done clean pubs')
    
    with open(args.title_embeddings_dir, "rb") as f:
        dic_title_embedding = pk.load(f)
    print('done loading title embeddings')

    with open(args.abstract_embeddings_dir, "rb") as f:
        dic_abstract_embedding = pk.load(f)
    print('done loading abstract embeddings')

    with open(args.venue_embeddings_dir, "rb") as f:
        dic_venue_embedding = pk.load(f)
    print('done loading venue embeddings')

    # with open('/mnt/f/workshop/competitions/2024-kdd-cup-whoiswho/dataset_test/bge_orgs_data.pkl', "rb") as f:
    #     dic_orgs_embedding = pk.load(f)
    # print('done loading orgs embeddings')

    #train
    with open(args.train_author_dir, "r", encoding="utf-8") as f:
        author_names = js.load(f)
    
    feats_data = pd.read_csv(args.train_feats_dir)
    
    build_dataset(args.save_train_dir)

    print(f'save path {args.save_train_dir}')

    # valid
    with open(args.test_author_dir, "r", encoding="utf-8") as f:
        author_names = js.load(f)
        
    feats_data = pd.read_csv(args.test_feats_dir)
    
    build_dataset(args.save_test_dir)

    print(f'save path {args.save_test_dir}')