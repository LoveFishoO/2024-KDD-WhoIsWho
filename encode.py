import json
import torch
import gc
import os
import voyageai
import pickle as pk
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import FlagModel

parser = argparse.ArgumentParser()
parser.add_argument("--api_key", type=str, help='The api key of voyageai')
args = parser.parse_args()

with open("./IND-WhoIsWho/pid_to_info_all.json", "r", encoding="utf-8") as f:
    papers = json.load(f)
    
e5_instruct_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def _e5_instruct_encode(texts):
    task = 'Given a paper title query, retrieve relevant passages to query'
    queries = [
        get_detailed_instruct(task, text) for text in texts
        
    ]

    embeddings = e5_instruct_model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)

    return embeddings

def e5_instruct_encode(target='title'):
    batch_size = 5000
    
    dic_paper_embedding ={}
    paper_list = [[key, value] for key,value in papers.items()]
    for ii in tqdm(range(0, len(paper_list), batch_size), total=len(paper_list)//batch_size):
        batch_papers = paper_list[ii: ii + batch_size]
        texts = [paper[1][target] for paper in batch_papers]
        
        embeddings = _encode(texts)
        
        tt = 0
        for jj in range(ii, ii+len(batch_papers)):
            paper_id = paper_list[jj][0]
            paper_vec = embeddings[tt]
            tt+=1
            dic_paper_embedding[paper_id] = paper_vec.to('cpu').detach().numpy()
    with open('./out_data' + f'e5_instruct_title_{target}_data.pkl', "wb") as f:
        pk.dump(dic_paper_embedding, f)

KEY = args.api_key
print(f'Key: {KEY}')
vo = voyageai.Client(api_key=KEY)

def _vo_encode(texts):
    result = vo.embed(texts, model="voyage-large-2-instruct")
    return result.embeddings

def vo_encode(target='title'):
    batch_size = 128
    
    dic_paper_embedding ={}
    paper_list = [[key, value] for key,value in papers.items()]
    for ii in tqdm(range(0, len(paper_list), batch_size), total=len(paper_list)//batch_size):
        batch_papers = paper_list[ii: ii + batch_size]
        texts = [paper[1][target] if paper[1][target] != None else '' for paper in batch_papers]
        
        embeddings = _encode(texts)
        
        tt = 0
        for jj in range(ii, ii+len(batch_papers)):
            paper_id = paper_list[jj][0]
            paper_vec = embeddings[tt]
            tt+=1
            dic_paper_embedding[paper_id] = paper_vec
    with open('./out_data' + f'voyage_{target}_data.pkl', "wb") as f:
        pk.dump(dic_paper_embedding, f)


bge = FlagModel('BAAI/bge-m3', use_fp16=True)

def org_encode():
    batch_size = 5000
    
    orgs = set()

    for _,value in papers.items():
        authors = value['authors']
        for i in authors:
            # if i['org'] != '' and i['org'] != ' ':
            orgs.add(i['org'])

    orgs = list(orgs)

    orgs_embedding = {}

    for ii in tqdm(range(0, len(orgs), batch_size), total=len(orgs)//batch_size):
        
        batch_orgs = orgs[ii: ii + batch_size]
        
        with torch.no_grad():
            orgs_embeddings = bge.encode(batch_orgs)
        
        gc.collect()

        tt = 0
        for jj in range(ii, ii+len(batch_orgs)):
            dic_paper_embedding = {}
            orgs_id = orgs[jj]
            orgs_vec = orgs_embeddings[tt]
            tt+=1
            orgs_embedding[orgs_id] = orgs_vec

    with open(f'./out_data/bge_orgs_data.pkl', "wb") as f:
        pk.dump(orgs_embedding, f)

device = torch.device('cuda:0')
e5_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
e5_model = AutoModel.from_pretrained('intfloat/multilingual-e5-large', torch_dtype=torch.float16).to(device)


papers_list = [[key, value] for key,value in papers.items()]

def average_pool(last_hidden_states,
                 attention_mask):
    attention_mask=attention_mask.to(device)
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def e5_encode(target='title'):
    batch_size = 5000

    dic_paper_embedding = {}
    
    for ii in tqdm(range(0, len(papers_list), batch_size), total=len(papers_list)//batch_size):
    
        batch_papers = papers_list[ii: ii + batch_size]
        texts = [paper[1][target] if paper[1][target] != None else '' for paper in batch_papers]
        
        batch_dict = e5_tokenizer(texts, max_length=50, padding=True, truncation=True, return_tensors='pt')

        inputs = {key: value.to(device) for key, value in batch_dict.items()}
        with torch.no_grad():
            outputs = e5_model(**inputs)

        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        del outputs
        del inputs
        del batch_dict
        gc.collect()

        tt = 0
        for jj in range(ii, ii+len(batch_papers)):
            paper_id = papers_list[jj][0]
            paper_vec = embeddings[tt]
            tt+=1
            dic_paper_embedding[paper_id] = paper_vec.to('cpu').detach().numpy()

    with open(f'./out_data/e5_{target}_data.pkl', "wb") as f:
        pk.dump(dic_paper_embedding, f)
    
if __name__ == '__main__':
    
    e5_instruct_encode('title')
    e5_instruct_encode('abstract')
    e5_instruct_encode('venue')
    
    vo_encode('title')
    vo_encode('abstract')
    vo_encode('venue')
    
    org_encode()
    
    # e5_encode('title')
    # e5_encode('abstruct')
    # e5_encode('venue')