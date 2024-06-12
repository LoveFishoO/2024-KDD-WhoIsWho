import json
sub = {}

with open('../output/voyage_lgb.json') as f:
    lgb_sub1 = json.load(f)

with open('../output/e5_instruct_lgb.json') as f:
    lgb_sub2 = json.load(f)

with open('../output/e5_instruct_gcn.json') as f:
    gcn_sub1 = json.load(f)

with open('../output/e5_instruct_embed_voyage_feats_gcn.json') as f:
    gcn_sub2 = json.load(f)

with open('../output/voyage_gcn.json') as f:
    gcn_sub3 = json.load(f)
    
with open('../output/voyage_embed_e5_instruct_feats_gcn.json') as f:
    gcn_sub4 = json.load(f)
    
    
for k1, dict1 in gcn_sub1.items():
    tmp ={}
    for k2, v in dict1.items():
        
        tmp[k2] = 0.3 * (0.25 * gcn_sub1[k1][k2] + 0.25 * gcn_sub2[k1][k2] + 0.25 * gcn_sub3[k1][k2] + 0.25 * gcn_sub4[k1][k2]) + 0.7 * (0.55 * lgb_sub1[k1][k2] + 0.45 * lgb_sub2[k1][k2]) 
    
    sub[k1] = tmp
         
with open(f'./output/essemble_2lgb_4gcn.json', 'w', encoding='utf-8') as f:
    json.dump(sub, f, ensure_ascii=False, indent=4)
