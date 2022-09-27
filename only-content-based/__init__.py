import logging
import numpy as np
import pandas as pd
import azure.functions as func
import json
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import time

def filter_old_article(input_blob,n_days):
    blob_bytes = input_blob.read()
    blob_stream = BytesIO(blob_bytes)
    blob_stream.seek(0)
    articles = pd.read_csv(blob_stream)
    
    max_ts = articles['created_at_ts'].max()
    _1_j_ts  =   86400000
    articles['published'] = round((max_ts - articles['created_at_ts'])/_1_j_ts,0)
    articles['published'] = articles['published'].astype('int')
    articles = articles.set_index('article_id')
    recent_articles = articles[articles['published']<n_days]
    return recent_articles

def recent_embedding(input_blob,l_id,recent_articles):
    blob_bytes = input_blob.read()
    blob_stream = BytesIO(blob_bytes)
    blob_stream.seek(0)
    emb = np.load(blob_stream)
    emb_df = pd.DataFrame(emb)
    recent_index = emb_df.index.isin(recent_articles.index)
    emb_recent = emb_df.iloc[recent_index]
    del emb_df
    last_article_emb = emb[l_id].reshape(1,-1)
    emb_recent = np.concatenate((last_article_emb,emb_recent),axis=0)

    return emb_recent

def calculate_content_based_reco(emb_r,corresp):
    dim_emb = emb_r.shape[0]
    best_co_sim = np.zeros((1,50),dtype='int32')
    score_co_sim = np.zeros((1,dim_emb),dtype='float16')

    reduce_emb = emb_r[0].reshape(1,-1)
    score_co_sim[0] = cosine_similarity(reduce_emb,emb_r).astype('float16')
    best_co_sim[0] = score_co_sim.argsort(axis=1)[...,::-1][...,:50]
    
    recommendation_list = {}
    for index,id_article in np.ndenumerate(best_co_sim[0,1:6]):
        recommendation_list[index[0]+1]={
            'article':corresp[id_article],
            'score':score_co_sim[0,id_article].astype(float)
            }
    return recommendation_list

def main(req: func.HttpRequest,
         metadata: func.InputStream,
         embedding: func.InputStream,
         ) -> func.HttpResponse:
    logging.info("Python HTTP trigger content based recommandation function processed a request.")

    report_time = {}
    u_id = int(req.route_params.get('u_id'))
    art_id = int(req.route_params.get('art_id'))
    
    # Get content-based recommandation list
    # Import of the reduce embedding matrix or co-sim dict for content-based

    t0 = time.time()
    recent_articles = filter_old_article(metadata,
                                         30,
                                        )
    corr_index = { i:ind for i,ind in enumerate(recent_articles.index)}
    dt = time.time() - t0
    report_time['load_recent_articles'] = dt

    t0 = time.time()
    emb_r = recent_embedding(embedding,
                             art_id,
                             recent_articles,
                            )
    dt = time.time() - t0
    report_time['load_embedding'] = dt

    t0 = time.time()
    best_article_CB = calculate_content_based_reco(emb_r,corr_index)
    dt = time.time() - t0
    report_time['compute_content_based'] = dt

    return func.HttpResponse(json.dumps({'cb':best_article_CB,'times':report_time}))
