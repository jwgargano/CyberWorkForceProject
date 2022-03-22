import re
import pickle
import pandas as pd
import glob
import ast
from nltk.corpus import stopwords


def remove_stopwords(text_list):
    """
    remove_stopwords

    params:
    text_list: list of raw texts to apply the function too

    returns:
    non_stop_text: a list of text with stop words removed
    """
    cachedStopWords = stopwords.words("english")
    non_stop_text = []
    for text in text_list:
        new_text = ' '.join([word for word in text.split() if word not in cachedStopWords]).strip()
        non_stop_text.append(new_text)
    return non_stop_text


def cleanup_text(text_list):
    """
    cleanup_text

    params:
    text_list: list of raw texts to apply the function too

    returns:
    text_list: a list of text that have been cleaned of words/symbols
    """
    cleaned = [re.sub(r"[^a-zA-Z0-9\'\&\,\.]+", ' ', k).strip() for k in text_list]
    return cleaned


def split_sents(text_list):
    """
    split_sents

    params:
    text_list: list of raw texts to apply the function too

    returns:
    text_list: a list of text that have been split on each new line
    """
    split_text = []
    for text in text_list:
        cleaned = []
        temp = re.split('\/n|\.', text)
        for s in temp:
            s = s.strip().lstrip()
            if len(s) > 1:
                cleaned.append(s)
        split_text.append(cleaned)
    # split_text =  [re.split('\/n|\.', text) for text in text_list]
    return split_text


def save_embeddings(embeddings, file):
    """
    save_embeddings
    params: 
    embeddings: np.array of embedding vectors
    file: file locaiton and name to save it too
    """
    if file[-6:] != 'pickle':
        file +='.pickle'
    with open(file, 'wb') as pkl:
        pickle.dump(embeddings, pkl)
    return 'pickle saved!'


def read_embeddings(file):
    """
    read_embeddings
    params: 
    file: file locaiton and name to read from. Must be pickle file

    returns:
    doc_embeddings: the saved vector array of embeddings
    """
    if file[-6:] != 'pickle':
        return 'ERROR: Must be a .pickle file'
    with open(file, 'rb') as pkl:
        doc_embedding = pickle.load(pkl)
    return doc_embedding


def rescore_group( dfs=[], folder=None, new_file=None):
    """
    rescore_group
    only use dfs OR folder not both
    params: 
    df: list of dataframes of the data category
    folder: location of files (will use all files in folder)
    new_file: name and path for new file results if you want to save it

    returns:
    new_df: new df with re-scored metrics/scores
    """
    df_all = pd.DataFrame()
    if folder and len(dfs) <1:
        for file in glob.glob(folder+'/*.csv'):
            df_temp = pd.read_csv(file,converters={'matches':ast.literal_eval, 'similar': ast.literal_eval, 
                                               'missing': ast.literal_eval})
            df_all = pd.concat([df_all, df_temp], sort=False)
    if len(dfs) >0 and not folder:
        dfs.append(df_all)
        df_all = pd.concat(dfs, sort=False)

    df_group = df_all.groupby('ksa', sort=False).agg({'matches': 'sum', 'similar': 'sum', 'missing':'sum'}).reset_index()

    sent_total = len(df_group['similar'][0]) +len(df_group['matches'][0]) +len(df_group['missing'][0])

    df_group['matched_score'] = df_group.apply(lambda x: len(x['matches'])/sent_total, axis=1)
    df_group['similar_score'] = df_group.apply(lambda x: len(x['similar'])/sent_total, axis=1)
    df_group['missing_score'] = df_group.apply(lambda x: len(x['missing'])/sent_total, axis=1)
    if new_file:
        df_group.to_csv(new_file, index=False)

    print('Successfully re-scored all data!')
    return df_group
    

def reduce_df_column(df,column, tgt_key):
    """
    reduce_df_column
    reduce the matches, similar, missing to just the index number
    deletes the old column
    params: 
    df: dataframe in question
    tgt_column: column name you want to reduce
    tgt_key: name of the idx you want to reduce by

    example: 
    final_df = reduce_df_column(final_df, 'matches', 'job_idx')
    final_df = reduce_df_column(final_df, 'similar', 'job_idx')
    final_df = reduce_df_column(final_df, 'missing', 'job_idx')

    returns:
    df: df with new column with a list of the job idxs only. Deletes old column
    """

    new_col = column+'_'+'idx'
    df[new_col] = df.apply(lambda x: [i[tgt_key] for i in x[column]], axis=1)
    df.drop(column, axis=1, inplace=True)
    return df
