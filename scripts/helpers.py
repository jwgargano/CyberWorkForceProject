import re
import pickle

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