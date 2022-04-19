# CyberWorkForceProject
Team Github for DAEN 690-DL1 Cyber Workforce Project (Team Mo-Data-Mo-Problems)


## Project Description
This project aims to determine if the current cyber security knowledge, skills, abilities (KSAs) set by orgnizations like the National Institute of Standards and Technology (NIST) and the National Initiative for Cybersecurity Education (NICE) match the current public sector job market, private sector job market, and educational certification programs. To determine the matching, similar, or missing KSAs, our team utilized natural language processing techniques with BERT to calculate the cosine similarity between the KSAs and sentences in the data corpuses. In addition, our team used BERTopic to develop a topic model for each dataset and determine if the KSAs have any matching or similar topics.


## Getting Started
To install the required packages and pre-trained models, install the requirements.txt file in the scripts folder.
```
pip install -r scripts/requirements.txt
```
### Cosine Similarity
To calculate the cosine similarity for a given dataset and to determin the matching, similar, and missing KSAs, use the bert_similarity_template jupyter notebook located in the scripts folder.

#### Change the parameters to meet your local filepath or the filepath on the github.
```
# Params / Files to change 
input_file = '../data/cleaned_data/USAJobs.csv' # change to whatever file/filepath you are using
output_file = '../data/results/usa_jobs/similarity_match/batch2.csv' # change to your outpath
desc_column = 'Duties'
baseline_embedding_file = '../data/saved_embeddings/baseline.pickle' # Make None if you don't want to saev
sent_embedding_file = '../data/saved_embeddings/test_batch.pickle' #change to sentence embedding path
start_idx = 0 # file row to start at
end_idx = 100 # file row to end at
#how many jobs do you want to search / score against? make start_idx -1 if you want to use entire file
```
##### Next, update the baseline path and column parameter
```
# file for our cyber baseline
baseline_file = '../data/cleaned_data/KUKSAT_Baseline.csv'
ksa_col = 'KUKSAT'
```
##### Various helper functions have been created to standardize text cleanup and data pre-processing
```
# apply text cleanup functions to jobs and ksa base lists
jobs = helpers.cleanup_text(jobs)
ksas = helpers.cleanup_text(ksas)

# split jobs to sentence level
job_sent = helpers.split_sents(jobs)
```

##### Use BERT to encode the KSAs and dataset sentences
```
# gets vector embeddings for the ksa baselines
baseline_vecs = model.encode(ksas)

# encode all job sentences
all_job_vecs = [model.encode(sent) for sent in job_sent]
```

#### Loop through the sentence embeddings, calculate the cosine similartiy scores, and determine if they are matching, similar, or missing KSAs
Similarity thresholds were based on cognitive observations and include: Matching (x >= 0.6), Similar (0.6 < x > 0=4), Missing (x < 0.4)
```
# Try using each KSA and compare it to each sentence for each job description
# Find sentences that matches (x >= 0.6), similar matches ( 0.6< x >0.4), missing ( x < 0.4)
all_matches = []
all_similar =[]
all_missing =[]
# Loops through baseline vectors embeddings of the ksas
for idx, ksa in enumerate(baseline_vecs):
    print('ksa', idx, '_'*10)
    matched=[]
    similar=[]
    missing=[]
#   loops through all the job sentence vector embeddings
    for idx2, sent in enumerate(all_job_vecs):
        print('..job ', idx2)
#         evaluates the cosine similarity between the ksa and the job sentences
        val = cosine_similarity([ksa], sent)
    
#     Loops through the scores to determine if matched, similar, or missing
        for idx3, num in enumerate(val[0]):
#         created a temp dictionarty of key info and then appends that to the appropriate list
            temp = {}
#     if ksa text in sentence text then match too
            if num >=0.6 or (ksas[idx].lower().lstrip().strip() in job_sent[idx2][idx3].lower().lstrip().strip()):
                temp['job_idx'] = idx2
                temp['sentence_idx'] = idx3
                temp['ksa_idx'] = idx
                temp['sentence_text'] = job_sent[idx2][idx3]
                temp['ksa_text'] = ksas[idx]
                temp['sim_score'] = num
                matched.append(temp)
            elif num < 0.6 and num > 0.4:
                temp['job_idx'] = idx2
                temp['sentence_idx'] = idx3
                temp['ksa_idx'] = idx
                temp['sentence_text'] = job_sent[idx2][idx3]
                temp['ksa_text'] = ksas[idx]
                temp['sim_score'] = num
                similar.append(temp)
            else:
                temp['job_idx'] = idx2
                temp['sentence_idx'] = idx3
                temp['ksa_idx'] = idx
                temp['sentence_text'] = job_sent[idx2][idx3]
                temp['ksa_text'] = ksas[idx]
                temp['sim_score'] = num
                missing.append(temp)
    all_matches.append(matched)
    all_similar.append(similar)
    all_missing.append(missing)
    print('**\n**')
```

#### Calculate aggregate scores for each KSA
```
# calculates aggregate scores for matching, similar, missing
ksa_agg_matches = []
ksa_agg_similar = []
ksa_agg_missing = []
for idx, val in enumerate(baseline_vecs):
    print('ksa', idx, '_'*10)
    matched_score = len(all_matches[idx])/len(jobs)
    similar_score = len(all_similar[idx])/len(jobs)
    missing_score = 1 -(matched_score + similar_score)
    ksa_agg_matches.append(matched_score)
    ksa_agg_similar.append(similar_score)
    ksa_agg_missing.append(missing_score)
    print('Match Score ==', matched_score)
    print('Similar Score == ', similar_score)
    print('Missing Score ==', missing_score)
```

#### Save results to csv
```
# dataframes all results
final_df = pd.DataFrame({
    'ksa': ksas,
    'matches': all_matches,
    'similar': all_similar,
    'missing': all_missing,
    'matched_score': ksa_agg_matches,
    'similar_score': ksa_agg_similar,
    'missing_score': ksa_agg_missing
})
# save to output_file
final_df.to_csv(output_file, index=False)
```

## Contact Information and Team Members
Waseem Ashraf
Joseph Gargano
Laura Gibson
Austin Hembree
Shirinithi Thiruppathi
Joseph Ware
