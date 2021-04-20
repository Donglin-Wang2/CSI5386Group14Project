import os
import json
import pandas as pd

def gen_df(dataset, to_disk=None):
    year = '2015' if dataset == 'test' else '2014'
    
    V_PATH = f'./VQA/Images/{dataset}{year}/'
    Q_PATH = f'./VQA/Questions/v2_OpenEnded_mscoco_{dataset}{year}_questions.json'
    A_PATH = f'./VQA/Annotations/v2_mscoco_{dataset}{year}_annotations.json'
  
    id_tuples = []
    questions = {}
    train_data = []

    with open(A_PATH) as f:
        data = json.load(f)
        for annotation in data['annotations']:
            if annotation['answer_type'] == 'yes/no':
                id_tuples.append(
                    (annotation['image_id'], 
                    annotation['question_id'],  
                    annotation['multiple_choice_answer'])
                )
    
    with open(Q_PATH) as f:
        data = json.load(f)
        for question in data['questions']:
            questions[question['question_id']] = question['question']
    
    
    for id_tuple in id_tuples:
        question = questions[id_tuple[1]]
        img = f'{V_PATH}COCO_{dataset}{year}_{str(id_tuple[0]).zfill(12)}.jpg'
        train_data.append((img, question, id_tuple[-1]))

    df = pd.DataFrame(data=train_data, columns=['Image', 'Question', 'Answer'])
    df = df[(df.Answer == 'yes') | (df.Answer == 'no')]
    if to_disk:
        df.to_pickle(to_disk)
    
    return df