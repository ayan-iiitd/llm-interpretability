import json
import pandas as pd
import gc
import torch
from transformers import pipeline
import warnings
import logging

warnings.filterwarnings('ignore')

    
log_file = './logs/models_multi_runs.log'

logging.basicConfig(filename = log_file,
                    format = '%(asctime)s \t %(message)s',
                    level = logging.INFO)

with open(log_file, "w") as file_writer:
    file_writer.write("LOGS FOR ALL 100 RUNS FOR ALL MODELS\n\n")



def run_model(model_name):

    # to_log = "Starting\t" + model_name + "\t" + datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    to_log = "Starting\t" + model_name
    logging.info(to_log)
    
    with open('./data/betl2train_with_s_a_id.json', 'r') as file_reader:
        betl2train = json.load(file_reader)

    # with open('../hf_token.txt', 'r') as file_reader:
    #     hf_token = file_reader.readlines()[0]

    base_df = pd.read_csv('./results/0_base_data_no_result.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_df = pd.read_csv('./results/0_base_data_no_result.csv')
    result_df = pd.DataFrame()
    result_df['labels'] = base_df['label']
    
    pipe = pipeline(
        "text-generation",
        model = model_name,
        tokenizer = model_name,
        device = device,
        torch_dtype = torch.bfloat16)
    
    pipe.tokenizer.padding_side = "left"

    
    for it in range(1, 101):

        msgs = []

        print(model_name[model_name.find('/') + 1:].strip().lower() + "\tIteration: ", it)

        for s_a_id in betl2train:

            prompt = f'''Question: {betl2train[s_a_id]['q']}\n'''

            r_a = ""
            for i in range(len(betl2train[s_a_id]['r_a'])):
                r_a = r_a + f'''\nReference Answer {i + 1}: {betl2train[s_a_id]['r_a'][i]}\n'''

            prompt = prompt + r_a + f'''\nStudent Answer: {betl2train[s_a_id]['s_a']}\n\nWhat is the Student's grade?'''

            messages = [
                {"role": "system", "content": "You are an expert grader. You will be provided with a question, a set of reference answers which should be considered the golden standard, and a student's answer. Your task is to grade the student's answer as 'correct' or  'incorrect' based *only* on its semantic alignment with the reference answer. Output *EXACTLY ONLY ONLY ONE SINGLE WORD GRADE* that is either 'correct' or 'incorrect' and *NOTHING ELSE*."},
                {"role": "user", "content": prompt}
            ]

            msgs.append(messages)

            
        if 'llama' not in model_name:
        
            output = pipe(msgs,
                        do_sample = True,
                        max_new_tokens = 2,
                        return_full_text = False,
                        temperature =  0.1,
                        pad_token_id = pipe.tokenizer.eos_token_id,
                        batch_size = 64)
        
        else:

            output = pipe(msgs,
                        do_sample = True,
                        max_new_tokens = 2,
                        return_full_text = False,
                        pad_token_id = pipe.tokenizer.eos_token_id,
                        temperature =  0.1)
        
        clean_op = [op[0]["generated_text"].lower().strip() for op in output]
        result_df[f'result_iter_{str(it)}'] = clean_op

        to_log = "Finished Iteration\t" + str(it)
        logging.info(to_log)

        
    
    file_name = './results/' + model_name[model_name.find('/') + 1:].strip().lower() + '_multi_runs.csv'
    print('Saving: ' + file_name)
    result_df.to_csv(file_name, index = False)

    
    del pipe
    gc.collect()
    torch.cuda.empty_cache()