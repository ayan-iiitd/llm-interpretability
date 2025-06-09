import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch 
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
from rich import print as pp
import numpy as np


def decode_token_with_logitlens(model, device, tokenizer, input, tokens_to_gen=None): 
    '''
    outputs a dictionary with {'decoded_tokens': [num_layers, seq_len], 'decoded_logits': [num_layers, seq_len]}
    '''
    inputs = tokenizer(input, return_tensors="pt").to(device)
    
    text = tokenizer.decode(inputs['input_ids'][0])
    # run the loop to generate new tokens after input, append to input and decode 
    if tokens_to_gen != None: 
        # generate new tokens all at once; then append them to input, then logitlens them all
        output = model.generate(
            inputs['input_ids'],
            do_sample=True,
            top_p=0.95,
            temperature=0.001,
            top_k=0,
            max_new_tokens=tokens_to_gen,
            )
        new_token = tokenizer.decode(output[0][-tokens_to_gen:])
        text += new_token
        inputs = tokenizer(text, return_tensors="pt").to(device)
    text_tokens = [tokenizer.decode(id) for id in inputs['input_ids'][0]]
    
    # apply decoder lens
    classifier_head = model.lm_head # Linear(in_features=3072, out_features=32064, bias=False)

    hidden_states = model(**inputs, output_hidden_states = True).hidden_states
    decoded_intermediate_token = {}
    decoded_intermediate_logit = {}
    with torch.no_grad():
        for layer_id in range(len(hidden_states)): 
            hidden_state = hidden_states[layer_id]
            decoded_value = classifier_head(hidden_state) # [batch, seq_len, vocab_size]
            # get probabilities
            decoded_values = torch.nn.functional.softmax(decoded_value, dim=-1)
            # take max element
            argmax = torch.argmax(decoded_values, dim=-1)[0] # select first element in batch
            # decode all tokens
            decoded_token = [tokenizer.decode(el) for el in argmax]
            decoded_logit = [decoded_values[0, it, argmax[it]].item() for it in range(len(argmax))] # list of layers, per layer the sequence_length
            decoded_intermediate_token[layer_id] = decoded_token
            decoded_intermediate_logit[layer_id] = decoded_logit

    tokens = list(decoded_intermediate_token.values()) # [num_layers, seq_len]
    logits = list(decoded_intermediate_logit.values()) # [num_layers, seq_len]
    return {'text_tokens':text_tokens, 'decoded_tokens': tokens, 'decoded_logits': logits}

with open('/mnt/c/ayan/work/p2/data/beetle_2_train_formatted.json', 'r') as file_reader:
    betl2train = json.load(file_reader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'microsoft/Phi-4-mini-instruct'

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map = device, 
    torch_dtype = "auto", 
    trust_remote_code = True, 
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier_head = model.lm_head


gr_tkn_ids = []
grades = ['correct', 'Correct', 'Incorrect', 'incorrect']
for gr in grades:
    gr_tkn_ids.append(tokenizer.vocab[gr])



for qid in betl2train:

    prompt = f'''You are an expert grader. You will be provided with a question, a set of reference answers which should be considered the golden standard, and a student's answer. Your task is to grade the student's answer as 'correct' or  'incorrect' based *only* on its semantic alignment with the reference answer. Output *only* the single word grade ('correct' or 'incorrect') and nothing else.\n\nQuestion: {betl2train[qid]['q']}\n'''

    r_a = ""
    for i in range(len(betl2train[qid]['r_a'])):
        r_a = r_a + f'''\nReference Answer {i + 1}: {betl2train[qid]['r_a'][i]}\n'''
    
    prompt = prompt + r_a
    
    for s_a in betl2train[qid]['s_a']:
        
        eval_prompt = prompt + f'''\nStudent Answer: {s_a['answer']}\n\nGrade: '''
        s_a['accuracy'] = s_a['accuracy'].lower()
        correct_grade = s_a['accuracy']

        text_tokenized = tokenizer(eval_prompt, return_tensors = 'pt').to(device)
        tokens_to_gen = 2

        output = model.generate(
            text_tokenized['input_ids'],
            do_sample = True,
            top_p = 0.95,
            temperature = 0.001,
            top_k = 0,
            max_new_tokens = tokens_to_gen)
        
        text_decoded = tokenizer.decode(output[0])
        
        # pp(f'\noutput\n\n{output}')
        # pp(f'\ntext decoded\n\n{text_decoded}')
        pp(f'\nmodel predcition: {text_decoded[len(eval_prompt):].strip()}')
        
        txt_to_analyse = tokenizer(text_decoded, return_tensors="pt").to(device)
        hidden_states = model(**txt_to_analyse, output_hidden_states = True).hidden_states

        with torch.no_grad():
            
            for layer_id in range(len(hidden_states)):
                
                hidden_state = hidden_states[layer_id]
                decoded_value = classifier_head(hidden_state)
                decoded_values = torch.nn.functional.softmax(decoded_value, dim = -1)

                for j in range(tokens_to_gen):
                    
                    decoded_values_np = decoded_values[0][j - tokens_to_gen].detach().cpu().float().numpy()
                    sort_index = np.argsort(decoded_values_np)[::-1]

                    tkn_ranks = {'generated_token': j + 1,
                                 'layer_id': layer_id}
                    for idx, gr_tkn_id in enumerate(gr_tkn_ids):
                        key = '"' + grades[idx] + '" rank'
                        tkn_ranks[key] = np.where(sort_index == gr_tkn_id)[0][0]
                    
                    pp(tkn_ranks)






        # pp(f'\n\n----------------------------\nNumber of input tokens: {len(text_tokenized)}')
        # output = model.generate(text_tokenized,
        #                         num_beams = 4,
        #                         do_sample = True,
        #                         max_new_tokens = 2)
        # text_decoded = tokenizer.decode(output[0])
        # pp(f'\n\nFULL OUTPUT\n\n{text_decoded}')
        # pp(f'\nMODEL PREDICTION\n\n{text_decoded[len(eval_prompt):].strip()}')
        # pp(f'\nLABEL: {correct_grade}')
        # s_a['phi-4-prediction'] = text_decoded[len(eval_prompt):].strip().lower()
        # pp(s_a)

        # print(eval_prompt)

        # grades = ['correct', 'incorrect']
        # grade_combos = []
        
        # for k in grades:
        #     grade_combos += list(map(''.join, itertools.product(*zip(k.upper(), k.lower()))))
        
        # # print(grade_combos)

        # with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        #     utils.test_prompt(eval_prompt, ['incorrect', 'Incorrect', 'correct', 'Correct'], gemma_2_2b)
        #     op = buf.getvalue()
        break
    break



# pp(model)

# text = "1:sunshine, 2:apple, 3:koala, 2:apple, 1:sunshine, 3:koala, 1:"
# text_tokenized = tokenizer.encode(text, return_tensors = 'pt').to(device)


# output = model.generate(text_tokenized, num_beams=4, do_sample=True, max_new_tokens = 3)
# text_decoded = tokenizer.decode(output[0])
# pp(text_decoded)

# dict_output = decode_token_with_logitlens(model, device, tokenizer, text, tokens_to_gen=3)
# decoded_tokens = dict_output['decoded_tokens']
# decoded_logits = dict_output['decoded_logits']
# text_tokens = dict_output['text_tokens']
# pp(len(text_tokens), text_tokens)

# for element in range(len(decoded_tokens)):
#     pp(decoded_tokens[element])

# to_viz = (35,42)
# tokens_viz = [tok[to_viz[0]:to_viz[1]] for tok in decoded_tokens] # [num_layers, tokens_to_viz]
# logits_viz = np.array([log[to_viz[0]:to_viz[1]] for log in decoded_logits]) # [num_layers, tokens_to_viz]
# a = [it for it in range(to_viz[0],to_viz[1])]
# b = [tok for tok in text_tokens[to_viz[0]:to_viz[1]]]
# col_labels = [str(a_)+': '+b_ for a_, b_ in zip(a, b)]
# norm = plt.Normalize(logits_viz.min()-1, logits_viz.max()+1)
# colours = plt.cm.cool(norm(logits_viz))
# fig, ax = plt.subplots(figsize=(20,5), dpi=1000)
# # hide axes
# fig.patch.set_visible(False)
# ax.axis('off')
# ax.axis('tight')
# img = plt.imshow(norm(logits_viz), cmap="cool")
# plt.colorbar()
# img.set_visible(False)
# ax.table(cellText=tokens_viz, rowLabels=[lay for lay in range(len(decoded_tokens))], colLabels=col_labels, colWidths = [0.2]*logits_viz.shape[1], loc='center', cellColours=img.to_rgba(norm(logits_viz))) 
# plt.show()