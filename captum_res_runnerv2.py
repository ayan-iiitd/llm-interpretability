import json
from captum_res import captum_res
from rich import print as pp
# from time import time
import multiprocessing
import os

with open("./data/text_with_entities.json", 'r') as file_reader:
    data = json.load(file_reader)

med_entitites = list()
samples = list()
seq_attr = dict()

for k in data:

    if len(data[k]['entities']) == 1 and data[k]['entities'][0] not in med_entitites:
        samples.append(data[k])
        med_entitites.append(data[k]['entities'][0])
    
    if len(samples) == 10:
        break

def run_inference_worker_entry(sample, idx):
    from captum_res import captum_res
    res = captum_res(sample)
    seq_attr[idx] = res

    # print("Recieved")


try:
    multiprocessing.set_start_method('spawn', force = True)

    for idx, sample in enumerate(samples):

        process = multiprocessing.Process(
            target = run_inference_worker_entry, 
            args = (sample, idx)
            )
        
        process.start()
        process.join() # Wait for the current inference process to complete

        if process.exitcode != 0:
            print(f"Error processing sample - {idx}. Worker exited with code {process.exitcode}.")
            # Decide to stop or continue
            # break

        print(f"Finished processing sample {idx}. GPU memory should be fully released by now.")
        print("-" * 50)

    print("All samples processed.")

    with open('./for_team/samples_captum_llama.json', 'w') as file_writer:
        json.dump(seq_attr, file_writer, indent = 4)



except RuntimeError:
        # Might already be set, or not the first time calling it.
        # On win, 'spawn' is the default
        pass

    
    # seq_attr[idx] = captum_res(sample)
    # run_inference_worker_entry(sample)
    # print(f"compelted sample {idx}")
    # time.sleep(10)

# pp(seq_attr)