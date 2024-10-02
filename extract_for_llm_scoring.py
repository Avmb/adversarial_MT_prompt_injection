#import bleu
import json 
import os, sys
from collections import OrderedDict
import torch 
import re 
import tqdm
import pandas as pd 

output_root = "./pia/"
lang_2_id = {"Czech":"cs","Ukrainian":"uk","English":"en","German":"de","Hindi":"hi","Icelandic":"is","Japanese":"ja","Chinese":"zh","Spanish":"es","Russian":"ru"}
id_2_lang = {id:lang for lang,id in lang_2_id.items()}
tasks = ["clean","direct","switch_zero_shot","switch_one_shot","switch_zero_shot_json_formatted","switch_one_shot_json_formatted"]

num_sentences_clean = 817
num_sentences_per_adv_subtask_per_src_lang = {"en": 817, "cs": 817*2, "ja": 817*2}
num_adv_subtasks = 5
num_sentences_per_src_lang = {src_lang: num_sentences_clean+num_adv_subtasks*n for src_lang, n in num_sentences_per_adv_subtask_per_src_lang.items()}

json_parsed_by_llm_filename = "./json_parsed_by_llm.json"
json_to_llm_to_parse_filename = "./json_to_llm_to_parse.txt"
data_to_score_filename = "to_score.jsonl"

def load_ans(src_jsonl_file:str, tgt_jsonl_file:str):
    srcs = []
    ref_ans_dict = {}
    with open(src_jsonl_file,"r") as in_fs:
        for i, line in enumerate(in_fs):
            parsed_line = json.loads(line.strip())
            src_content = parsed_line["messages"][1]["content"]
            srcs.append(src_content)
    with open(tgt_jsonl_file,"r") as in_fs:
        for jsl_str in in_fs:
            jsl = json.loads(jsl_str.strip())

            src_content = jsl[0]["messages"][1]["content"]
            tgt_content = jsl[1]["choices"][0]["message"]["content"]

            answers = []
            try:
                answer_dict = json.loads(src_content)
                answers.append(answer_dict["best_answer"])
                answers.extend(answer_dict["correct_answers"])
                answers.extend(answer_dict["incorrect_answers"])
            except Exception:
                pass
            try:
                answer_dict = json.loads(tgt_content)
                answers.append(answer_dict["best_answer"])
                answers.extend(answer_dict["correct_answers"])
                answers.extend(answer_dict["incorrect_answers"])
            except Exception:
                pass
            answers = list(OrderedDict.fromkeys(answers))
            ref_ans_dict[src_content] = answers

    # reorder
    ref_ans = []
    for src in srcs:
        ref_ans.append(ref_ans_dict[src])
    assert len(ref_ans) == num_sentences_clean
    return ref_ans

def eval_file(src:str,tgt:str):
    """
    Format the reference and translation output files such that they are split into 6 subtasks. 
    save the results to a file 
    """
    res = []
    
    output_dir = os.path.join(output_root,src + "-" + tgt)
    ref_src = list(map(lambda x:re.sub(r"\n","",x),open(f"prompt_injection_attack_test_suite/{id_2_lang[src]}_{id_2_lang[tgt]}/test_pia.src.txt","r").readlines()))
    ref_tgt = list(map(lambda x:re.sub(r"\n","",x),open(f"prompt_injection_attack_test_suite/{id_2_lang[src]}_{id_2_lang[tgt]}/test_pia.tgt.txt","r").readlines())) 
    assert len(ref_src) == len(ref_tgt) == num_sentences_per_src_lang[src]


    ans_tgt = load_ans(f"TruthfulQA_to_openai_{id_2_lang[tgt]}.jsonl", f"TruthfulQA_from_openai_{id_2_lang[tgt]}.jsonl")
  
    for path in tqdm.tqdm(os.listdir(output_dir)):
        system_name = path.split(".")[0]
        path = os.path.join(output_dir,path)
        tr_out = list(map(lambda x:re.sub(r"\n","",x),open(path,"r").readlines()))
        if len(tr_out) != num_sentences_per_src_lang[src]:
            print(f"error occured at: {system_name}, incorrect number of sentences")
            continue
        for i,task in enumerate(tasks):
            n = num_sentences_clean if i == 0 else num_sentences_per_adv_subtask_per_src_lang[src]
            expected_json = (i == 4) or (i == 5) # Last two tasks use JSON
            tr_out_task = tr_out[i*n:(i+1)*n]
            ref_tgt_task = ref_tgt[i*n:(i+1)*n]
            ref_src_task = ref_src[i*n:(i+1)*n]
            if (src == "en") or (i == 0):
                try:
                    eval_res = eval(ref_tgt_task,tr_out_task,ref_src_task,ans_tgt, expected_json=expected_json)
                    eval_res = {"sentences": eval_res}
                    eval_res["task"] = task 
                    eval_res["system"] = system_name
                    eval_res["src_lang"] = src
                    eval_res["tgt_lang"] = tgt
                    res.append(eval_res)
                except Exception as e:
                    print(f"error occured at: {system_name}, {task}\n", file=sys.stderr)
                    print(e, flush=True, file=sys.stderr)
                    #raise
            else:
                # split odd (non-English src) and even (English src)
                tr_out_task_non_en = tr_out_task[::2]
                ref_tgt_task_non_en = ref_tgt_task[::2]
                ref_src_task_non_en = ref_src_task[::2]
                tr_out_task_en = tr_out_task[1::2]
                ref_tgt_task_en = ref_tgt_task[1::2]
                ref_src_task_en = ref_src_task[1::2]
                try:
                    eval_res = eval(ref_tgt_task_non_en, tr_out_task_non_en, ref_src_task_non_en, ans_tgt, expected_json=expected_json)
                    eval_res = {"sentences": eval_res}
                    eval_res["task"] = task+"_src_"+src
                    eval_res["system"] = system_name
                    eval_res["src_lang"] = src
                    eval_res["tgt_lang"] = tgt
                    res.append(eval_res)
                    eval_res = eval(ref_tgt_task_en,tr_out_task_en,ref_src_task_en,ans_tgt, expected_json=expected_json)
                    eval_res = {"sentences": eval_res}
                    eval_res["task"] = task+"_src_en"
                    eval_res["system"] = system_name
                    eval_res["src_lang"] = src
                    eval_res["tgt_lang"] = tgt
                    res.append(eval_res)
                except Exception as e:
                    print(f"error occured at: {system_name}, {task}\n", file=sys.stderr)
                    print(e, flush=True, file=sys.stderr)
                    #raise


    res.sort(key=lambda x:x["system"])
    print(json.dumps(res), file=data_to_score_filename_out_fs)
    return res
        
js_extract_pattern = re.compile(r'\{([^{}]*)\}')

def extract_from_json(sent):
    # If the string ends with a JSON dict, extract the string value in the "input" attribute, or the longest if there is no "input" attribute
    global json_to_llm_to_parse_out_fs
    
    if sent in json_parsed_by_llm:
        return json_parsed_by_llm[sent]
    
    substrings = re.findall(js_extract_pattern, sent)

    if len(substrings) == 0:
        print(sent, file=json_to_llm_to_parse_out_fs)
        return sent
    last_substring = substrings[-1]
    
    if '{'+last_substring+'}' in json_parsed_by_llm:
        return json_parsed_by_llm['{'+last_substring+'}']
        
    try:
        parsed = json.loads('{'+last_substring+'}')
    except json.JSONDecodeError:
        print('{'+last_substring+'}', file=json_to_llm_to_parse_out_fs)
        return last_substring
    if 'input' in parsed:
        return str(parsed['input'])
    print('{'+last_substring+'}', file=json_to_llm_to_parse_out_fs)
    parsed_values = [str(x) for x in parsed.values()]
    if len(parsed_values) == 0:
        return last_substring
    return max(parsed_values, key=len)

def eval(ref:list[str],hyp:list[str],src:list[str],ref_ans:list[list[str]], expected_json: bool=False):
    global data_to_score_filename_out_fs
    
    if not expected_json:
        extracted_answers = hyp
        extracted_references = ref
        extracted_questions = src
    else:
        extracted_answers = []
        for h in hyp:
            extracted_answers.append(extract_from_json(h))
        extracted_references = []
        for r in ref:
            extracted_references.append(extract_from_json(r))
        extracted_questions = []
        for q in src:
            extracted_questions.append(extract_from_json(q))

    #res["corpus_comet"] = comet.compute(predictions=hyp,references=ref,sources=src,gpus=1 if device=="cuda" else 0)['mean_score']


    rv_acc = []
    for i, sent in enumerate(hyp):
        extracted_ans = extracted_answers[i]
        #ref_sent = ref[i]
        ref_sent = extracted_references[i]
        ref_ans_sentences = ref_ans[i]
        ref_src = extracted_questions[i]
        sent_dict = {"src": ref_src, "transl": extracted_ans, "ref_transl": ref_sent, "ref_answers": ref_ans_sentences}
        rv_acc.append(sent_dict)
    
    return rv_acc

if __name__ == "__main__":
    with open(json_parsed_by_llm_filename, "r") as in_fs:
        json_parsed_by_llm = json.load(in_fs)
    json_to_llm_to_parse_out_fs = open(json_to_llm_to_parse_filename, "w")
    data_to_score_filename_out_fs = open(data_to_score_filename, "w")
    #"Czech":"cs","Ukrainian":"uk","English":"en","German":"de","Hindi":"hi","Icelandic":"is","Japanese":"ja","Chinese":"zh","Spanish":"es","Russian":"ru"
    lang = ["cs", "uk", "de", "hi", "is", "ja", "zh", "es", "ru"]
    for l in lang:
        print("en-"+l, flush=True)
        eval_file("en",l)
        print("done.")

    # non-English src
    print("cs-uk", flush=True)
    eval_file("cs","uk")
    print("done.")
    print("ja-zh", flush=True)
    eval_file("ja","zh")
    print("done.")
    print("All done.")
    json_to_llm_to_parse_out_fs.close()
    data_to_score_filename_out_fs.close()
