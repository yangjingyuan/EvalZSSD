
import json
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import random
import pandas as pd

sem16_label2stance = {0:"against", 1:"favor", 2:"neutral"}
sem16_stance2label = {"against":0, "favor":1, "neutral":2}

def read_data(data_path):
    data = []
    with open(data_path) as rfile:
        for line in rfile:
            sample = json.loads(line.strip())
            data.append(sample)

    return data

def post_process_label(chat_pred_str):
    chat_pred_str = chat_pred_str.lower()
    chat_pred_str = chat_pred_str.replace(".", "")
    if "against" in chat_pred_str:
        chat_pred_str = "against"
    elif "neutral" in chat_pred_str or "irrelevant" in chat_pred_str:
        chat_pred_str = "neutral"
    elif "favor" in chat_pred_str:
        chat_pred_str = "favor"
    pred = sem16_stance2label.get(chat_pred_str, 2)
    return pred

if __name__ == '__main__':
    print_debug_info = True
    debug_list = []
    for template_choice in [1,2,3,4,5]:
        print(f"Evaluate results_prompt_{template_choice}")
        score_list = []
        for domain in ["DT", "HC", "FM", "LA", "A", "CC"]:
            data = read_data(f"./results_prompt_{template_choice}/{domain}_result.txt")
            preds = []
            labels = []
            for sample in data:
                prompt = sample["prompt"]
                sent = sample["sent"]
                topic = sample["topic"]
                label_id = sample["label"]
                label_str = sem16_label2stance[label_id]
                chat_str = sample["chat_result"]
                pred_id = post_process_label(chat_str)
                preds.append(pred_id)
                labels.append(label_id)
                is_right = "N"
                if pred_id == label_id:
                    is_right = "Y"
                debug_list.append([sent, topic, chat_str, label_str, prompt, is_right])

            target_names = ['against', 'favor', 'neutral']
            cls_report = classification_report(labels, preds, target_names=target_names)
            f1_report = f1_score(labels, preds, labels=[0, 1], average='macro')
            #print(f"---------{domain}: {cls_report}---------")
            print(f"---------{domain}: {f1_report}----------------")
            score_list.append(f1_report)

        print(f"average f1_score is {sum(score_list)/len(score_list)}")
        if print_debug_info:
            random.shuffle(debug_list)
            debug_list = debug_list[:200]
            df = pd.DataFrame(debug_list, columns=['SENT', 'TOPIC', "PRED", "LABEL", "PROMPT", "IS_RIGHT"])
            df.to_excel(f"./output_sem16_prompt_{template_choice}.xlsx", index=False)