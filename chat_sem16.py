
import json
import csv
import openai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

openai.api_key = "Your Own API-KEY"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def read_sem16(test_path):
    sents, topics, labels = [], [], []
    with open(test_path) as rfile:
        csv_reader = csv.DictReader(rfile)
        for idx, row in enumerate(csv_reader):
            tweet = row["tweet"]
            topic = " ".join(json.loads(row["topic"]))
            label = int(row["label"])
            sents.append(tweet)
            topics.append(topic)
            labels.append(label)
    print(f"{test_path} has {len(sents)} samples")
    return sents, topics, labels

def chat_gpt(prompt):
    completion = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content

def chatgpt_predict(sents, topics, labels, template_choice):
    total_result = []
    idx = 0
    for sent, topic, label in zip(sents, topics, labels):
        if template_choice == 1:
            prompt = f'What is the stance of the sentence: "{sent}" to the target "{topic}". select from "favor, against or neutral".'
        elif template_choice == 2:
            prompt = f'For the target "{topic}", what is the stance for the "{sent}"? select from "favor, against or neutral".'
        elif template_choice == 3:
            prompt = f'For the sentence "{sent}", what is the stance for the target "{topic}"? select from "favor, against or neutral".'

        chat_result = chat_gpt(prompt)
        temp = {}
        temp["sent"] = sent
        temp["topic"] = topic
        temp["label"] = label
        temp["chat_result"] = chat_result
        temp["prompt"] = prompt
        total_result.append(json.dumps(temp, ensure_ascii=False))
        idx += 1
        print(idx, temp)
    return total_result

def save_result(data, save_path):
    with open(save_path, "w") as wfile:
        for item in data:
            wfile.write(item + "\n")

if __name__ == '__main__':
    for template_choice in [1,2,3]:
        for domain in ["DT", "A", "CC", "FM", "HC", "LA"]:
            sents, topics, labels = read_sem16(f"./twitter_data_naacl/twitter_test{domain}_seenval/test.csv")
            total_result = chatgpt_predict(sents, topics, labels, template_choice)
            save_result(total_result, f"./results_prompt_{template_choice}/{domain}_result.txt")