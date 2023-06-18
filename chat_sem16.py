
import os
import json
import threading
import csv
import openai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

openai.api_key = "your own apikey"

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
            prompt = f"tweet: {sent}\n\ntopic: {topic}\n\n" + \
                     f"Give you the above tweet and topic, determine the stance of the tweet towards the topic, the stance can be favor, against, neutral or irrelevant."
        elif template_choice == 2:
            prompt = f"What is the stance of the tweet '''{sent}''' to the topic '''{topic}'''?" + \
                       f"\n\n" + f"select the stance from favor, against, neutral or irrelevant."
        elif template_choice == 3:
            prompt = f"Give you a tweet and a topic, determine the stance of the tweet towards the topic, the stance can be favor, against, neutral or irrelevant." + \
                       f"\n\ntweet: {sent}\n\ntopic: {topic}"
        elif template_choice == 4:
            prompt = f"Provide a tweet and a topic, assess what attitude the tweet holds towards the topic; the stance could be favor, against, neutral or irrelevant." + \
                     f"\n\ntweet: {sent}\n\ntopic: {topic}"
        elif template_choice == 5:
            prompt = f"Given a topic and a tweet related to it, identify the tweet’s attitude towards the topic as either favor, against, neutral or irrelevant." + \
                     f"\n\ntweet: {sent}\n\ntopic: {topic}"
        elif template_choice == 6:
            prompt = f"tweet: {sent}\n\ntopic: {topic}\n\n" + \
                     f"Provide you the above tweet and topic, assess what attitude the tweet holds towards the topic; the stance could be favor, against, neutral or irrelevant."
        elif template_choice == 7:
            prompt = f"tweet: {sent}\n\ntopic: {topic}\n\n" + \
                     f"Given the above topic and a tweet related to it, identify the tweet’s attitude towards the topic as either favor, against, neutral or irrelevant."
        elif template_choice == 8:
            prompt = f"Choose the attitude of the tweet '''{sent}''' towards the topic '''{topic}'''. " + \
                     f"\n\n" + f"select it from favor, against, neutral or irrelevant."
        elif template_choice == 9:
            prompt = f"How does the tweet '''{sent}''' relate to the topic '''{topic}'''? " + \
                     f"\n\n" + f"pick the stance that matches: favor, against, neutral or irrelevant."

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

def run_thread(domain, template_choice):
    sents, topics, labels = read_sem16(f"./twitter_data_naacl/twitter_test{domain}_seenval/test.csv")
    total_result = chatgpt_predict(sents, topics, labels, template_choice)
    save_result(total_result, f"./results_prompt_{template_choice}/{domain}_result.txt")

if __name__ == '__main__':
    for template_choice in [1,2,3,4,5,6,7,8,9]:
        if not os.path.exists(f"./results_prompt_{template_choice}"):
            os.makedirs(f"./results_prompt_{template_choice}")

        thread_list = []
        for domain in ["DT", "A", "CC", "FM", "HC", "LA"]:
            t = threading.Thread(target=run_thread, args=(domain, template_choice,))
            thread_list.append(t)

        for t in thread_list:
            t.start()
        for t in thread_list:
            t.join()