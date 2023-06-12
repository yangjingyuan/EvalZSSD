
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
    return openai.Completion.create(**kwargs)

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

def gpt3(sent, topic):
    prompt = f"Give you a tweet and a topic, determine the stance of the tweet towards the topic, the stance can be favor, against, neutral or irrelevant." + \
             f"\n\ntweet: {sent}\n\ntopic: {topic}"

    output = completion_with_backoff(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=256
    )

    return prompt, output.choices[0].text

def gpt3_few_shot_cot(sent, topic):
    few_shot_examples = {
    "climate change is a real concern": '''tweet: Wearing a sweater at the end of June! #itfeelslikespring #SemST

topic: climate change is a real concern

rational: This is an interesting task. I would say the stance of the tweet towards the topic is favor, because the tweet implies that the weather is unusual for the season and uses hashtags that suggest dissatisfaction with the situation. This could indicate that the tweeter is concerned about climate change and its effects on the environment.

stance: favor''',
    "atheism": '''tweet: @AJEnglish and, who left real #spiritual and #material #devastation of the countries where is grafted,and installed. #Ummah,#Hadith #SemST

topic: atheism

rational: because the tweet uses words that suggest a religious perspective and criticizes a media outlet that may be seen as secular or liberal. This could indicate that the tweeter is opposed to atheism and its influence on society.

stance: against''',
    "donald trump": '''tweet: @SCOTUScare penned by @ChiefJusticeJR  ends rule of law and freedom. What law will SCOTUS  edit next? @RedNationRising #SemST

topic: donald trump

rational: Supreme Court’s decision on health care and its implications for the rule of law and freedom. This topic has nothing to do with Donald Trump or his policies, so the tweet is not expressing any opinion for or against him.

stance: irrelevant''',
    "feminist movement":'''tweet: @POBREClTA @JustLove_Katie @WillBrianna "running your dick sucker" yes because THAT is the purpose of your mouth lol #SemST

topic: feminist movement

rational: The tweet uses a vulgar and derogatory term for the mouth and mocks the person for using it to express their opinion. This could indicate that the tweeter is against the feminist movement and its values of equality and empowerment for women.

stance: against''',
    "legalization of abortion": '''tweet: Anti-choice laws are entirely men controlling women  #SemST

topic: legalization of abortion

rational: The tweet expresses a strong opinion against anti-choice laws, which are laws that restrict or prohibit abortion. This could indicate that the tweeter is in favor of legalization of abortion and women’s right to choose.

stance: favor''',
    "hillary clinton": '''tweet: @Upworthy @HillaryClinton It should be illegal for an employer to discriminate against their workers. #EqualityForAll #SemST

topic: hillary clinton

rational: The tweet is quoting a statement made by Hillary Clinton and uses hashtags that support her campaign slogan and values. This could indicate that the tweeter is in favor of Hillary Clinton and her policies on workers’ rights and equality.

stance: favor'''
    }

    prompt = f"Give you a tweet and a topic, determine the stance of the tweet towards the topic, the stance can be favor, against, neutral or irrelevant. " + \
        "\n——————————————————————\n" + \
        "——————————————————————\n".join([v+"\n" for k, v in few_shot_examples.items() if k != topic]) + \
        f'''\n——————————————————————\ntweet: {sent}

topic: {topic}

rational: '''

    output = completion_with_backoff(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=256
    )

    return prompt, output.choices[0].text

def chatgpt_predict(sents, topics, labels, template_choice):
    total_result = []
    idx = 0
    for sent, topic, label in zip(sents, topics, labels):
        if template_choice == 1:
            prompt, chat_result = gpt3(sent, topic)
        elif template_choice == 2:
            prompt, chat_result = gpt3_few_shot_cot(sent, topic)
        print("-" * 20)
        print(prompt)
        print("-" * 20)
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

def thread_run(domain, template_choice):
    sents, topics, labels = read_sem16(f"./twitter_data_naacl/twitter_test{domain}_seenval/test.csv")
    total_result = chatgpt_predict(sents, topics, labels, template_choice)
    save_result(total_result, f"./gpt3_results_prompt_CoT_{template_choice}/{domain}_result.txt")


if __name__ == '__main__':
    for template_choice in [1,2]:
        if not os.path.exists(f"./gpt3_results_prompt_CoT_{template_choice}"):
            os.makedirs(f"./gpt3_results_prompt_CoT_{template_choice}")

        thread_list = []
        for domain in ["DT", "A", "CC", "FM", "HC", "LA"]:
            t = threading.Thread(target=thread_run, args=(domain, template_choice,))
            thread_list.append(t)

        for t in thread_list:
            t.start()
        for t in thread_list:
            t.join()