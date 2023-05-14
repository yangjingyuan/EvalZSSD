
import json
import csv
import openai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

openai.api_key = "you own key"

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

def chat_gpt_cot_1(sent, topic):
    prompt_1 = f'What is the stance of the sentence: "{sent}" to the target "{topic}" and explain why?'
    history = []
    output_1 = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt_1}
        ]
    )
    history.append(f"USER: {prompt_1}")
    history.append(f"ASSISTANT: {output_1.choices[0].message.content}")

    prompt_2 = f'According to the above content, select corresponding stance from "favor, against or neutral".'
    output_2 = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt_1},
            {"role": "assistant", "content": output_1.choices[0].message.content},
            {"role": "user", "content": prompt_2},
        ]
    )
    history.append(f"USER: {prompt_2}")
    history.append(f"ASSISTANT: {output_2.choices[0].message.content}")

    return "\n".join(history), output_2.choices[0].message.content

def chat_gpt_cot_2(sent, topic):
    prompt = f'''What is the stance of the sentence: "@realDonaldTrump: We have got to take our country back. It's time! Win it Mr. Trump #SemST" to the target "donald trump"？ Give me an explanation and select from "favor, against or neutral".
    Explanation:
    The sentence "@realDonaldTrump: We have got to take our country back. It's time! Win it Mr. Trump #SemST" expresses a positive stance towards Donald Trump. The mention of "@realDonaldTrump" indicates that the message is attributed to Donald Trump's Twitter account, suggesting that it aligns with his views. The use of phrases like "take our country back" and "Win it Mr. Trump" implies support for Trump and his agenda.
    
    Result:
    favor
    
    What is the stance of the sentence: "{sent}" to the target "{topic}"？? Give me an explanation and select from "favor, against or neutral".
    '''
    history = []
    output = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    history.append(f"USER: {prompt}")
    history.append(f"ASSISTANT: {output.choices[0].message.content}")

    return "\n".join(history), output.choices[0].message.content

def chat_gpt_cot_3(sent, topic):
    prompt = f'''
    What is the stance of the sentence: "@realDonaldTrump: We have got to take our country back. It's time! Win it Mr. Trump #SemST" to the target "donald trump"？ Give me an explanation and select from "favor, against or neutral".

    Explanation:
    The sentence "@realDonaldTrump: We have got to take our country back. It's time! Win it Mr. Trump #SemST" expresses a positive stance towards Donald Trump. The mention of "@realDonaldTrump" indicates that the message is attributed to Donald Trump's Twitter account, suggesting that it aligns with his views. The use of phrases like "take our country back" and "Win it Mr. Trump" implies support for Trump and his agenda.

    Result:
    favor
    
    
    What is the stance of the sentence: "H stands for Holding Back e-mails belonging to #WeThePeople #obstructingjustice #WakeUpAmerica #HillaryLiesMatter #UniteBIue #SemST" to the target "hillary clinton"? Give me an explanation and select from "favor, against or neutral".
    
    Explanation:
    Based on the given sentence, the stance towards Hillary Clinton is against. The sentence suggests that "H" stands for Holding Back emails belonging to #WeThePeople, #obstructingjustice, #WakeUpAmerica, #HillaryLiesMatter, #UniteBIue, and #SemST. The hashtags used in the sentence, such as #HillaryLiesMatter, imply a negative sentiment towards Hillary Clinton. Therefore, the stance of the sentence is against Hillary Clinton.
    
    Result:
    against
    
    
    What is the stance of the sentence: "@DavidAttenborough meets @BarackObama on #TVNZOne interview on his life and the effects of #SemST" to the target "climate change is a real concern"? Give me an explanation and select from "favor, against or neutral".
    
    Explanation:
    The given sentence does not directly express a stance towards the target "climate change is a real concern." The sentence states that "@DavidAttenborough meets @BarackObama on #TVNZOne interview on his life and the effects of #SemST." It mentions the interview between David Attenborough and Barack Obama, focusing on their discussion of "his life and the effects of #SemST." It does not provide a clear indication of whether the sentence is in favor of, against, or neutral towards the statement "climate change is a real concern." Therefore, the stance of the sentence towards the target is neutral.
    
    Result:
    neutral


    What is the stance of the sentence: "{sent}" to the target "{topic}"？? Give me an explanation and select from "favor, against or neutral".
    '''
    history = []
    output = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    history.append(f"USER: {prompt}")
    history.append(f"ASSISTANT: {output.choices[0].message.content}")

    return "\n".join(history), output.choices[0].message.content

def chatgpt_predict(sents, topics, labels, template_choice):
    total_result = []
    idx = 0
    for sent, topic, label in zip(sents, topics, labels):
        if template_choice == 1:
            prompt, chat_result = chat_gpt_cot_1(sent, topic)
        elif template_choice == 2:
            prompt, chat_result = chat_gpt_cot_2(sent, topic)
        elif template_choice == 3:
            prompt, chat_result = chat_gpt_cot_3(sent, topic)
        print(prompt)
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
    for template_choice in [1, 2, 3]:
        for domain in ["DT", "A", "CC", "FM", "HC", "LA"]:
            sents, topics, labels = read_sem16(f"./twitter_data_naacl/twitter_test{domain}_seenval/test.csv")
            total_result = chatgpt_predict(sents, topics, labels)
            save_result(total_result, f"./results_prompt_1_CoT_{template_choice}/{domain}_result.txt")