import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

# import openai  # 注释掉
# import google.generativeai as genai
from imagenet_prompts import emotion6
from tqdm import tqdm
from pathlib import Path
import json
import openai
import json


openai.api_key = "sk-SPj7hvSJkF57LAliKTFPi2N0nrk4oFULZbS9b17AtzU18LjP"
# genai.configure(api_key="AIzaSyAB34HK9AGBWkrDiw7a1T9fQkS4QaklEHk")  # 设置 Gemini API 密钥

# 数据集定义保持不变
category_list_all = {
    'Emotion6': emotion6
}

vowel_list = ['A', 'E', 'I', 'O', 'U']

Path(f"generic").mkdir(parents=True, exist_ok=True)

for k, v in category_list_all.items():
    all_json_dict = {}
    all_responses = {}
    print('Generating descriptions for ' + k + ' dataset.')
    json_name_all = f"generic/{k}.json"

    if Path(json_name_all).is_file():
        raise ValueError("File already exists")

    for i, category in enumerate(tqdm(v)):
        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"

        if '_' in category:
            cat = category.replace('_', ' ')
        else:
            cat = category

        prompts = []
        prompts.append("Describe what " + article + " " + cat + " looks like")
        prompts.append("How can you identify " + article + " " + cat + "?")
        prompts.append("What does " + article + " " + cat + " want to express?")
        prompts.append("Describe an image from the internet of " + article + " " + cat)
        prompts.append("A caption of an image of " + article + " " + cat + ":")
        prompts.append("Describe what possible picture that can express"+ article + " " + cat +"'?")
        all_result = []
        for curr_prompt in prompts:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=curr_prompt,
                temperature=.99,
                max_tokens=50,
                n=5,
                stop="."
            )

            for r in range(len(response["choices"])):
                result = response["choices"][r]["text"]
                all_result.append(result.replace("\n\n", "") + ".")

        all_responses[category] = all_result

        # if i % 10 == 0:
    with open(json_name_all, 'w') as f:
        json.dump(all_responses, f, indent=4)