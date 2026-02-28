import re
import os.path as osp
import time
import json
import pdb
import random
from openai import AzureOpenAI, RateLimitError, InternalServerError, APITimeoutError


def readJson(path):
    if not path or not osp.exists(path):
        print("JSON file read failed.{}".format(path))
        raise FileNotFoundError(f"File not found: {path}")

    file_extension = osp.splitext(path)[1].lower()

    try:
        if file_extension == ".json":
            with open(path, "r", encoding="utf8") as f:
                jData = json.load(f)
        elif file_extension == ".jsonl":
            jData = []
            with open(path, "r", encoding="utf8") as f:
                for line in f:
                    jData.append(json.loads(line.strip()))
        else:
            raise ValueError("Unsupported file extension. Use '.json' or '.jsonl'")
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        raise e

    return jData


def writeJson(data, path, att=True):
    if osp.exists(path):
        if att:
            print("File exists, pay attention!!!!!")

    file_extension = osp.splitext(path)[1].lower()

    try:
        if file_extension == ".json":
            with open(path, "w", encoding="utf8") as f:
                json.dump(data, f, ensure_ascii=False)
        elif file_extension == ".jsonl":
            with open(path, "w", encoding="utf8") as f:
                for item in data:
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + "\n")
        else:
            raise ValueError("Unsupported file extension. Use '.json' or '.jsonl'")
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        raise e
    return True


def img2base64_complete(img):
    """
    support img path or uncompleted base64 (not start with data:image/xxx) as input
    """
    # if img is completed base64 or pure url
    base64_encoded_data = img  # if img is uncompleted base64
    img_type = "png"  # if img is uncompleted base64
    check_strs = ["data:image/", "http://", "https://"]
    for x in check_strs:
        if x in img:
            return img
    import base64

    try:  # if img is image path
        img_type = img.split(".")[-1]
        if img_type in ["png", "jpg", "jpeg", "gif"]:
            # Read and encode the image file
            with open(img, "rb") as image_file:
                base64_encoded_data = base64.b64encode(image_file.read()).decode(
                    "utf-8"
                )
            if img_type == "jpg":
                img_type = "jpeg"
    except:  # if img is uncompleted base64
        pass
    base64_complete = f"data:image/{img_type};base64,{base64_encoded_data}"
    return base64_complete


LINEPERAPI = 3
DEBUG = False
MAX_TRY = 5


class ChatBot(object):
    def __init__(
        self, api, client_id=None, max_try=10, mode="AZURE", max_tokens=4096, tem=0.000001
    ) -> None:
        """
        mode is in ['AZURE','REQUEST']
        """
        if mode == "AZURE":
            self.client = AzureOpenAI(
                api_key=api["key"],
                api_version=api["version"],
                azure_endpoint=api["base"],
                # base_url=f"{api['base']}openai/deployments/{api['engine']}",
            )
        if mode == "REQUEST":
            self.base_url = f"{api['base']}openai/deployments/{api['engine']}/chat/completions?api-version={api['version']}"
            self.key = api["key"]
        self.model = api["engine"]
        self.max_tokens = max_tokens
        self.mode = mode
        self.client_id = client_id
        self.max_try = max_try
        self.tem = tem
        self.key = api["key"]
        if client_id is None:
            self.client_id = 99

    def call(self, txt, img=None, isMsg=False, test=False, system_prompt=None):
        if not isMsg:
            if img is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": txt},
                            {
                                "type": "image_url",
                                "image_url": {"url": img2base64_complete(img)},
                            },
                        ],
                    }
                ]
            else:
                messages = [{"role": "user", "content": str(txt)}]
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            messages = txt
        for i in range(self.max_try):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.tem,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                return [content, prompt_tokens, completion_tokens]
            except RateLimitError as e:
                sleep_time = int(
                    re.findall("Please retry after ([0-9]*) seconds", str(e))[0]
                )
                delayTime = random.randint(5, 10)
                print(
                    f"Client {self.client_id} (Try {i}) failed with error {e}. Will retry in {sleep_time + delayTime} seconds"
                )
                time.sleep(sleep_time + delayTime)
                if test:
                    break
            except InternalServerError as e:
                print("InternalServerError!!!")
                break
            except APITimeoutError as e:
                print("APITimeoutError!!!")
                break
            except Exception as e:
                if "invalid_request_error" in str(e):
                    print("Tooooooooo Longggggggg")
                    print("Please make sure you are using vision model for images.")
                    break
                print(
                    f"Client {self.client_id} (Try {i})  failed with error {e}. Will retry in 5 seconds"
                )
                time.sleep(5)
                if test:
                    break
        return None


class ChatBots(object):
    def __init__(self, apis, max_try=10) -> None:
        self.chat_bots = [
            ChatBot(api, client_id=i, max_try=1) for i, api in enumerate(apis)
        ]
        self.chat_bots_num = len(apis)
        self.max_try = max_try

    def call(self, txt, img=None, isMsg=False, test=False, system_prompt=None):
        cur_try = 0
        completion_result = None
        while cur_try < self.max_try and not completion_result:
            chatbot = self.chat_bots[random.randint(0, self.chat_bots_num - 1)]
            print(chatbot.key)
            completion_result = chatbot.call(txt, img, isMsg, test, system_prompt)
            cur_try += 1
        return completion_result

GPT4s = [
    {
        "type": "azure",
        "base": "xxxxxx",
        "version": "2024-02-15-preview",
        "key": "xxxxxx",
        "engine": "gpt4",
    },
    {
        "type": "azure",
        "base": "xxxxxx",
        "version": "2024-02-15-preview",
        "key": "xxxxx",
        "engine": "gpt4",
    }
]

GPT4os = [
    {
        "type": "azure",
        "base": "",
        "version": "2024-02-01",
        "key": "",
        "engine": "gpt4o",
    },
    {
        "type": "azure",
        "base": "xxxxxx",
        "version": "2024-02-15-preview",
        "key": "xxxxx",
        "engine": "gpt4o",
    }
]



if __name__ == "__main__":
    # example for txt calling
    inputString = "鲁迅和周树人的关系是什么？"
    gpt4_chat_bots = ChatBots(GPT4os)
    print(gpt4_chat_bots.call(inputString))


    # # example for img calling
    # inputString = "请帮我描述一下这个图片的内容"
    # image_path = r'path to your image XXXXXXXXXXXXXXXXXXX'
    # gpt4o_chat_bots = ChatBots(GPT4os)
    # print(gpt4o_chat_bots.call(inputString, img=image_path))


