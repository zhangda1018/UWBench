import os
import re
import json
import random
import time
import base64
from openai import AzureOpenAI, OpenAI, api_key
from openai._exceptions import RateLimitError, APITimeoutError, InternalServerError
from .models_config import build_completion_params


os.environ['PYTHONUNBUFFERED'] = '1'

def readJson(path):
    '''
    入参：
        path:需要读取的 json 文件路径，支持 json 和 jsonl 文件
    出参：
        jData：读取到的 json 文件内容，如果是 json 文件则直接返回 json 文件内容，如果是 json 则返回一个 list
    '''
    if not path or not os.path.exists(path):
        print("JSON file read failed.{}".format(path))
        raise FileNotFoundError(f"File not found: {path}")

    file_extension = os.path.splitext(path)[1].lower()

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

def writeJson(data, path, att=True,ensure_ascii=False):
    '''
    入参：
        data: 需要保存的数据
        path: 保存的路径
        att: 是否提示路径存在文件
        ensure_ascii: ensure_ascii 选项默认false
    出参：
        如果正常保存，则返回 True
    '''
    if os.path.exists(path):
        if att:
            print("File exists, pay attention!!!!!")

    file_extension = os.path.splitext(path)[1].lower()

    try:
        if file_extension == ".json":
            with open(path, "w", encoding="utf8") as f:
                json.dump(data, f, ensure_ascii=ensure_ascii)
        elif file_extension == ".jsonl":
            with open(path, "w", encoding="utf8") as f:
                for item in data:
                    json_line = json.dumps(item, ensure_ascii=ensure_ascii)
                    f.write(json_line + "\n")
        else:
            raise ValueError("Unsupported file extension. Use '.json' or '.jsonl'")
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        raise e
    return True

def img2base64_complete(img):
    """
    入参：
        img: 输入的img，可以是path、需要补全的 base64 或者是 url（url 不会解析，会原封不动的返回）
    出参：
        base64_complete: 转换 or 补全后的 base64 图片
    """
    support_type = ["png", "jpg", "jpeg", "gif","bmp","svg",'tiff']
    # 如果是完整的 base64 或者是 url，直接返回
    check_strs = ["data:image/", "http://", "https://"]
    for x in check_strs:
        if x in img:
            return img
    # 如果输入是不完整的 base64 里面不会包含 .
    if not '.' in img:
        base64_encoded_data = img  
        img_type = "png"
    else:
        img_type = img.split(".")[-1].lower()
        if img_type in support_type:
            if not img or not os.path.exists(img): # 如果路径不存在
                print(f'img:{img}')
                raise ValueError("File not exists error")
            # 读取 img 并转换为 base64 编码
            with open(img, "rb") as image_file:
                base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
            # jpg --> jpeg
            if img_type == "jpg": img_type = "jpeg"
        else:
            raise ValueError(f"Unsupported file extension. Please check your image file is in {support_type}")
    base64_complete = f"data:image/{img_type};base64,{base64_encoded_data}"
    return base64_complete

class ChatBot(object):
    """
    入参：
        api: api 定义，python 字典，格式如下
            {
            "type":  必填，api 类型，目前支持：['AZURE','REQUEST','OPENAI']中的任意一个,
            "base": 必填，模型的 baseurl,
            "key": 必填，模型的 key,
            "engine": 必填，模型名称例如："qwen-vl-plus",
            "max_tokens": 选填，如果模型支持 4096 以上的输出长度，且需要指定输出长度超过 4096，在这里设置，如果不设置默认为 4096 
            "temperature": 选填，模型温度，默认为 0.0"
            }
    """
    def __init__(self, api, client_id=99, max_try=10) -> None:
        """
        mode is in ['AZURE','REQUEST','OPENAI']
        """
        self.api = api
        self.check_api()
        if api["type"] == "AZURE":
            # 检查是否使用Responses API
            if api.get("use_responses", False):
                self.client = OpenAI(
                    base_url=f"{api['base']}/openai/v1",
                    api_key=api["key"],
                    # v1 GA 无需传递 api-version
                )
                self.use_responses = True
            else:
                self.client = AzureOpenAI(
                    api_key=api["key"],
                    api_version=api["version"],
                    azure_endpoint=api["base"],
                )
                self.use_responses = False
        if api["type"] == "REQUEST":
            self.base_url = f"{api['base']}openai/deployments/{api['engine']}/chat/completions?api-version={api['version']}"
        if api["type"] == "OPENAI":
            self.client = OpenAI( api_key=api["key"], base_url=api["base"])
        self.client_id = client_id
        self.max_try = max_try

    def check_api(self):
        """
        set some default params like temperature, max_tokens
        and check if the api is valid
        """
        for required_keys in ["base", "key", "engine","type"]:
            if required_keys not in self.api.keys():
                raise ValueError(f"Api must have {required_keys}, plase check key: {self.api}")
        
        # 对于Azure类型，version字段是必需的
        if self.api["type"].upper() == "AZURE" and "version" not in self.api.keys():
            raise ValueError(f"Azure API must have version field, please check key: {self.api}")
            
        self.api["type"] = self.api["type"].upper()
        if "max_tokens" not in self.api.keys():
            self.api["max_tokens"] = 4096
        if "temperature" not in self.api.keys():
            self.api["temperature"] = 0.0
        if "comment" in self.api.keys():
            self.api["comment"] = self.api["comment"].lower()
    def call(self, txt, img=None, isMsg=False, test=False, system_prompt=None):
        if test:print(f'client_id: {self.client_id}, api_key: {self.api}', flush=True)
        if self.api.get("type") in ["OPENAI","AZURE"]:
            return self.call_openai(txt, img, isMsg, test, system_prompt)
        else:
            raise ValueError(f"Unsupported api type: {self.api['type']}")
    def get_messages(self, txt, img=None, isMsg=False, system_prompt=None):
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
        return messages
    
    def call_openai(self, txt, img=None, isMsg=False, test=False, system_prompt=None):
        messages = self.get_messages(txt, img, isMsg, system_prompt)
        for i in range(self.max_try):
            try:
                if hasattr(self, 'use_responses') and self.use_responses:
                    # 使用Responses API
                    if img is not None:
                        # GPT-5的Responses API支持图像输入，使用正确的格式
                        print("使用GPT-5 Responses API处理图像输入")
                        
                        # 构建Responses API的图像输入格式
                        input_content = []
                        instructions = ""
                        
                        for msg in messages:
                            if msg["role"] == "system":
                                instructions = msg["content"]
                            elif msg["role"] == "user":
                                if isinstance(msg["content"], list):
                                    # 处理多模态内容
                                    for content_item in msg["content"]:
                                        if content_item["type"] == "text":
                                            input_content.append({
                                                "type": "input_text", 
                                                "text": content_item["text"]
                                            })
                                        elif content_item["type"] == "image_url":
                                            # 直接使用image_url字符串，不是对象
                                            input_content.append({
                                                "type": "input_image",
                                                "image_url": content_item["image_url"]["url"]
                                            })
                                else:
                                    # 纯文本
                                    input_content.append({
                                        "type": "input_text",
                                        "text": msg["content"]
                                    })
                        
                        extra_args = {}
                        if "reasoning_effort" in self.api:
                            extra_args["reasoning"] = {"effort": self.api["reasoning_effort"]}
                        if "text_verbosity" in self.api:
                            extra_args["text"] = {"verbosity": self.api["text_verbosity"]}

                        response = self.client.responses.create(
                            model=self.api["engine"],
                            max_output_tokens=self.api.get("max_tokens", 4096),
                            instructions=instructions if instructions else None,
                            input=[{
                                "role": "user",
                                "content": input_content
                            }],
                            **extra_args,
                        )
                        content = response.output_text
                        # 优先从 Responses API 的 usage 中读取 token，用于 Azure v1 Responses
                        usage = getattr(response, 'usage', None)
                        if usage is not None:
                            prompt_tokens = getattr(usage, 'input_tokens', getattr(usage, 'prompt_tokens', 0))
                            completion_tokens = getattr(usage, 'output_tokens', getattr(usage, 'completion_tokens', 0))
                        else:
                            # 兼容旧字段（通常不存在）
                            prompt_tokens = getattr(response, 'prompt_tokens', 0)
                            completion_tokens = getattr(response, 'completion_tokens', 0)
                    else:
                        # 纯文本输入，使用Responses API
                        # 将messages转换为instructions和input
                        instructions = ""
                        input_text = ""
                        
                        for msg in messages:
                            if msg["role"] == "system":
                                instructions = msg["content"]
                            elif msg["role"] == "user":
                                input_text = msg["content"]
                        
                        extra_args = {}
                        if "reasoning_effort" in self.api:
                            extra_args["reasoning"] = {"effort": self.api["reasoning_effort"]}
                        if "text_verbosity" in self.api:
                            extra_args["text"] = {"verbosity": self.api["text_verbosity"]}

                        response = self.client.responses.create(
                            model=self.api["engine"],
                            max_output_tokens=self.api.get("max_tokens", 4096),
                            instructions=instructions if instructions else None,
                            input=input_text,
                            **extra_args,
                        )
                        content = response.output_text
                        # 优先从 Responses API 的 usage 中读取 token
                        usage = getattr(response, 'usage', None)
                        if usage is not None:
                            prompt_tokens = getattr(usage, 'input_tokens', getattr(usage, 'prompt_tokens', 0))
                            completion_tokens = getattr(usage, 'output_tokens', getattr(usage, 'completion_tokens', 0))
                        else:
                            prompt_tokens = getattr(response, 'prompt_tokens', 0)
                            completion_tokens = getattr(response, 'completion_tokens', 0)
                else:
                    # 使用传统的Chat Completions API
                    if self.api.get("comment") in ["cot"]:
                        response = self.client.chat.completions.create(
                        model=self.api["engine"],
                        messages=messages,)
                    else:
                        # 使用配置系统自动处理不同模型的参数
                        params = build_completion_params(self.api, messages)
                        # API调用参数准备完成
                        response = self.client.chat.completions.create(**params)
                    content = response.choices[0].message.content
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                
                return [content, prompt_tokens, completion_tokens]
            except RateLimitError as e:
                try:
                    sleep_time = int(re.findall("Please retry after ([0-9]*) second", str(e))[0])
                except:
                    sleep_time = 10
                delayTime = random.randint(5, 10)
                print(f"Client {self.client_id} (Try {i}) failed with error {e}. Will retry in {sleep_time + delayTime} seconds")
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
                print( f"Client {self.client_id} (Try {i})  failed with error \n****\n\n {e} \n****\n\n. Will retry in 5 seconds" )
                time.sleep(5)
                if test:
                    break
        return None

class ChatBots(object):
    def __init__(self, apis, max_try=10, do_log=True, log_file_path=None) -> None:
        self.chat_bots = [ChatBot(api, client_id=i, max_try=max_try) for i, api in enumerate(apis)]
        self.chat_bots_num = len(apis)
        self.max_try = max_try
        self.do_log = do_log
        self.log_file_path = log_file_path or 'gptCallLog.jsonl.log'

    def call(self, txt, img=None, isMsg=False, test=False, system_prompt=None):
        cur_try = 0
        completion_result = None
        while cur_try < self.max_try and not completion_result:
            chatbot = self.chat_bots[random.randint(0, self.chat_bots_num - 1)]
            if test:print(chatbot.api)
            completion_result = chatbot.call(txt, img, isMsg, test, system_prompt)
            cur_try += 1
        if self.do_log:
            with open(self.log_file_path,'a') as f:
                f.write(json.dumps({"txt":txt,"img":img,"isMsg":isMsg,"system_prompt":system_prompt,"completion_result":completion_result}, ensure_ascii=False) + '\n')
        return completion_result
    
    def test(self,txt=False,isVLLM=False):
        if not txt:
            txt = "请如实回答我你是不是 ChatGPT？鲁迅和周树人是什么关系？"
            img = None
        if isVLLM:
            txt = "请告诉我你是什么型号的模型，并告诉我图中是什么？"
            img = [
            r"data:image/png;base64,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"][0]
        # for bot in self.chat_bots:
        #     print(bot.call(txt, img, test=True))
        return
    
if __name__ == "__main__":
    #from keys import GPT4os,GPT4s,static_keys,testkeys,o1minis,o1s
    from keys import claudetest as testedkyes
    chat_bots = ChatBots(testedkyes)
    chat_bots.test()
    # test VLLMs
    # chat_bots.test(isVLLM=True)

    