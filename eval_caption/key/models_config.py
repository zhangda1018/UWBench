#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型API配置文件
支持多种模型的API配置和选择
"""

# 模型特殊参数配置
MODEL_PARAM_SPECS = {
    "gpt-5": {
        "supports_custom_temperature": False,  # 不支持自定义temperature
        "max_tokens_param": "max_completion_tokens",  # 使用max_completion_tokens而不是max_tokens
        "default_temperature": 1.0,  # 默认temperature
    },
    "gpt-5-mini": {
        "supports_custom_temperature": False, 
        "max_tokens_param": "max_completion_tokens",
        "default_temperature": 1.0,
    },
    "o3": {
        "supports_custom_temperature": False,  # o3不支持自定义temperature
        "max_tokens_param": "max_completion_tokens",  # 使用max_completion_tokens而不是max_tokens
        "default_temperature": 1.0,  # 默认temperature
    },
    "o4-mini": {
        "supports_custom_temperature": False,  # o4-mini不支持自定义temperature
        "max_tokens_param": "max_completion_tokens",  # 使用max_completion_tokens而不是max_tokens
        "default_temperature": 1.0,  # 默认temperature
    },
    # 其他模型使用标准参数（默认行为）
}

# 所有可用的模型API配置
AVAILABLE_MODELS = {
    "gpt4o": [
        {
            "type": "AZURE",
            "base": "https://research-01-01.openai.azure.com/",
            "key": "",
            "engine": "gpt4o",
            "version": "",
            "max_tokens": 4096,
            "temperature": 0.0,
            "comment": "Azure GPT-4o"
        }
    ],
    # "gpt4o": [
    #     {
    #         "type": "AZURE",
    #         "base": "",
    #         "key": "",
    #         "engine": "gpt4o",
    #         "version": "",
    #         "max_tokens": 4096,
    #         "temperature": 0.0,
    #         "comment": "Azure GPT-4o"
    #     }
    # ],
    "GLM-4.5V-106B-A12B": [
        {
            "type": "OPENAI",
            "base": "http://127.0.0.1:8100/v1",
            "key": "EMPTY",
            "engine": "GLM-4.5V-106B-A12B",
            "max_tokens": 4096,
            "temperature": 0.0,
            "comment": "LMDeploy OpenAI 兼容端点（GLM-4.5V-106B-A12B，图像输入已启用）"
        }
    ],
    "GLM-4.1V-9B-Thinking": [
        {
            "type": "OPENAI",
            "base": "http://127.0.0.1:8100/v1",
            "key": "EMPTY",
            "engine": "GLM-4.1V-9B-Thinking",
            "max_tokens": 4096,
            "temperature": 0.0,
            "comment": "LMDeploy OpenAI 兼容端点（GLM-4.1V-9B-Thinking，图像输入已启用）"
        }
    ],
    "qwen2.5-vl-3b": [
        {
            "type": "OPENAI",
            "base": "http://127.0.0.1:8010/v1",
            "key": "EMPTY",
            "engine": "Qwen2.5-VL-3B-Instruct",
            "max_tokens": 4096,
            "temperature": 0.0,
            "comment": "vLLM OpenAI 兼容端点（3B，图像输入已启用）"
        }
    ],
    "qwen2.5-vl-7b": [
        {
            "type": "OPENAI",
            "base": "http://127.0.0.1:8020/v1",
            "key": "EMPTY",
            "engine": "Qwen2.5-VL-7B-Instruct",
            "max_tokens": 4096,
            "temperature": 0.0,
            "comment": "vLLM OpenAI 兼容端点（7B，图像输入已启用）"
        }
    ],
    "qwen2.5-vl-72b": [
        {
            "type": "OPENAI",
            "base": "http://127.0.0.1:8030/v1",
            "key": "EMPTY",
            "engine": "Qwen2.5-VL-72B-Instruct",
            "max_tokens": 4096,
            "temperature": 0.0,
            "comment": "vLLM OpenAI 兼容端点（72B，TP=4，图像输入已启用）"
        }
    ],
    "gpt-4o-mini": [
        {
            "type": "AZURE",
            "base": "",
            "key": "",
            "engine": "gpt-4o-mini",
            "version": "",
            "max_tokens": 4096,
            "temperature": 0.0,
            "use_responses": True,  # 使用Responses API
            "comment": "Azure GPT-4o-mini with Responses API"
        }
    ],
    "gpt-5": [
        {
            "type": "AZURE",
            "base": "",
            # "key": "",
            "key": "",
            "engine": "gpt-5",
            "version": "",
            "max_tokens": 8192,  # 增加token限制
            "temperature": 0.0,
            "use_responses": True,  # 使用Responses API
            "reasoning_effort": "medium",  # 压缩思维链以提速: minimal/low/medium/high
            "text_verbosity": "medium",        # 输出更简洁: low/medium/high
            "comment": "GPT-5 with Responses API (auto-fallback to Chat Completions for images)"
        }
    ],
    "gpt-5-mini": [
        {
            "type": "AZURE",
            "base": "",
            "key": "",
            "engine": "gpt-5-mini",
            "version": "",
            "max_tokens": 4096,  # 增加token限制
            "temperature": 0.0,
            "use_responses": True,  # 使用Responses API
            "reasoning_effort": "minimal",  # 压缩思维链以提速
            "text_verbosity": "low",        # 输出更简洁
            "comment": "GPT-5-mini with Responses API (supports images)"
        }
    ],
    "gpt-o3": [
        {
            "type": "AZURE",
            "base": "",
            "key": "",
            "engine": "o3",
            "version": "",
            "max_tokens": 4096,  # 增加token限制
            "temperature": 0.0,
            "use_responses": True,  # 使用Responses API
            "reasoning_effort": "low",  # 压缩思维链以提速
            "comment": "GPT-o3 with Azure OpenAI (supports images)"
        }
    ],
    "o4-mini": [
        {
            "type": "AZURE",
            "base": "",
            "key": "",
            "engine": "o4-mini",
            "version": "",
            "max_tokens": 4096,  # 增加token限制
            "temperature": 0.0,
            "use_responses": True,  # 使用Responses API
            "reasoning_effort": "low",  # 压缩思维链以提速
            "comment": "GPT-o4-mini with Azure OpenAI (supports images)"
        }
    ],
    "gemini-2.5-flash": [
        {
            "type": "OPENAI",
            "base": "https://vip.dmxapi.com/v1",
            # "key": "",
            "key": "",
            "engine": "gemini-2.5-flash",
            "max_tokens": 8192,
            "temperature": 0.0,
            "comment": "DMX OpenAI 兼容端点（Gemini 2.5 Flash）"
        }
    ],
    "gemini-2.5-pro": [
        {
            "type": "OPENAI",
            "base": "https://vip.dmxapi.com/",
            "key": "",
            "engine": "gemini-2.5-pro",
            "max_tokens": 131072,
            "temperature": 0.0,
            "comment": "DMX OpenAI 兼容端点（Gemini 2.5 Pro）"
        }
    ],
    "claude-sonnet-4-20250514": [
        {
            "type": "OPENAI",
            "base": "https://vip.dmxapi.com/v1",
            "key": "",
            # "key":"",
            "engine": "claude-sonnet-4-20250514",
            "max_tokens": 8192,
            "timeout_seconds": 120,
            "temperature": 0.0,
            "comment": "DMX OpenAI 兼容端点（Claude Sonnet 4 20250514）"
        }
    ],
    # 更多模型...
}

# 评测配置
EVALUATION_SETTINGS = {
    "max_try": 6,
    "results_dir": "results",
    "max_tasks_per_category": 100,
    "save_interval": 50,
}

# 支持的文件扩展名
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']


def get_available_model_names():
    """获取所有可用的模型名称"""
    return list(AVAILABLE_MODELS.keys())


def get_model_config(model_name):
    """根据模型名称获取配置"""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"未知的模型: {model_name}. 可用模型: {get_available_model_names()}")
    return AVAILABLE_MODELS[model_name]


def get_model_param_spec(engine_name):
    """根据engine名称获取模型特殊参数规范"""
    return MODEL_PARAM_SPECS.get(engine_name, {
        "supports_custom_temperature": True,  # 默认支持自定义temperature
        "max_tokens_param": "max_tokens",     # 默认使用max_tokens
        "default_temperature": 0.0,          # 默认temperature
    })


def build_completion_params(api_config, messages):
    """根据模型配置构建completion参数"""
    engine = api_config["engine"]
    param_spec = get_model_param_spec(engine)
    
    params = {
        "model": engine,
        "messages": messages,
    }
    
    # 处理temperature参数
    if param_spec["supports_custom_temperature"]:
        params["temperature"] = api_config.get("temperature", param_spec["default_temperature"])
    # 如果不支持自定义temperature，则不传递该参数，使用模型默认值
    
    # 处理max_tokens参数
    max_tokens_param = param_spec["max_tokens_param"]
    if "max_tokens" in api_config:
        params[max_tokens_param] = api_config["max_tokens"]
    
    # 透传可选参数（若后端支持）
    if "top_p" in api_config:
        params["top_p"] = api_config["top_p"]
    if "stop" in api_config:
        params["stop"] = api_config["stop"]
    if "response_format" in api_config:
        params["response_format"] = api_config["response_format"]
        
    return params


def select_model_interactively():
    """交互式选择模型"""
    available_models = get_available_model_names()
    
    print("可用的模型:")
    for i, model_name in enumerate(available_models, 1):
        model_config = AVAILABLE_MODELS[model_name][0]  # 取第一个配置作为显示
        print(f"   {i}. {model_name} ({model_config.get('comment', 'No description')})")
    
    while True:
        try:
            choice = input(f"\n请选择要使用的模型 (1-{len(available_models)}): ").strip()
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(available_models):
                    selected_model = available_models[index]
                    print(f"已选择模型: {selected_model}")
                    return selected_model
            print("无效的选择，请重新输入。")
        except KeyboardInterrupt:
            print("\n已取消选择")
            return None
        except Exception as e:
            print(f"输入错误: {e}")
