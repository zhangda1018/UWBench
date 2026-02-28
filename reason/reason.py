import openai
import base64

AZURE_OPENAI_KEY = ""
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_DEPLOYMENT = "gpt4o"
API_VERSION = "2024-02-01"

client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=API_VERSION,
)

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_path = r"1.jpg"
base64_image = image_to_base64(image_path)

response = client.chat.completions.create(
    model="gpt-4o",               # ← 模型名
    deployment_id= 'gpt4o',      # ← 你的Azure部署名
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这张图片"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ],
        }
    ],
    max_tokens=1024,
)

print(response.choices[0].message.content)
