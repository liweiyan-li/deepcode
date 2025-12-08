import os
from openai import OpenAI
import base64
os.environ["ARK_API_KEY"] = "aa209038-62e0-4271-b8a3-de7c9be8cb72"

# 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
# 初始化Ark客户端，从环境变量中读取您的API Key
client = OpenAI(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
    api_key=os.environ.get("ARK_API_KEY"),
)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

image_path = "/home/user02/项目中期/下载视频/autodl_tmp/3-3.jpg"
base64_image = encode_image(image_path)

response = client.chat.completions.create(
  # 替换 <MODEL> 为模型的Model ID
  model="doubao-seed-1-6-thinking-250615",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
          # 需要注意：传入Base64编码前需要增加前缀 data:image/{图片格式};base64,{Base64编码}：
          # PNG图片："url":  f"data:image/png;base64,{base64_image}"
          # JPEG图片："url":  f"data:image/jpeg;base64,{base64_image}"
          # WEBP图片："url":  f"data:image/webp;base64,{base64_image}"
            "url":  f"data:image/jpeg;base64,{base64_image}",
            "detail": "high"
          },         
        },
        {
          "type": "text",
          "text": "图片里讲了什么?",
        },
      ],
    }
  ],
)

print(response.choices[0])