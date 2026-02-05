from zhipuai import ZhipuAI
import os
import re


API_KEY = os.getenv("ZHIPU_API_KEY")

client = ZhipuAI(
    api_key=API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

def _clean_output(text: str) -> str:
    if not text:
        return text

    # Keep first line only
    text = re.split(r'\n', text)[0]
    return text.strip()


def fix_asr_text(text: str) -> str:

    if not text or text.strip() == "":
        return text

    try:
        response = client.chat.completions.create(
            model="glm-4.7-flash",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是语音识别文本修正器。"
                        "只输出最终修正后的中文句子。"
                        "禁止解释或分析。"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "修正以下语音识别文本，使其成为自然标准书面中文。"
                        "允许添加标点和删除重复词，但不得改变含义：\n\n"
                        f"{text}"
                    )
                }
            ],

            temperature=0,
            max_tokens=150,

            #----- THIS IS THE KEY FIX -------
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            }
            #---------------------------------
        )

        raw = response.choices[0].message.content.strip()
        print(response)
        return _clean_output(raw)

    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return text
