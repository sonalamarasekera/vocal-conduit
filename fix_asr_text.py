from zhipuai import ZhipuAI
import os
import re
import time

API_KEY = "8446ac418fc747d08ebe1e60e24d0733.YoOoPTf6ftBepD4b" #os.getenv("ZHIPU_API_KEY")

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
    start = time.time()

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
                        "任务是将ASR结果修正为自然、标准的书面中文。"
                        "只输出最终修正后的句子。"
                        "禁止解释、分析或额外说明。"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "请修正以下语音识别文本：\n"
                        "1. 添加正确标点。\n"
                        "2. 修复语法错误。\n"
                        "3. 根据上下文修复明显的错别字或误识别词语（例如同音词错误）。\n"
                        "4. 仅在确定存在识别错误时才修改词语。\n"
                        "5. 不允许改写句子结构。\n"
                        "6. 不允许总结、扩展或改变原始语义。\n\n"
                        f"{text}"
                    )
                }
            ],

            temperature=0,
            max_tokens=150,

            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            }
        )

        raw = response.choices[0].message.content.strip()
        #print(response)
        latency = time.time() - start
        return _clean_output(raw), latency

    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return text
