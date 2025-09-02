from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import os
import torch

model, processor = None, None

from model.modelclass import Model

class Qwen25VL(Model):
    def __init__(self):
        Qwen25VL_Init()

    def Run(self, file, inp):
        return Qwen25VL_Run(file, inp)

    def name(self):
        return "Qwen2.5-VL"

def Qwen25VL_Init():
    """
    初始化 Qwen2.5-VL-7B-Instruct
    """
    global model, processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",  # 可按需去掉
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def Qwen25VL_Run(file, inp):
    """
    file: 本地视频路径
    inp : 文本指令
    """
    # 根据文件名里两帧索引计算帧数，动态设定 fps
    numbers = re.findall(r'\d+', os.path.basename(file))
    frame_num = int(numbers[-1]) - int(numbers[-2])

    if 300 < frame_num < 600:
        fps = 0.5
    elif frame_num >= 600:
        fps = 0.2
    else:
        fps = 1.0

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{file}",
                    "max_pixels": 360 * 420,  # 可按显存调
                    "fps": fps,               # Qwen2.5-VL 会利用绝对时间
                },
                {"type": "text", "text": inp},
            ]
        }
    ]

    # 1) 生成 chat 模板
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 2) 解析视觉输入；2.5 版本建议把 video kwargs 一并拿到并传回 processor
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    # 3) 构造张量（把 fps 一并传给 processor；video_kwargs 里也可能包含读取所需信息）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=fps,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    # 4) 推理
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0]
    print(response)
    return response