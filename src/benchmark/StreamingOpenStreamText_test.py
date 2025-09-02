import os
import json
from tqdm import tqdm

# 复用现有工具
from utils.data_execution import get_model_response_text_stream
from utils.video_execution import split_video

# ====== PROMPT（与文件内保持一致）======
PROMPT_TEMPLATE = '''You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}
{}
{}
{}'''

PROMPT_TEMPLATE_WITHOUT_OPTIONS = '''You are an advanced video question-answering AI assistant. You have been provided with a video and a question related to the video. Your task is to carefully analyze the video and provide the answer to the question. 

Question: {}
'''

def _ensure_labeled_options(options):
    """把['opt1','opt2','opt3','opt4'] 规范为 ['A. ...','B. ...','C. ...','D. ...']"""
    if len(options) != 4:
        raise ValueError("Options must have length 4.")
    if not options[0].strip().startswith("A."):
        options = [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"]
    return options

def _letter_from_response(text):
    """从模型输出中抽取首个 A/B/C/D 字母。"""
    for ch in (text or "").strip():
        up = ch.upper()
        if up in ("A","B","C","D"):
            return up
    return ""

def run_memory_consistency_probe_stream_text(
    data,
    MODEL,
    output_path="consistency_probe_stream_text.json",
    step_seconds=6,        # 每步追加的“新视觉流”时长（V2 开始）
    max_steps=8,           # 最多追加几步（V1 + max_steps = Vn）
    only_first_question=True,  # True：仅测每个样本的 Q1；False：对每个样本的所有问题都做“同题反复询问”探测
    session_id="probe-123" # 会话标识（供流式接口复用内部记忆）
):
    """
    验证：随着视觉流持续注入（按 step_seconds 追加片段），同一题在 V1 答对、随后某一步开始变错（True->False）。
    - V1：切 [0, ts] 作为第一段视觉流，提问 Q1。
    - V2...Vn：继续切 [prev_end, ts + k*step] 作为“新注入”的视觉流，重复同一题 Q1。
    - 通过 get_model_response_text_stream 保持 isBegin=False 持续同一会话，考察记忆/状态对答案的影响。
    结果写入 JSON，并打印翻转摘要。
    """
    results = {
        "model": MODEL.name(),
        "step_seconds": step_seconds,
        "max_steps": max_steps,
        "probes": []  # 每个被测问题的一次完整探测记录
    }
    flips = []  # 收集 True->False 的样本，用于摘要打印

    # data: 形如 [{ "video_path":..., "questions":[{...}, ...] }, ...]
    for subset in tqdm(data):
        video_path = subset["video_path"]
        questions = subset["questions"]
        if only_first_question:
            questions = questions[:1]  # 只测 Q1

        for q in questions:
            # 解析时间戳
            ts = sum(int(x) * 60 ** i for i, x in enumerate(reversed(q["time_stamp"].split(":"))))
            ques = q["question"]
            gold = (q.get("answer","") or "").strip().upper()
            audio_path = q.get("audio_path", None)

            # 构造 Prompt
            has_options = "options" in q and q["options"] and len(q["options"]) == 4
            if has_options:
                options = _ensure_labeled_options(q["options"])
                prompt = PROMPT_TEMPLATE.format(ques, *options) + "\n\nThe best option is:"
            else:
                prompt = PROMPT_TEMPLATE_WITHOUT_OPTIONS.format(ques) + "\n\nAnswer:"

            # ====== 开始一次完整探测（V1..Vn）======
            probe_seq = []
            flipped_at = None

            # V1：把 [0, ts] 作为第一段视觉流
            isBegin = True
            prev_end = 0  # 上一次切片的结束时间，用于流式“增量注入”
            try:
                clip_file = split_video(video_path, max(0, prev_end), ts)
                prev_end = ts
            except Exception as e:
                # 切片失败，记录后跳过
                results["probes"].append({
                    "video_path": video_path,
                    "question": ques,
                    "time_stamp": q["time_stamp"],
                    "error": f"split_video failed at V1: {repr(e)}"
                })
                continue

            # 首次推理（V1）
            resp, meta = get_model_response_text_stream(MODEL, clip_file, audio_path, session_id, isBegin, prompt)
            isBegin = False  # 后续保持 False
            pred = _letter_from_response(resp) if has_options else resp.strip()
            correct = (pred == gold) if has_options else None

            probe_seq.append({
                "step": 0,
                "start_time": 0,
                "end_time": ts,
                "pred": pred,
                "gold": gold if has_options else None,
                "correct": correct,
                "raw_response": (resp or "").strip(),
                "results_meta": meta  # 若过大可选择不写出
            })
            try:
                os.remove(clip_file)
            except Exception:
                pass

            # V2..Vn：继续注入新视觉流（同一题、同一 prompt、不变的会话）
            for k in range(1, max_steps + 1):
                t_end = ts + k * step_seconds
                try:
                    clip_file = split_video(video_path, max(0, prev_end), t_end)
                    prev_end = t_end
                except Exception as e:
                    probe_seq.append({
                        "step": k,
                        "end_time": t_end,
                        "error": f"split_video failed at step {k}: {repr(e)}"
                    })
                    break

                resp, meta = get_model_response_text_stream(MODEL, clip_file, audio_path, session_id, isBegin, prompt)
                pred = _letter_from_response(resp) if has_options else resp.strip()
                correct_k = (pred == gold) if has_options else None

                probe_seq.append({
                    "step": k,
                    "start_time": probe_seq[-1]["end_time"],  # 上一步的末尾
                    "end_time": t_end,
                    "pred": pred,
                    "gold": gold if has_options else None,
                    "correct": correct_k,
                    "raw_response": (resp or "").strip(),
                    "results_meta": meta
                })
                try:
                    os.remove(clip_file)
                except Exception:
                    pass

                # 记录首次 True->False 翻转点
                if has_options and flipped_at is None:
                    prev_correct = probe_seq[-2].get("correct", False)  # 上一步
                    now_correct = correct_k
                    if prev_correct and (now_correct is False):
                        flipped_at = {"step": k, "end_time": t_end}

            # 汇总记录
            record = {
                "video_path": video_path,
                "question": ques,
                "time_stamp": q["time_stamp"],
                "has_options": has_options,
                "gold": gold if has_options else None,
                "sequence": probe_seq,
                "initial_correct": (probe_seq[0].get("correct", False) if has_options and probe_seq else None),
                "flipped_true_to_false": (flipped_at is not None) if has_options else None,
                "flip_point": flipped_at
            }
            results["probes"].append(record)

            # 摘要清单
            if has_options and record["initial_correct"] and flipped_at is not None:
                flips.append({
                    "video": os.path.basename(video_path),
                    "q_time": q["time_stamp"],
                    "flip_step": flipped_at["step"],
                    "flip_end_time": flipped_at["end_time"],
                    "question": ques
                })

    # 输出 JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # 控制台摘要
    print("\n=== Streaming OpenStreamText: Memory Consistency Probe (True→False) ===")
    print(f"Model: {results['model']}")
    print(f"Total probed items: {len(results['probes'])}")
    if flips:
        print(f"True→False flips: {len(flips)}")
        for item in flips[:20]:
            print(f" - {item['video']} @ {item['q_time']} | flip at step {item['flip_step']} (end={item['flip_end_time']}s)")
    else:
        print("No True→False flips detected under current settings.")