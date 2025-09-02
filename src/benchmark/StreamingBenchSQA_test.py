import os
import json
import tqdm
import torch

from utils.data_execution import get_model_response
from utils.video_execution import split_video


from benchmark.Benchmark import Benchmark

PROMPT_TEMPLATE = '''You are an advanced video question-answering AI assistant. You have been provided with a video and a multiple-choice question related to the video. Your task is to carefully analyze the video and the provided context to answer the question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

{}

Here is the question. Answer it and don't confuse it with the previous conversation.
Question: {}

Options:
{}
{}
{}
{}

The best option is:'''
class StreamingBenchSQA_test(Benchmark):
    def __init__(self, data):
        StreamingBenchSQAInit(data)

    def eval(self, data, model, output_path, context_time, single_video, end_time_cap=None):
        if single_video == 1:
            run_single_video(data, model, output_path, context_time, end_time_cap=end_time_cap)
        elif single_video == 0:
            run_cross_video(data, model, output_path, context_time, end_time_cap=end_time_cap)
        else:
            raise ValueError("Invalid value for single_video. Must be 0 or 1.")

def StreamingBenchSQAInit(data):
    pass


def _ensure_labeled_options(options):
    """把['GS.', 'MIN.', ...] 规范成['A. GS.', 'B. MIN.', ...]"""
    if len(options) != 4:
        raise ValueError("Options must have length 4.")
    if not options[0].strip().startswith("A."):
        options = [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"]
    return options

def _letter_from_response(text):
    """从模型输出中抽取首个 A/B/C/D 字母。"""
    for ch in text.strip():
        up = ch.upper()
        if up in ("A","B","C","D"):
            return up
    # 兜底：如果没抓到，返回空
    return ""

def run_single_video(
    data,
    model,
    output_path,
    context_time=0,       # 历史窗口（问句时间点之前的上下文）
    step_seconds=450,          # 每步向后扩展的时长（V2 开始）
    max_steps=1,             # 最多扩展几步（V1 + max_steps = Vn）
    end_time_cap=None         # 绝对的 end_time 上限（秒），用于固定显存限制
):
    """
    对“同一问题在不断输入更多视频后，答案是否从 True 翻转为 False”进行探测。
    data: 读自 questions_sqa_stream.json 的 Python 对象（list[dict]）
    model: 你的模型封装（带 .name() 且可被 get_model_response 调用）
    output_path: 结果 JSON 路径
    """
    results = {
        "model": model.name(),
        "context_time": context_time,
        "step_seconds": step_seconds,
        "max_steps": max_steps,
        "probes": []  # 每个被测问题的一次完整探测记录
    }

    flips = []  # 收集出现 True->False 的条目，便于打印摘要

    for sample in tqdm.tqdm(data):
        # 每个样本可能由若干 subset（同一视频不同片段结构），兼容现在的数据结构
        subsets = sample if isinstance(sample, list) else [sample]
        context = ""
        for subset in subsets:
            video_path = subset["video_path"]
            questions = subset["questions"]
            # if only_first_question:
            #     questions = questions[:1]  # 只测 Q1

            for q in questions:
                # 目标问题与时间戳
                ques = q["question"]
                ans_letter = q["answer"].strip().upper()
                options = _ensure_labeled_options(q["options"])
                # ts = timestamp of the question
                # 把 "HH:MM:SS" 转成秒
                ts = sum(int(x) * 60 ** i for i, x in enumerate(reversed(q["time_stamp"].split(":"))))

                # V1：
                # context_time 为正时 看到 [ts - context_time, ts] 的视频
                # 否则 [0, ts]
                if context_time > 0:
                    t_start = max(0, ts - context_time)
                else:
                    t_start = 0

                # 逐步扩展终止时间：V1 的 end=ts；V2 的 end=ts+step_seconds；…
                probe_seq = []
                flipped_at = None
                last_end = None

                for step in range(0, max_steps + 1):
                    desired_end = ts + step * step_seconds
                    # 直接限制到固定上限，不需要寻找上限
                    t_end = min(desired_end, t_start + end_time_cap) if end_time_cap is not None else desired_end
                    # 早停
                    if last_end is not None and t_end == last_end:
                        break
                    last_end = t_end
                    try:
                        clip_file = split_video(video_path, t_start, t_end)
                    except Exception as e:
                        # 如果切片失败（超出视频末尾等），就结束这个问题的探测
                        probe_seq.append({
                            "step": step,
                            "end_time": t_end,
                            "error": f"split_video failed: {repr(e)}"
                        })
                        break

                    prompt = PROMPT_TEMPLATE.format(context, ques, *options)
                    print(f"[video {video_path.split('/')[-1]}], [step {step}]: time_start={t_start}, time_end={t_end}, \n input: {prompt}")

                    try:
                        resp = get_model_response(model, clip_file, prompt) or ""
                        pred = _letter_from_response(resp)
                        correct = (pred == ans_letter)
                        probe_seq.append({
                            "step": step,         # 0=V1, 1=V2, ...
                            "start_time": t_start,
                            "end_time": t_end,
                            "pred": pred,
                            "gold": ans_letter,
                            "correct": correct,
                            "raw_response": resp.strip()
                        })
                    except RuntimeError as e:
                        msg = str(e)
                        if ("CUDA out of memory" in msg) or ("CUDA error: out of memory" in msg):
                            probe_seq.append({
                                "step": step,
                                "end_time": t_end,
                                "error": f"CUDA OOM at end_time={t_end}s"
                            })
                            break
                        else:
                            probe_seq.append({
                                "step": step,
                                "end_time": t_end,
                                "error": f"inference failed: {repr(e)}"
                            })
                            break
                    finally:
                        # 删除临时视频
                        try:
                            os.remove(clip_file)
                        except Exception:
                            pass
                    print("current answer is ",ans_letter)
                    print("[ANSWER] is ", correct)
                    # 记录“由真转假”的最早时刻
                    if step == 0 and not correct:
                        # V1 就错了，不是我们要找的类型，但仍记录完整序列
                        pass
                    elif step > 0:
                        prev_correct = probe_seq[step - 1].get("correct", False)
                        if prev_correct and (not correct) and flipped_at is None:
                            flipped_at = {"step": step, "end_time": t_end}
                if not context:
                    context += "Here are the contextual information related to the video. Please answer the questions based on the contextual information: "
                context += f"At timestamp {q['time_stamp']}, the following question and answer occurred: Question: {ques}; Options: {options[0]}, {options[1]}, {options[2]}, {options[3]}; Answer: {q['answer']}; "

                # 汇总本题的探测
                record = {
                    "video_path": video_path,
                    "question": ques,
                    "time_stamp": q["time_stamp"],
                    "gold": ans_letter,
                    "options": options,
                    "sequence": probe_seq,
                    "initial_correct": (probe_seq[0].get("correct", False) if probe_seq else False),
                    "flipped_true_to_false": flipped_at is not None,
                    "flip_point": flipped_at  # None 或 {"step":..., "end_time":...}
                }
                results["probes"].append(record)

                # 若确实发生了 True->False，加入清单以打印摘要
                if record["initial_correct"] and flipped_at is not None:
                    flips.append({
                        "video": os.path.basename(video_path),
                        "q_time": q["time_stamp"],
                        "flip_step": flipped_at["step"],
                        "flip_end_time": flipped_at["end_time"],
                        "question": ques
                    })

    # 写出 JSON 报告
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # 控制台摘要
    print("\n=== Single Video ===")
    print(f"Model: {results['model']}")
    print(f"Total probed items: {len(results['probes'])}")
    print(f"True→False flips: {len(flips)}")
    if flips:
        print("First few flips:")
        for item in flips[:20]:
            print(f" - {item['video']} @ {item['q_time']} | flip at step {item['flip_step']} (end={item['flip_end_time']}s)")
    else:
        print("No True→False flips detected under current settings.")

def run_cross_video(
    data,
    model,
    output_path,
    context_time=0,      # 与现有 SQA 的时间窗一致：每轮仍按 [ts-context_time, ts] 裁剪
    only_first_question=True,   # True: 每个子视频用其Q1；False: 你也可指定其它索引
    end_time_cap=None
):
    """
    多视频多轮对同一问题的记忆一致性探测：
    - 维持“问题/选项/提示词”完全一致（prompt 固定不变）
    - V1 使用第一个子视频的 Q1（或指定问题），得到 result_1
    - V2 使用第二个子视频，但问题保持不变（同一文本与选项），得到 result_2
    - ...
    观察是否从 True → False 翻转
    注：仍使用非流式接口 get_model_response；“记忆效应”由 *累计文本 context* 模拟（可按需关闭）。
    """
    results = {
        "model": model.name(),
        "context_time": context_time,
        "probes": []
    }

    # 遍历每组视频（data 的元素通常是一组同源子视频/片段）
    for video_group in tqdm.tqdm(data):
        # 组内的子视频顺序即 V1, V2, ..., Vk
        subsets = video_group if isinstance(video_group, list) else [video_group]
        if len(subsets) == 0:
            continue

        # 选择“对齐的问题”：默认各子视频都取各自的 Q1，并以第一个子视频的 Q1 文本为“固定问题”
        def pick_question(subset):
            qs = subset.get("questions", [])
            return qs[0] if (qs and only_first_question) else (qs[0] if qs else None)

        # 先从第一个子视频拿到“固定问题”和“固定选项”
        first_q = pick_question(subsets[0])
        if first_q is None:
            continue

        fixed_question_text = first_q["question"]
        fixed_options = first_q["options"]
        # 规范化选项标号
        if not fixed_options[0].startswith("A."):
            fixed_options = [f"A. {fixed_options[0]}", f"B. {fixed_options[1]}", f"C. {fixed_options[2]}", f"D. {fixed_options[3]}"]
        # 固定 prompt，并做一个指纹，确保后续不会改变
        fixed_prompt = PROMPT_TEMPLATE.format("", fixed_question_text, *fixed_options)
        prompt_sig = hash(fixed_prompt)

        # 记录正确答案（以第一个子视频标注为准）
        correct_answer = (first_q.get("answer", "") or "").strip().upper()

        # 用于“模拟记忆”的文本 context：可改为 "" 禁用
        # context_text = "We will repeatedly ask the SAME question across multiple related videos. Consider all prior clips as context. "
        context_text = ""
        # V1..V(k-1)：跨视频逐轮
        probe_seq = []
        flipped_at = None

        def _slice_and_ask(subset, extra_end_shift=0, start_from_ts=True):
            """在给定子视频上，用固定问题进行一次推理。"""
            video_path = subset["video_path"]
            timestamp = subset["questions"][0]["time_stamp"]
            ts = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(":"))))
            # 时间裁剪逻辑保持一致：仍取 [ts-context_time, ts(+shift)]
            if context_time > 0:
                time_start = max(0, ts - context_time)
            else:
                time_start = 0
            t_end = ts + max(0, extra_end_shift)
            if end_time_cap is not None:
                t_end = min(t_end, end_time_cap)

            clip = split_video(video_path, time_start, t_end if t_end > 0 else ts)
            # 复用固定 prompt，并断言未被更改
            assert hash(fixed_prompt) == prompt_sig, "Fixed prompt mutated unexpectedly."
            inp = PROMPT_TEMPLATE.format(context_text, fixed_question_text, *fixed_options)
            resp = get_model_response(model, clip, inp) or ""
            try:
                os.remove(clip)
            except Exception:
                pass
            # 解析 ABCD
            pred = None
            for ch in resp.strip():
                up = ch.upper()
                if up in ("A","B","C","D"):
                    pred = up
                    break
            correct = (pred == correct_answer)
            print("current answer is correct?",correct)
            return {
                "video_path": video_path,
                "time_start": time_start,
                "time_end": (t_end if t_end > 0 else ts),
                "pred": pred,
                "gold": correct_answer,
                "correct": correct,
                "raw_response": resp.strip()
            }

        # 处理所有子视频（V1..V_{k}）
        for idx, subset in enumerate(subsets):
            rec = _slice_and_ask(subset, extra_end_shift=0)
            rec.update({"round": idx + 1, "type": "cross-video"})
            probe_seq.append(rec)
            # 累计到 context，保持“同一问题，但视觉在变”的会话语义
            context_text += f"[Round {idx+1}] On clip-{os.path.basename(subset['video_path'])}, you answered {rec.get('pred')}. "
                        
            # 早停逻辑：如果不是第一轮，且前一轮正确但当前错误，则记录翻转并跳出循环
            if idx > 0:
                prev_correct = probe_seq[idx - 1].get("correct", False)
                curr_correct = probe_seq[idx].get("correct", False)
                if prev_correct and (not curr_correct) and flipped_at is None:
                    flipped_at = {"round": idx + 1}
                    break

        # 汇总一组
        initial_correct = (probe_seq[0].get("correct", False) if probe_seq else False)
        record = {
            "fixed_question": fixed_question_text,
            "fixed_options": fixed_options,
            "gold": correct_answer,
            "sequence": probe_seq,
            "initial_correct": initial_correct,
            "flipped_true_to_false": flipped_at is not None,
            "flip_point": flipped_at
        }
        results["probes"].append(record)

    # 写结果
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Cross-Video ===")
    print(f"Model: {results['model']}")
    print(f"Groups probed: {len(results['probes'])}")
    flips = sum(1 for r in results["probes"] if r.get("flipped_true_to_false"))
    print(f"True→False flips: {flips}")
    print(f"Saved JSON: {output_path}")
