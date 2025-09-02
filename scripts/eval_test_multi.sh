cd ../src

# Change the model name to the model you want to evaluate

EVAL_MODEL="FlashVstream"
Devices=1

# -1 means all context, i. e. (0, query_time); any integer t greater than 0 means (query_time - t, query_time)
CONTEXT_TIME=-1
SINGLE_VIDEO=0

# For sequential question answering(Offline + Text Instruction)

TASK="sqa"
DATA_FILE="./data/questions_${TASK}.json"
OUTPUT_FILE="./data/${TASK}_output_${EVAL_MODEL}_test.json"
BENCHMARK="StreamingSQA_test"
# 0 multiple video test or 1 single video test


if [ "$EVAL_MODEL" = "FlashVstream" ]; then
    CUDA_VISIBLE_DEVICES=$Devices python eval_test.py --model_name $EVAL_MODEL --benchmark_name $BENCHMARK --data_file $DATA_FILE --output_file $OUTPUT_FILE --context_time $CONTEXT_TIME --single_video $SINGLE_VIDEO 
fi


# (Streaming/Online + Text Instruction)
# Optional Task(real, omni, sqa)

# TASK="sqa"
# DATA_FILE="./data/questions_${TASK}_stream.json"
# OUTPUT_FILE="./data/${TASK}_text_stream_output_${EVAL_MODEL}_test.json"
# BENCHMARK="StreamingOpenStreamText_test"
# SINGLE_VIDEO=False

# if [ "$EVAL_MODEL" = "MiniCPM-V" ]; then
#     conda activate MiniCPM-V
#     CUDA_VISIBLE_DEVICES=$Devices python eval_test.py --model_name $EVAL_MODEL --benchmark_name $BENCHMARK --data_file $DATA_FILE --output_file $OUTPUT_FILE --context_time $CONTEXT_TIME --single_video $SINGLE_VIDEO
# fi

# if [ "$EVAL_MODEL" = "FlashVstream" ]; then
#     conda activate streamingbench
#     CUDA_VISIBLE_DEVICES=$Devices python eval_test.py --model_name $EVAL_MODEL --benchmark_name $BENCHMARK --data_file $DATA_FILE --output_file $OUTPUT_FILE --context_time $CONTEXT_TIME --single_video $SINGLE_VIDEO
# fi