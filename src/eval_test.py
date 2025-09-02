from utils.data_execution import load_data

from model.modelclass import Model
from benchmark.Benchmark import Benchmark

import argparse

import os, sys, time

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main(args):
    log(f"Loading data from {args.data_file} ...")
    data = load_data(args.data_file)

    ####### BENCHMARK #######
    log("Initializing benchmark ...")
    benchmark = Benchmark(data)

    if args.benchmark_name == "StreamingSQA_test":
        from benchmark.StreamingBenchSQA_test import StreamingBenchSQA_test
        benchmark = StreamingBenchSQA_test(data)

    ##########################

    ####### MODEL ############
    log(f"Benchmark ready: {args.benchmark_name}")

    log("Preparing model wrapper ...")
    model = Model()
    log("Model wrapper created. Selecting concrete model ...")

    if args.model_name == "FlashVstream":

        from model.FlashVstream import FlashVstream
        model = FlashVstream()
        log("FlashVstream initialized.")
    ######################

    log("Starting evaluation ...")
    t0 = time.time()
    benchmark.eval(data, model, args.output_file, args.context_time, args.single_video, args.end_time_cap)
    log(f"Evaluation finished in {time.time() - t0:.1f}s. Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--benchmark_name", type=str, required=True, help="Name of the benchmark")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    parser.add_argument("--context_time", type=int, required=True, help="Time before the query")
    parser.add_argument("--single_video",type=int, required=True, help="Single video test(1) or not(0)")
    parser.add_argument("--end_time_cap", type=int, default=None, help="Absolute cap (seconds) for clip end time to limit memory usage")
    args = parser.parse_args()
    main(args)
