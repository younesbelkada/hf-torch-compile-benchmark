import os
import argparse

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
)

from utils import timing_cuda


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of batches to run. The average time across these runs will be reported. Larger runs might give a better estimate of the average time, but it will take longer to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Input batch size.",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=256,
        help="`max_new_tokens` to generate. This argument is equivalent to the `max_new_tokens` argument in `model.generate`.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="The model name to use. Only decoder and encoder-decoder models are supported for now.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        help="The model compilation mode to use. Refer to the official tutorial of torch.compile: https://pytorch.org/tutorials//intermediate/torch_compile_tutorial.html for more details.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="output.csv",
        help="The output file to write results. If the file does not exist, it will be created. If the file exists, the results will be appended to the file.",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Use CUDA if available.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="torch.float32",
        help="Precision of torch dtype"
    )
    parser.add_argument(
        "--run-generate",
        action="store_true",
        help="Run the benchmarks using `generate` method.",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    model_config = AutoConfig.from_pretrained(args.model_name)
    model_cls = getattr(transformers, model_config.architectures[0])

    model = model_cls.from_pretrained(
        args.model_name,
        torch_dtype=eval(args.precision)
    )

    if args.use_cpu:
        model = model.to("cpu")
    else:
        model = model.to("cuda")

    shape = (args.batch_size, args.max_num_tokens)
    dtype = torch.long

    if model.config.model_type == "whisper":
        args.max_num_tokens = 2 * model.config.max_source_positions
        shape = (args.batch_size, model.config.num_mel_bins, args.max_num_tokens)
        dtype = eval(args.precision)

    inputs = torch.randint(
        0,
        model.config.vocab_size,
        shape,
        dtype=dtype,
        device=model.device,
    )
    if args.run_generate:
        generation_config = GenerationConfig(
            max_new_tokens=args.max_num_tokens,
            pad_token_id=0,
            eos_token_id=None,
            do_sample=False,
            num_beams=1,
            # TODO: add more args
        )
    else:
        generation_config = None

    # warmup
    _ = timing_cuda(
        model=model,
        num_runs=4,
        inputs=inputs,
        generation_config=generation_config,
        device=model.device,
    )

    # real timing
    hf_time, hf_max_memory = timing_cuda(
        model=model,
        num_runs=args.num_runs,
        inputs=inputs,
        generation_config=generation_config,
        device=model.device,
    )

    model = model.to_bettertransformer()

    # warmup
    _ = timing_cuda(
        model=model,
        num_runs=4,
        inputs=inputs,
        generation_config=generation_config,
        device=model.device,
    )

    # real timing
    sdpa_no_compile_time, no_compile_max_memory = timing_cuda(
        model=model,
        num_runs=args.num_runs,
        inputs=inputs,
        generation_config=generation_config,
        device=model.device,
    )

    model = torch.compile(model, mode=args.compile_mode, fullgraph=True)

    # warmup
    _ = timing_cuda(
        model=model,
        num_runs=4,
        inputs=inputs,
        generation_config=generation_config,
        device=model.device,
    )

    # real time
    sdpa_compile_time, compile_max_memory = timing_cuda(
        model=model,
        num_runs=args.num_runs,
        inputs=inputs,
        generation_config=generation_config,
        device=model.device,
    )

    full_header = "pt_version,model_name,compile_mode,batch_size,max_num_tokens,run_type,precision,hf_time,sdpa_no_compile_time,sdpa_compile_time\n"

    if os.path.isfile(args.output_file):
        with open(args.output_file, "r") as f:
            header = f.readline()
        if header != full_header:
            raise ValueError("Output file exists but has incorrect header")
    else:
        with open(args.output_file, "w") as f:
            f.write(full_header)

    with open(args.output_file, "a") as f:
        max_tokens = args.max_num_tokens
        run_type = "forward-only" if not args.run_generate else "generate"
        precision = str(model.dtype)

        f.write(
                f"{torch.__version__},{args.model_name},{args.compile_mode},{args.batch_size},{max_tokens},{run_type},{precision},{round(hf_time, 5)},{round(sdpa_no_compile_time, 5)},{round(sdpa_compile_time, 5)}\n"
        )
