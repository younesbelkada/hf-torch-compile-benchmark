import os
import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
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
        "--use-half",
        action="store_true",
        help="Use half precision.",
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
    if getattr(model_config, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if args.use_half else torch.float32,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if args.use_half else torch.float32,
        )

    if args.use_cpu:
        model = model.to("cpu")
    else:
        model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (args.batch_size, args.max_num_tokens),
        dtype=torch.long,
        device=model.device,
    )
    if args.run_generate:
        generation_config = GenerationConfig(
            max_new_tokens=args.max_num_tokens,
            pad_token_id=tokenizer.pad_token_id,
            # TODO: add more args
        )
    else:
        generation_config = None

    # warmup
    _ = timing_cuda(
        model=model,
        num_runs=2,
        input_ids=input_ids,
        generation_config=generation_config,
        device=model.device,
        forward_only=args.forward_only,
    )

    # real timing
    hf_time, hf_max_memory = timing_cuda(
        model=model,
        num_runs=args.num_runs,
        input_ids=input_ids,
        generation_config=generation_config,
        device=model.device,
        forward_only=args.forward_only,
    )

    model = model.to_bettertransformer()
    model = torch.compile(model, mode=args.compile_mode, fullgraph=True)

    # warmup
    _ = timing_cuda(
        model=model,
        num_runs=2,
        input_ids=input_ids,
        generation_config=generation_config,
        device=model.device,
        forward_only=args.forward_only,
    )

    # real timing
    compile_time, compile_max_memory = timing_cuda(
        model=model,
        num_runs=args.num_runs,
        input_ids=input_ids,
        generation_config=generation_config,
        device=model.device,
        forward_only=args.forward_only,
    )

    if os.path.isfile(args.output_file):
        with open(args.output_file, "r") as f:
            header = f.readline()
        if (
            header
            != "pt_version,model_name,compile_mode,num_runs,batch_size,max_num_tokens,run_generate,use_cuda,use_half,hf_time,hf_max_memory,compile_time,compile_max_memory\n"
        ):
            raise ValueError("Output file exists but has incorrect header")
    else:
        with open(args.output_file, "w") as f:
            f.write(
                "pt_version,model_name,compile_mode,num_runs,batch_size,max_num_tokens,run_generate,use_cuda,use_half,hf_time,hf_max_memory,compile_time,compile_max_memory\n"
            )

    with open(args.output_file, "a") as f:
        mode = "forward" if args.forward_only else "generate"
        f.write(
            f"{torch.__version__},{args.model_name},{args.compile_mode},{args.num_runs},{args.batch_size},{args.max_num_tokens},{args.run_generate},{args.use_cuda},{args.use_half},{hf_time},{hf_max_memory},{compile_time},{compile_max_memory}\n"
        )
