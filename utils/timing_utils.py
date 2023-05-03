import torch

from typing import Tuple
from tqdm import tqdm


def timing_cuda(
    model: torch.nn.Module,
    num_runs: int,
    input_ids: torch.LongTensor,
    attention_masks: torch.FloatTensor = None,
    generation_config: "GenerationConfig" = None,
    device: torch.device = torch.device("cpu"),
    forward_only: bool = False,
) -> Tuple[float, int]:
    
    if attention_masks is None:
        attention_masks = torch.ones_like(input_ids)

    if getattr(model.config, "is_encoder_decoder", False):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "decoder_input_ids": input_ids if generation_config is None else None,
        }
    else:
        inputs = {
            "input_ids": input_ids, 
            "attention_mask": attention_masks,
        }

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()

    for _ in tqdm(range(num_runs)):
        if generation_config is not None:
            _ = model.generate(**inputs, generation_config=generation_config)
        else:
            _ = model(**inputs)

    end_event.record()
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated(device)

    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_runs, max_memory
