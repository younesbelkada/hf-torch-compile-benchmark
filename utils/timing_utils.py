import torch

from typing import Tuple
from tqdm import tqdm


def timing_cuda(
    model: torch.nn.Module,
    num_runs: int,
    inputs: torch.LongTensor,
    generation_config: "GenerationConfig" = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, int]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()

    for _ in tqdm(range(num_runs)):
        if generation_config is not None:
            _ = model.generate(inputs, generation_config=generation_config)
        else:
            kwargs = {}
            if model.config.is_encoder_decoder:
                shape = inputs.shape
                if model.config.model_type == "whisper":
                    shape = (inputs.shape[0], model.config.max_target_positions)
                
                kwargs["decoder_input_ids"] = torch.ones(shape, dtype=torch.long, device=inputs.device)
            _ = model(inputs, **kwargs)

    end_event.record()
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated(device)

    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_runs, max_memory
