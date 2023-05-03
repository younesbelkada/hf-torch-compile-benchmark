# hf-torch-compile-benchmark :fire:

A repository to benchmark the expected speedups using `torch.compile` and `torch.scaled_dot_product_attention`

## How to use it?

First, the project for now depends on the `main` branch of `transformers` and `optimum`, and you need at least `torch>=2.0`.

```bash
pip install -r requirements.txt
```

Use `main.py` to benchmark the speedups. Run `python main.py -h` to understand how to use the benchmarking script. The results will be added by default inside `output.csv` file in the current directory. If that file already exists, the results will be appended to it. 

Better to use a shell script to loop over various combination of hyper parameters (batch_size, seq_len, etc.) and append the results to the same file. 

## TODOs

- properly deal with attention masks
- script to nicely render the results