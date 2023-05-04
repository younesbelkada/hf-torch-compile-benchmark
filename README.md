# hf-torch-compile-benchmark :fire:

A repository to benchmark the expected speedups using `torch.compile` and `torch.scaled_dot_product_attention`

## How to use it?

First, the project for now depends on the `main` branch of `transformers` and `optimum`, and you need at least `torch>=2.0`.

```bash
pip install -r requirements.txt
```

Use `main.py` to benchmark the speedups. Run `python main.py -h` to understand how to use the benchmarking script. The results will be added by default inside `output.csv` file in the current directory. If that file already exists, the results will be appended to it. 

Better to use a shell script to loop over various combination of hyper parameters (batch_size, seq_len, etc.) and append the results to the same file. 

## Benchmarks

### RTX4090



|pt_version             |model_name         |compile_mode   |batch_size|max_num_tokens|run_type    |precision    |hf_time|sdpa_no_compile_time|sdpa_compile_time|speedup_sdpa+compile|speedup_sdpa|problems                                |
|-----------------------|-------------------|---------------|----------|--------------|------------|-------------|-------|--------------------|-----------------|--------------------|------------|----------------------------------------|
|                       |                   |               |          |              |            |             |       |                    |                 |                    |            |                                        |
|2.0.0+cu118            |gpt2               |reduce-overhead|1         |256           |forward-only|torch.float16|0.00426|0.00274             |0.00126          |238.10%             |55.47%      |                                        |
|2.0.0+cu118            |gpt2               |reduce-overhead|8         |256           |forward-only|torch.float16|0.00819|0.00817             |0.00615          |33.17%              |0.24%       |                                        |
|2.0.0+cu118            |gpt2               |reduce-overhead|32        |256           |forward-only|torch.float16|0.03371|0.0316              |0.02269          |48.57%              |6.68%       |                                        |
|2.0.0+cu118            |gpt2               |reduce-overhead|1         |256           |forward-only|torch.float32|0.00431|0.00327             |0.00286          |50.70%              |31.80%      |                                        |
|2.0.0+cu118            |gpt2               |reduce-overhead|8         |256           |forward-only|torch.float32|0.01882|0.01907             |0.01633          |15.25%              |-1.31%      |                                        |
|2.0.0+cu118            |gpt2               |reduce-overhead|32        |256           |forward-only|torch.float32|0.08607|0.08528             |0.0616           |39.72%              |0.93%       |                                        |
|2.0.0+cu118            |t5-base            |reduce-overhead|1         |256           |forward-only|torch.float16|0.01098|0.01114             |0.00588          |86.73%              |-1.44%      |                                        |
|2.0.0+cu118            |t5-base            |reduce-overhead|8         |256           |forward-only|torch.float16|0.02174|0.02028             |0.01717          |26.62%              |7.20%       |                                        |
|2.0.0+cu118            |t5-base            |reduce-overhead|32        |256           |forward-only|torch.float16|0.0965 |0.07942             |0.07             |37.86%              |21.51%      |                                        |
|2.0.0+cu118            |t5-base            |reduce-overhead|1         |256           |forward-only|torch.float32|0.01473|0.01053             |0.00916          |60.81%              |39.89%      |                                        |
|2.0.0+cu118            |t5-base            |reduce-overhead|8         |256           |forward-only|torch.float32|0.03639|0.03587             |0.03302          |10.21%              |1.45%       |                                        |
|2.0.0+cu118            |t5-base            |reduce-overhead|32        |256           |forward-only|torch.float32|0.1413 |0.14658             |0.13006          |8.64%               |-3.60%      |                                        |
|                       |                   |               |          |              |            |             |       |                    |                 |                    |            |                                        |
|                       |                   |               |          |              |            |             |       |                    |                 |                    |            |                                        |
|2.0.0+cu118            |gpt2               |reduce-overhead|1         |256           |generate    |torch.float16|0.87543|0.67787             |0.6773           |29.25%              |29.14%      |<- probably also problems with k/v cache|
|2.0.0+cu118            |gpt2               |reduce-overhead|8         |256           |generate    |torch.float16|0.93707|0.78795             |0.7868           |19.10%              |18.93%      |<- probably also problems with k/v cache|
|2.0.0+cu118            |gpt2               |reduce-overhead|32        |256           |generate    |torch.float16|1.22092|0.85482             |0.85341          |43.06%              |42.83%      |<- probably also problems with k/v cache|
|2.0.0+cu118            |gpt2               |reduce-overhead|1         |256           |generate    |torch.float32|0.90596|0.66562             |0.66414          |36.41%              |36.11%      |<- probably also problems with k/v cache|
|2.0.0+cu118            |gpt2               |reduce-overhead|8         |256           |generate    |torch.float32|0.97111|0.82092             |0.82009          |18.42%              |18.30%      |<- probably also problems with k/v cache|
|2.0.0+cu118            |gpt2               |reduce-overhead|32        |256           |generate    |torch.float32|1.54068|1.36056             |1.36055          |13.24%              |13.24%      |<- probably also problems with k/v cache|
|2.0.0+cu118            |t5-base            |reduce-overhead|1         |256           |generate    |torch.float16|0.58538|0.59828             |0.59933          |-2.33%              |-2.16%      |<- problems with k/v cache              |
|2.0.0+cu118            |t5-base            |reduce-overhead|8         |256           |generate    |torch.float16|0.64183|0.65154             |0.65241          |-1.62%              |-1.49%      |<- problems with k/v cache              |
|2.0.0+cu118            |t5-base            |reduce-overhead|32        |256           |generate    |torch.float16|0.67339|0.67719             |0.67812          |-0.70%              |-0.56%      |<- problems with k/v cache              |
|2.0.0+cu118            |t5-base            |reduce-overhead|1         |256           |generate    |torch.float32|0.49707|0.53413             |0.53497          |-7.08%              |-6.94%      |<- problems with k/v cache              |
|2.0.0+cu118            |t5-base            |reduce-overhead|8         |256           |generate    |torch.float32|0.54263|0.5798              |0.58015          |-6.47%              |-6.41%      |<- problems with k/v cache              |
|2.0.0+cu118            |t5-base            |reduce-overhead|32        |256           |generate    |torch.float32|0.60027|0.64                |0.63894          |-6.05%              |-6.21%      |<- problems with k/v cache              |
|                       |                   |               |          |              |            |             |       |                    |                 |                    |            |                                        |
|                       |                   |               |          |              |            |             |       |                    |                 |                    |            |                                        |
|2.1.0.dev20230504+cu118|gpt2               |reduce-overhead|1         |256           |forward-only|torch.float16|0.00426|0.003               |0.05625          |-92.43%             |42.00%      |<- problems with torch compile          |
|2.1.0.dev20230504+cu118|gpt2               |reduce-overhead|8         |256           |forward-only|torch.float16|0.00838|0.00842             |0.06906          |-87.87%             |-0.48%      |<- problems with torch compile          |
|2.1.0.dev20230504+cu118|gpt2               |reduce-overhead|32        |256           |forward-only|torch.float16|0.03402|0.03188             |0.09866          |-65.52%             |6.71%       |<- problems with torch compile          |
|2.1.0.dev20230504+cu118|gpt2               |reduce-overhead|1         |256           |forward-only|torch.float32|0.00463|0.00345             |0.05444          |-91.50%             |34.20%      |<- problems with torch compile          |
|2.1.0.dev20230504+cu118|gpt2               |reduce-overhead|8         |256           |forward-only|torch.float32|0.01917|0.01945             |0.10395          |-81.56%             |-1.44%      |<- problems with torch compile          |
|2.1.0.dev20230504+cu118|gpt2               |reduce-overhead|32        |256           |forward-only|torch.float32|0.08936|0.08605             |0.13856          |-35.51%             |3.85%       |<- problems with torch compile          |
|2.1.0.dev20230504+cu118|t5-base            |reduce-overhead|1         |256           |forward-only|torch.float16|0.01189|0.0121              |0.00609          |95.24%              |-1.74%      |                                        |
|2.1.0.dev20230504+cu118|t5-base            |reduce-overhead|8         |256           |forward-only|torch.float16|0.02182|0.0203              |0.01801          |21.15%              |7.49%       |                                        |
|2.1.0.dev20230504+cu118|t5-base            |reduce-overhead|32        |256           |forward-only|torch.float16|0.09746|0.08032             |0.08354          |16.66%              |21.34%      |                                        |
|2.1.0.dev20230504+cu118|t5-base            |reduce-overhead|1         |256           |forward-only|torch.float32|0.01053|0.01124             |0.00929          |13.35%              |-6.32%      |                                        |
|2.1.0.dev20230504+cu118|t5-base            |reduce-overhead|8         |256           |forward-only|torch.float32|0.03523|0.03615             |0.03444          |2.29%               |-2.54%      |                                        |
|2.1.0.dev20230504+cu118|t5-base            |reduce-overhead|32        |256           |forward-only|torch.float32|0.13752|0.1446              |0.12911          |6.51%               |-4.90%      |                                        |
|2.1.0.dev20230504+cu118|huggyllama/llama-7b|reduce-overhead|1         |256           |forward-only|torch.float16|0.04381|0.02948             |0.23042          |-80.99%             |48.61%      |<- problems with torch compile          |
|2.1.0.dev20230504+cu118|EleutherAI/gpt-j-6b|reduce-overhead|1         |256           |forward-only|torch.float16|0.02678|0.0247              |0.19665          |-86.38%             |8.42%       |<- problems with torch compile          |

To reproduce:
- https://github.com/younesbelkada/hf-torch-compile-benchmark/blob/main/run_all_forward.sh
- https://github.com/younesbelkada/hf-torch-compile-benchmark/blob/main/run_all_generate.sh


## TODOs

- properly deal with attention masks
- script to nicely render the results
