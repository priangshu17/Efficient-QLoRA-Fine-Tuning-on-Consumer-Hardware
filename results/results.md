## Results

### Experimental Setup Recap

We evaluate FP16 LoRA and NF4-based QLora configurations with LoRA ranks r = {8, 16, 32} on the GSM8K dataset using a single consumer-grade GPU (RTX4050, 6GB VRAM). All configurations use identical data splits, tokenization, and training schedules, enabling controlled comparison of wall-clock time, memory usage, and validation perplexity.

### 1. Memory Feasibility and Wall-Clock Performance

![1767024312353](image/results/1767024312353.png)

The memory-time plot immediately reveals a stark separation between FP16 and NF4 configurations.

FP16 LoRA requires approximately 15-16 GB of peak allocated memory, far exceeding the physical VRAM capacity of the GPU. As a result, training proceeds via heavy memory oversubscription and host-device paging, leading to an extreme wall-clock training time of ~31 hours for a single run.

In contrast, all NF4 configurations remain will within GPU memory limits (~5-10GB peak allocation) and complete training in 2-3 hours, representing an order-of-magnitude reduction in wall-clock time. This demonstrates that FP16 fine-tuning of 7B-scale models is not merely slower but infeasible on constrained hardware, whereas NF4 enables stable and efficient training.

### 2. Quality-Time Trade-off Across Quantization Strategies

![1767024570915](image/results/1767024570915.png)

The perplexity-time plot highlights the central efficiency trade-off.

FP16 achieves the highest validation perplexity but at an extreme computational cost. NF4 configurations achieve comparable perplexity levels with dramatically reduced training time. In particular:

* NF4 r=8 completes fastest but noticeably bad perplexity.
* NF4 r=16 achieves a substantial improvement in perplexity with only a modest increase in training time.
* NF4 r=32 yields marginal additional gains while incurring the same cost as compared to the other NF4 settings.

Importantly, NF4 r=8 approaches FP16-level perplexity more than 10x faster.

### 3. Effect of LoRA Rank Under NF4 Quantization

![1767024986947](image/results/1767024986947.png)

When restricting attention to NF4 Configurations, validation perplexity improves monotonically with LoRA rank, but with clear diminishing returns.

* Increasing rank from r=8 to r=16 yields a large perplexity improvement.
* Increasing rank from r=16 to r=32 yields only marginal gains, despite a substantial increase in training time.

This indicates a clear "knee" in the trade-off curve at r=16, where representational capacity and computational efficiency are best balanced.

### 4. Summary of Findings

Across all experiments, three consistent conclusions emerge:

1. FP16 fine-tuning is impractical for 7B-scale models on 6GB GPUs due to memory oversubscription and pathological slowdowns.
2. NF4-based QLoRA dramatically improves feasibility and speed, enabling stable fine-tuning within tight memory budgets.
3. LoRA rank r=16 represents a sweet spot, achieving near-optimal perplexity with substantially lower wall-clock cost than both lower and higher ranks.
