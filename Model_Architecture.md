## Table of Contents
1. [Model Architecture of LLM Models](#1.-Model-Architecture-of-LLM-Models)
2. [Experimental Setting of LLM Models](#1.-Experimental-Setting-of-LLM-Models)

### 1. Model Architecture of LLM Models 

|Feature|Mistral|LLaMA 2|LLaMA 3|
|:------:|:---:|:---:|:---:|
|Tokenizer|SentencePiece|SentencePiece|Tiktoken|
|Context Length|4096 tokens|4096 tokens|8192 tokens|
|Attention Mechanism|Sliding Window Attention|Grouped Query Attention|Grouped Query Attention|
|Activation Function|SwiGLU + Mixture of Experts|SwiGLU|SwiGLU|

#### 1-1. Tokenizer using Byte-Pair Encoding (BPE)
All LLM models use Byte-Pair Encoding (BPE), but Mistral and LLaMA 2 use SentencePiece and LLaMA 3 uses Tiktoken introduced by OpenAI. 

#### 1-2. Context Length Extension by Rotary Positional Embedding (RoPE) 


#### 1-3. Pre-Normalization : Root Mean Square (RMS) Normalization 

   Inspired from GPT-3, most LLM models normalize the input of each Transformer sub-layer, instead of normalizing the output.
  
* Why RMS Normalization instead of Layer Normalization?

   The reason behind using RMS Normalization (RMSNorm) over Layer Normalization (LayerNorm) is that RMSNorm has comparable performance but less computer overhead (reducing the running time by 7~64%). It simplifies LayerNorm by totally removing the mean statistic in LayerNorm. 

![image](https://github.com/domyown/LLM_Archive/assets/43026521/1bbe2e97-3c63-4164-a7ce-4d91b0cab876)

($a_i$ : activation of ith neuron, $g$ is the gain parameter used to re-scale the standardized summed inputs) 

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

#### 1-4. SwiGLU Activation Function

Inspired from PaLM, 

```python
class Swish(nn.Module):
    def __init__(self, train_beta=False):
        super().__init__()
        if train_beta:
            self.beta = Parameter(torch.Tensor([1.]))
        else:
            self.beta = 1.0

    @weak_script_method
    def forward(self, input):
        return input * torch.sigmoid(self.beta * input)
```

### 2. Experimental Setting of LLM Models

## Reference 
* https://github.com/FareedKhan-dev/Building-llama3-from-scratch?tab=readme-ov-file#implementing-rope
* https://akgeni.medium.com/llama-concepts-explained-summary-a87f0bd61964
