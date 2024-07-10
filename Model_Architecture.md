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

### 2. Experimental Setting of LLM Models
