# LLM Benchmarks

*See commit date for last update.*

| LLM Model                                                                                    | Params       | Context    | Creator                                                 | [SWE-bench Verified](#swe-bench-verified) | [Terminal-Bench Hard](#terminal-bench-hard) | [τ²-Bench Telecom](#tau2-bench-telecom) | [BrowseComp](#browsecomp) | [AIME 2025](#aime-2025) | [GPQA Diamond](#gpqa-diamond) | [LiveCodeBench](#livecodebench) | [HLE](#hle) | [AA Intelligence Index](#aa-intelligence-index) |
| -------------------------------------------------------------------------------------------: | -----------: | ---------: | ------------------------------------------------------: | ----------------------------------------: | ------------------------------------------: | --------------------------------------: | ------------------------: | ----------------------: | ----------------------------: | ------------------------------: | ----------: | ----------------------------------------------: |
| [Devstral 2](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512)                 | `123B`       | `256k`     | [Mistral](https://mistral.ai/products/studio#models)    | 72.2                                      | 18                                          | 24.9                                    |                           | 37                      | 45                            | 33                              | **16**      | 22                                              |
| [Devstral Small 2](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)      | `24B`        | `256k`     | [Mistral](https://mistral.ai/products/studio#models)    | 68.0                                      | 16                                          | 23.4                                    |                           | 34                      | 35                            | 29                              | 13          | 19                                              |
| [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)                                | `30B-A3B`    | `200k`     | [Z.AI](https://docs.z.ai/guides/llm/glm-4.7)            | 59.2                                      | 22                                          | **99**                                  | **42.8**                  | 91.6                    | 75.2                          | 64                              | 14.4        | 30                                              |
| [K2 Think V2](https://huggingface.co/LLM360/K2-Think-V2)                                     | `70B`        | **`262k`** | [MBZUAI](https://www.k2think.ai/k2think)                |                                           | 7                                           | 25                                      |                           |                         | 71                            |                                 | 10          | 25                                              |
| [Qwen3-Coder-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | `30B-A3B`    | **`262k`** | [Alibaba](https://qwen.ai/)                             | 51.6                                      | 15                                          | 35                                      |                           | 29                      | 52                            | 40                              | 4           | 20                                              |
| [Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507)       | `30B-A3B`    | **`262k`** | [Alibaba](https://qwen.ai/)                             | 22.0                                      | 5                                           | 28.2                                    | 22.9                      | 56                      | 71                            | 33                              | 14          | 23                                              |
| [Qwen3 Next 80B A3B Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking)       | `80B-A3B`    | **`262k`** | [Alibaba](https://qwen.ai/)                             | **74.5**                                  | 10                                          | 42                                      |                           | 84                      | 76                            | **78**                          | 12          | 26                                              |
| [gpt-oss-20B (high)](https://huggingface.co/openai/gpt-oss-20b)                              | `117B-A5.1B` | `131k`     | [OpenAI](https://openai.com/index/introducing-gpt-oss/) | 34.0                                      | 10                                          | 60.2                                    | 28.3                      | 89                      | 78                            | 34                              | 7           | 25                                              |
| [gpt-oss-120B (high)](https://huggingface.co/openai/gpt-oss-120b)                            | `21B-A3.6B`  | `131k`     | [OpenAI](https://openai.com/index/introducing-gpt-oss/) | 62.4                                      | **24**                                      | 65.8                                    | 28.7                      | **93**                  | **88**                        | 39                              | 10          | **33**                                          |

## Sources
- https://artificialanalysis.ai/leaderboards/models?is_open_weights=open_source
- https://arxiv.org/html/2508.10925v1
- https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512
- https://huggingface.co/zai-org/GLM-4.7-Flash

## Legend
<a id="swe-bench-verified"></a>
### SWE-bench Verified
Human-validated subset of SWE-bench for fixing real GitHub issues; solutions are judged by running unit tests.  
https://www.swebench.com/SWE-bench/

<a id="terminal-bench-hard"></a>
### Terminal-Bench Hard (Agentic Coding & Terminal Use)
Hard subset of Terminal-Bench tasks; benchmark of real terminal tasks in sandboxed environments with test scripts.  
https://www.tbench.ai/

<a id="tau2-bench-telecom"></a>
### τ²-Bench Telecom (Agentic Tool Use)
Dual-control conversational agent benchmark where both agent and user use tools in a shared environment; includes a compositional task generator; Telecom domain.  
https://arxiv.org/abs/2506.07982

<a id="browsecomp"></a>
### BrowseComp
Browsing benchmark with 1,266 challenging short-answer questions designed to be hard to find but easy to verify.  
https://openai.com/index/browsecomp/

<a id="aime-2025"></a>
### AIME 2025 (Competition Math)
American Invitational Mathematics Examination; 15-question, 3-hour exam with integer answers from 0 to 999, using the 2025 exam set.  
https://maa.org/maa-invitational-competitions/

<a id="gpqa-diamond"></a>
### GPQA Diamond (Scientific Reasoning)
Graduate-level Google-proof multiple-choice science questions (biology, physics, chemistry) from GPQA; Diamond split used in evals.  
https://arxiv.org/abs/2311.12022

<a id="livecodebench"></a>
### LiveCodeBench (Coding)
Contamination-resistant coding benchmark that continuously collects new problems; evaluates code generation, self-repair, code execution, and test output prediction.  
https://livecodebench.github.io/

<a id="hle"></a>
### HLE (Reasoning & Knowledge)
**Humanity's Last Exam**: multi-modal benchmark with 2,500 expert-level questions across many subjects, designed to resist internet retrieval.  
https://lastexam.ai/

<a id="aa-intelligence-index"></a>
### AA Intelligence Index
Artificial Analysis composite index combining multiple evaluations (including τ²-Bench Telecom, Terminal-Bench Hard, GPQA Diamond, and HLE) to synthesize overall model capability.  
https://artificialanalysis.ai/methodology/intelligence-benchmarking
