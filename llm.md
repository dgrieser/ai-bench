# LLM Benchmarks

*See commit date for last update.*

| LLM Model                                                                                    | Date         | Params       | Context    | Creator                                                   | [SWE-bench Verified](#swe-bench-verified) | [Terminal-Bench Hard](#terminal-bench-hard) | [Terminal-Bench 2.0](#terminal-bench-2-0) | [τ²-Bench Telecom](#tau2-bench-telecom) | [BrowseComp](#browsecomp) | [AIME 2025](#aime-2025) | [AIME 2026](#aime-2026) | [GPQA Diamond](#gpqa-diamond) | [LiveCodeBench](#livecodebench) | [SciCode](#scicode) | [HLE](#hle) | [AA Intelligence Index](#aa-intelligence-index) | [AA Coding Index](#aa-coding-index) |
| -------------------------------------------------------------------------------------------: | -----------: | -----------: | ---------: | --------------------------------------------------------: | ----------------------------------------: | ------------------------------------------: | ----------------------------------------: | --------------------------------------: | ------------------------: | ----------------------: | ----------------------: | ----------------------------: | ------------------------------: | ------------------: | ----------: | ----------------------------------------------: | ----------------------------------: |
| [devstral-2](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512)                 | `2026-02-03` | `123B`       | **`256k`** | [Mistral](https://mistral.ai/products/studio#models)      | 72.2                                      | 18.9                                        | 32.6                                      | 24.9                                    |                           | 36.7                    |                         | 59.4                          | 44.8                            | 33.1               | 3.6         | 22.0                                            | 23.7                                |
| [devstral-small-2](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)      | `2026-02-03` | `24B`        | **`256k`** | [Mistral](https://mistral.ai/products/studio#models)      | 68.0                                      | 16.7                                        | 22.5                                      | 23.4                                    |                           | 34.3                    |                         | 53.2                          | 34.8                            | 28.8               | 3.4         | 19.3                                            | 20.7                                |
| [glm-4-7-flash](https://huggingface.co/zai-org/GLM-4.7-Flash)                                | `2026-02-03` | `30B-A3B`    | `200k`     | [Z.AI](https://docs.z.ai/guides/llm/glm-4.7)              | 59.2                                      | 22                                          |                                           | **98.8**                                | 42.8                      | 91.6                    |                         | 58.1                          | 64                              | 33.7               | 7.1         | 30.1                                            | 25.9                                |
| [step-3-5-flash-reasoning](https://huggingface.co/stepfun-ai/Step-3.5-Flash)                 | `2026-02-03` | `196B-A11B`  | **`256k`** | [StepFun](https://static.stepfun.com/blog/step-3.5-flash) | 74.4                                      | 51.0                                        | 51.0                                      | 88.2                                    | 69.0                      | **97.3**                | 96.67                   | 83.5                          | 86.4                            |                    | 23.1        |                                                 |                                     |
| [minimax-m2-5](https://www.minimax.io/news/minimax-m25/)                                     | `2026-02-13` | `230B-A10B`  | `205k`     | [MiniMax](https://www.minimax.io/)                        | **80.2**                                  | 34.8                                        | 51.7                                      | 97.8                                    | 76.3                      | 86.3                    |                         | 84.8                          |                                 | **42.6**           | 19.1        | 42.0                                            | 37.4                                |
| [k2-think-v2](https://huggingface.co/LLM360/K2-Think-V2)                                     | `2026-02-03` | `70B`        | **`256k`** | [MBZUAI](https://www.k2think.ai/k2think)                  |                                           | 6.8                                         |                                           | 25.4                                    |                           | 90.42                   |                         | 71.3                          |                                 | 33.0               | 9.5         | 24.5                                            | 15.5                                |
| [qwen3-coder-30b-a3b-instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)     | `2026-02-03` | `30B-A3B`    | **`256k`** | [Alibaba](https://qwen.ai/)                               | 51.6                                      | 15.2                                        | 31.3                                      | 34.5                                    |                           | 29.0                    |                         | 51.6                          | 40.3                            | 27.8               | 4.0         | 20.0                                            | 19.4                                |
| [qwen3-coder-next](https://huggingface.co/Qwen/Qwen3-Coder-Next)                             | `2026-02-04` | `80B-A3B`    | **`256k`** | [Alibaba](https://qwen.ai/)                               | 70.6                                      | 18.2                                        | 36.2                                      | 79.5                                    |                           | 80                      |                         | 73.7                          | 47.5                            | 32.3               | 9.3         | 28.1                                            | 22.9                                |
| [qwen3-30b-a3b-2507-reasoning](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507)      | `2026-02-03` | `30B-A3B`    | **`256k`** | [Alibaba](https://qwen.ai/)                               | 22.0                                      | 5.3                                         |                                           | 28.1                                    | 22.9                      | 56.3                    |                         | 70.7                          | 70.7                            | 33.3               | 9.8         | 22.4                                            | 14.7                                |
| [qwen3-next-80b-a3b-reasoning](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking)      | `2026-02-03` | `80B-A3B`    | **`256k`** | [Alibaba](https://qwen.ai/)                               | 74.5                                      | 9.8                                         |                                           | 41.5                                    |                           | 84.3                    |                         | 75.9                          | 78.4                            | 38.8               | 11.7        | 26.5                                            | 19.5                                |
| [qwen3-5-397b-a17b](https://artificialanalysis.ai/models/qwen3-5-397b-a17b)                  | `2026-02-23` | `397B-A17B`  | **`256k`** | [Alibaba](https://qwen.ai/)                               | 76.4                                      | **40.9**                                    | **52.5**                                  | 95.6                                    | **78.6**                  |                         | 91.3                    | **89.3**                      | 83.6                            | 42.0               | **27.3**    | **45.0**                                        | **41.3**                            |
| [longcat-flash-lite](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite)              | `2026-02-23` | `68.5B-A3B`  | **`256k`** | [LongCat](https://longcat.chat/)                          | 54.4                                      | 33.75                                       |                                           | 72.8                                    |                           | 63.23                   |                         | 66.78                         |                                 |                    |             |                                                 |                                     |
| [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) (high)                              | `2026-01-23` | `21B-A3.6B`  | `131k`     | [OpenAI](https://openai.com/index/introducing-gpt-oss/)   | 34.0                                      | 10.6                                        | 3.1                                       | 60.2                                    | 28.3                      | 89.3                    |                         | 68.8                          | 77.7                            | 34.4               | 9.8         | 24.5                                            | 18.5                                |
| [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) (high)                            | `2026-01-23` | `117B-A5.1B` | `131k`     | [OpenAI](https://openai.com/index/introducing-gpt-oss/)   | 62.4                                      | **23.5**                                    | 18.7                                      | 65.8                                    | 28.7                      | 93.4                    |                         | **78.2**                      | **87.8**                        | 38.9               | **18.5**    | **33.3**                                        | **28.6**                            |

## Sources
- https://artificialanalysis.ai/leaderboards/models?is_open_weights=open_source
- https://arxiv.org/html/2508.10925v1
- https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512
- https://huggingface.co/zai-org/GLM-4.7-Flash
- https://huggingface.co/stepfun-ai/Step-3.5-Flash
- https://static.stepfun.com/blog/step-3.5-flash
- https://huggingface.co/meituan-longcat/LongCat-Flash-Lite
- https://longcat.chat/
- https://www.minimax.io/news/minimax-m25/
- https://www.tbench.ai/leaderboard/terminal-bench/2.0
- https://arxiv.org/pdf/2602.10604
- https://arxiv.org/abs/2407.13168
- https://huggingface.co/SamsungSAILMontreal/Qwen3-Coder-Next-REAP
- https://felloai.com/minimax-m2-5-model/
- https://huggingface.co/datasets/MathArena/aime_2026

## Legend
<a id="swe-bench-verified"></a>
### SWE-bench Verified
Human-validated subset of SWE-bench for fixing real GitHub issues; solutions are judged by running unit tests.  
https://www.swebench.com/SWE-bench/

<a id="terminal-bench-hard"></a>
### Terminal-Bench Hard (Agentic Coding & Terminal Use)
Hard subset of Terminal-Bench tasks; benchmark of real terminal tasks in sandboxed environments with test scripts.  
https://www.tbench.ai/

<a id="terminal-bench-2-0"></a>
### Terminal-Bench 2.0
Terminal-Bench 2.0 is a collection of tasks and an evaluation harness to help agent makers quantify their agents' terminal mastery.  
https://www.tbench.ai/leaderboard/terminal-bench/2.0

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
https://huggingface.co/datasets/MathArena/aime_2025

<a id="aime-2026"></a>
### AIME 2026 (Competition Math)
American Invitational Mathematics Examination; 15-question, 3-hour exam with integer answers from 0 to 999, using the 2026 exam set.  
https://maa.org/maa-invitational-competitions/
https://huggingface.co/datasets/MathArena/aime_2026

<a id="gpqa-diamond"></a>
### GPQA Diamond (Scientific Reasoning)
Graduate-level Google-proof multiple-choice science questions (biology, physics, chemistry) from GPQA; Diamond split used in evals.  
https://arxiv.org/abs/2311.12022

<a id="livecodebench"></a>
### LiveCodeBench (Coding)
Contamination-resistant coding benchmark that continuously collects new problems; evaluates code generation, self-repair, code execution, and test output prediction.  
https://livecodebench.github.io/

<a id="scicode"></a>
### SciCode (Coding)
Scientific coding benchmark evaluating code generation and problem solving in scientific domains.  
https://arxiv.org/abs/2407.13168

<a id="hle"></a>
### HLE (Reasoning & Knowledge)
**Humanity's Last Exam**: multi-modal benchmark with 2,500 expert-level questions across many subjects, designed to resist internet retrieval.  
https://lastexam.ai/

<a id="aa-intelligence-index"></a>
### AA Intelligence Index
Artificial Analysis composite index combining multiple evaluations (including τ²-Bench Telecom, Terminal-Bench Hard, GPQA Diamond, and HLE) to synthesize overall model capability.  
https://artificialanalysis.ai/methodology/intelligence-benchmarking

<a id="aa-coding-index"></a>
### AA Coding Index
Artificial Analysis composite index for coding performance combining multiple evaluations (including Terminal-Bench Hard, and SciCodeBench) to synthesize overall model capability.  
https://artificialanalysis.ai/methodology/intelligence-benchmarking
