# LLM Interview Questions

**430 technical interview questions with detailed answers for LLM Researcher, LLM Engineer, and AI/ML Engineer roles.**

Covers everything from transformer internals to production deployment — theory, math, code, and system design.

## Topics

| # | Topic | Questions | Key Areas |
|---|-------|-----------|-----------|
| 1 | **Core LLM Concepts** | Q1–Q86 | Prompting, chain-of-thought, transformer architecture, context windows, sampling parameters |
| 2 | **Fine-Tuning** | Q87–Q117 | LoRA, QLoRA, PEFT, RLHF, DPO, data preparation |
| 3 | **Mixture of Experts** | Q118–Q131 | MoE architecture, routing, load balancing, Mixtral, DeepSeek |
| 4 | **RAG** | Q132–Q154 | Retrieval pipelines, chunking, embeddings, reranking, evaluation |
| 5 | **Open-Source LLMs** | Q155–Q169 | Llama, Mistral, DeepSeek, Qwen, Gemma, Phi, model selection |
| 6 | **Inference Optimization** | Q170–Q191 | Quantization, KV cache, vLLM, FlashAttention, speculative decoding |
| 7 | **Attention Mechanisms** | Q192–Q203 | MHA, MQA, GQA, MLA, FlashAttention, linear attention |
| 8 | **Training Fundamentals** | Q204–Q224 | Pre-training, scaling laws, distributed training, data pipelines |
| 9 | **Multimodal Models** | Q225–Q236 | Vision-language models, ViT, LLaVA, video understanding |
| 10 | **Agent Architectures** | Q237–Q250 | Tool use, ReAct, LangChain/LangGraph, MCP, evaluation |
| 11 | **Evaluation & Benchmarks** | Q251–Q263 | MMLU, LMSYS Arena, LLM-as-judge, contamination |
| 12 | **Safety & Alignment** | Q264–Q276 | RLHF, constitutional AI, hallucination, red-teaming |
| 13 | **Tokenization** | Q277–Q287 | BPE, WordPiece, vocabulary tradeoffs, multilingual |
| 14 | **Production & Deployment** | Q288–Q300 | Serving, monitoring, caching, latency, cost optimization |
| 15 | **Applied Scenarios** | Q301–Q315 | Real-time AI systems, commentators, avatars, system design |
| 16 | **System Design** | Q316–Q331 | RAG design, scaling, cost optimization, on-premise deployment |
| 17 | **Voice Assistants & Speech AI** | Q332–Q365 | TTS, STT, Whisper, Pipecat, LiveKit, WebRTC, voice cloning |
| 18 | **Digital Avatars** | Q366–Q395 | Lip sync, FACS, talking heads, 3D rendering, NeRF, Gaussian Splatting |
| 19 | **Multi-Agent Systems** | Q396–Q430 | Orchestration, LangGraph, CrewAI, AutoGen, state management |

## Files

- **[`questions.md`](questions.md)** — All 430 questions organized by topic
- **[`answers.md`](answers.md)** — One-paragraph answer for each question with references to papers, tools, and frameworks

## How to Use

**For interview prep:** Start with the topics most relevant to your target role. Each answer includes references to key papers and tools for deeper study.

**For self-assessment:** Try answering each question before reading the answer. The questions progress from foundational to advanced within each topic.

**For interviewers:** Use as a question bank for screening LLM/AI candidates. Mix questions across topics for a balanced assessment.

## Key Papers Referenced

| Paper | Year | Topic |
|-------|------|-------|
| Attention Is All You Need (Vaswani et al.) | 2017 | Transformer architecture |
| LoRA (Hu et al.) | 2021 | Parameter-efficient fine-tuning |
| QLoRA (Dettmers et al.) | 2023 | Efficient fine-tuning of quantized LLMs |
| DPO (Rafailov et al.) | 2023 | Direct preference optimization |
| Chain-of-Thought (Wei et al.) | 2022 | Reasoning in LLMs |
| RAG (Lewis et al.) | 2020 | Retrieval-augmented generation |
| FlashAttention (Dao et al.) | 2022 | Efficient attention computation |
| RoPE (Su et al.) | 2021 | Rotary position embeddings |
| Constitutional AI (Bai et al.) | 2022 | AI alignment via principles |
| GQA (Ainslie et al.) | 2023 | Grouped-query attention |

## Contributing

Found an error, outdated info, or want to add questions? PRs welcome.

- Keep answers to one paragraph
- Include references to papers/tools where relevant
- Follow the existing numbering scheme

## License

MIT
