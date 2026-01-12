CONTEXT: Technical figure for "Infinite Context" - a research project enabling local LLMs to access 11M+ tokens of memory via Hierarchical Attention Trees (HAT). This is academic-grade open source software, similar in presentation to papers from NeurIPS/ICML or flagship GitHub projects like LangChain, Ollama, or Hugging Face Transformers.

VISUAL STYLE:
- Dark mode technical illustration (#0d1117 background)
- Accent: cyan/blue (#58a6ff primary, #1f6feb secondary)
- Text/lines: white (#f0f6fc) and gray (#8b949e)
- Clean vector aesthetic, no gradients or 3D effects
- Geometric precision, aligned grids, consistent spacing

WHAT THIS PROJECT IS:
- HAT = Hierarchical Attention Tree (3-level index: Session → Document → Chunk)
- Beam search retrieval: O(log n) complexity, 0.51ms latency at 3M tokens
- 100% retrieval accuracy on hierarchical conversation data
- Works with any Ollama model (Gemma, Phi, Llama, Mistral)
- Fully local, privacy-preserving, no API costs

---

FIGURE 5: Accuracy Comparison Bar Chart

Horizontal bar chart on dark background (#0d1117). Title: "Retrieval Accuracy on Multi-Session Benchmark". Three horizontal bars. Top bar labeled "HAT (hierarchical)" reaches 100% mark in cyan, showing "100%". Middle bar labeled "RAG (semantic)" reaches 70% mark in gray, showing "~70%". Bottom bar labeled "Recent-K (naive)" reaches 45% mark in darker gray, showing "~45%". Subtle grid lines at 25%, 50%, 75%, 100% marks. Clean minimal style.
