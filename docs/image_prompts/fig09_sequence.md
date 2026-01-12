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

FIGURE 9: Query Flow Sequence

Technical sequence diagram on dark background (#0d1117). Four vertical swim lanes labeled "User", "Embedder", "HAT Index", "Ollama". Horizontal arrows showing flow: "1. Query" from User to Embedder, "2. Embed (5ms)" to HAT Index, "3. Beam Search (0.51ms)", "4. Top-K Context" to Ollama, "5. Generate", "6. Response" back to User. Bottom: "~500ms end-to-end". Clean UML-style, time flows downward.
