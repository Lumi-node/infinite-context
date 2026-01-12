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

FIGURE 13: Model Compatibility

Technical hub diagram on dark background (#0d1117). Central node labeled "HAT Index". Six surrounding nodes connected by cyan lines: "gemma3:1b (1413x)", "phi4 (706x)", "llama3.2 (1413x)", "mistral (1413x)", "qwen (706x)", "any model...". All connections highlighted to show universal compatibility. Clean network/hub diagram style.
