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

FIGURE 6: Centroid Update Diagram

Technical diagram on dark background (#0d1117). Three-level tree structure. Bottom level shows a new node being added with a "+" symbol, labeled "Chunk embedding". Dotted cyan arrows flow upward to parent node labeled "Document centroid", then to grandparent labeled "Session centroid". Each node shows a small vector representation (3-4 horizontal bars). Formula displayed: "c_new = (c_old * (n-1) + e_new) / n". Note at bottom: "Incremental O(1) update". Clean technical illustration style.
