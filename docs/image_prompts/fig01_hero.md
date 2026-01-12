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

FIGURE 1: Hero - HAT Architecture Overview

Technical diagram on dark background (#0d1117). Left side: input arrow with label "Query". Center: three stacked horizontal rounded rectangles connected by vertical arrows. Top box labeled "Embed (sentence-transformers)", middle box labeled "HAT Beam Search" with cyan highlight, bottom box labeled "Context Injection". Right side: output arrow with label "Response". Below the diagram: stats bar reading "11M+ tokens | 0.51ms | 100% accuracy". Clean vector style, thin white lines, cyan accent color for highlights. Minimal, no decorations.
