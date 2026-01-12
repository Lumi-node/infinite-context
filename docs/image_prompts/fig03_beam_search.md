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

FIGURE 3: Beam Search Visualization

Technical diagram on dark background (#0d1117). Three horizontal rows of circles showing beam search pruning. Top row labeled "Sessions (n=847)": 8 circles, 3 highlighted in cyan, 5 grayed out. Middle row labeled "Documents (n=12,340)": 12 circles beneath highlighted sessions, 3 highlighted, rest grayed. Bottom row labeled "Chunks (n=375,000)": 15 small circles, top 5 highlighted with cyan connecting to output arrow. Right side annotations: "beam_width=10", "top-k=5". Bottom text: "Total comparisons: ~35 vs 375,000 (brute force)". Clean vector style.
