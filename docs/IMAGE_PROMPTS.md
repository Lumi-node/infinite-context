# Image Prompts for Infinite Context

Technical documentation visuals for the HAT (Hierarchical Attention Tree) paper and GitHub repository.

---

## Global Context (Prepend to ALL prompts)

Copy this block before every image generation prompt for visual consistency:

```
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
```

---

## 1. Hero Figure - HAT Architecture Overview

**Use:** README hero, paper Figure 1

**Prompt:**
```
Technical diagram on dark background (#0d1117). Left side: input arrow with label "Query". Center: three stacked horizontal rounded rectangles connected by vertical arrows. Top box labeled "Embed (sentence-transformers)", middle box labeled "HAT Beam Search" with cyan highlight, bottom box labeled "Context Injection". Right side: output arrow with label "Response". Below the diagram: stats bar reading "11M+ tokens | 0.51ms | 100% accuracy". Clean vector style, thin white lines, cyan accent color for highlights. Minimal, no decorations.
```

---

## 2. HAT Tree Structure Diagram

**Use:** "How It Works" section, paper Figure 2

**Prompt:**
```
Technical tree diagram on dark background (#0d1117). Single root node at top labeled "Index". Three large circular nodes on second level connected to root, each labeled "Session". Each session connects to 3-4 medium circular nodes labeled "Document". Each document connects to 5-6 small circular nodes labeled "Chunk". All connections are thin straight lines. Top level cyan, middle level lighter blue, bottom level white. Side annotation: "O(log n) beam search". Clean geometric style.
```

---

## 3. Beam Search Visualization

**Use:** Algorithm explanation, paper Figure 3

**Prompt:**
```
Technical diagram on dark background (#0d1117). Three horizontal rows of circles showing beam search pruning. Top row labeled "Sessions (n=847)": 8 circles, 3 highlighted in cyan, 5 grayed out. Middle row labeled "Documents (n=12,340)": 12 circles beneath highlighted sessions, 3 highlighted, rest grayed. Bottom row labeled "Chunks (n=375,000)": 15 small circles, top 5 highlighted with cyan connecting to output arrow. Right side annotations: "beam_width=10", "top-k=5". Bottom text: "Total comparisons: ~35 vs 375,000 (brute force)". Clean vector style.
```

---

## 4. Complexity Comparison Chart

**Use:** Performance section, paper Figure 4

**Prompt:**
```
Technical line graph on dark background (#0d1117). X-axis labeled "Indexed Tokens" from 0 to 10M. Y-axis labeled "Query Latency (ms)" from 0 to 500ms. Orange/red line curves steeply upward labeled "Flat Index O(n)". Cyan line stays nearly flat along bottom labeled "HAT O(log n)". Data point callout on cyan line: "11M tokens: 0.51ms". Subtle grid lines. Clean minimal chart style.
```

---

## 5. Accuracy Comparison Bar Chart

**Use:** Benchmark section, paper Table 1 visualization

**Prompt:**
```
Horizontal bar chart on dark background (#0d1117). Title: "Retrieval Accuracy on Multi-Session Benchmark". Three horizontal bars. Top bar labeled "HAT (hierarchical)" reaches 100% mark in cyan, showing "100%". Middle bar labeled "RAG (semantic)" reaches 70% mark in gray, showing "~70%". Bottom bar labeled "Recent-K (naive)" reaches 45% mark in darker gray, showing "~45%". Subtle grid lines at 25%, 50%, 75%, 100% marks. Clean minimal style.
```

---

## 6. Centroid Update Diagram

**Use:** Technical deep-dive, paper Section 3.2

**Prompt:**
```
Technical diagram on dark background (#0d1117). Three-level tree structure. Bottom level shows a new node being added with a "+" symbol, labeled "Chunk embedding". Dotted cyan arrows flow upward to parent node labeled "Document centroid", then to grandparent labeled "Session centroid". Each node shows a small vector representation (3-4 horizontal bars). Formula displayed: "c_new = (c_old * (n-1) + e_new) / n". Note at bottom: "Incremental O(1) update". Clean technical illustration style.
```

---

## 7. Context Window Extension Diagram

**Use:** Stats highlight, social media

**Prompt:**
```
Technical diagram on dark background (#0d1117). Left side: small rectangle labeled "8K native context". Center: large transformation arrow with "1,413x" multiplier. Right side: very large rectangle labeled "11.3M effective context" (represented as nested grid showing scale). Bottom label: "gemma3:1b + HAT". Clean geometric style, cyan accents.
```

---

## 8. System Components Diagram

**Use:** Architecture documentation, README

**Prompt:**
```
Technical block diagram on dark background (#0d1117). Four rounded rectangles in 2x2 grid connected by bidirectional arrows. Top-left: "Ollama (LLM)". Top-right: "sentence-transformers (Embeddings)". Bottom-left: "HAT Index (Memory)". Bottom-right: "CLI/API (Interface)". Center hub label: "InfiniteContext". Each block different shade of blue/cyan. Clean software architecture diagram style.
```

---

## 9. Query Flow Sequence Diagram

**Use:** Technical documentation, paper Figure 5

**Prompt:**
```
Technical sequence diagram on dark background (#0d1117). Four vertical swim lanes labeled "User", "Embedder", "HAT Index", "Ollama". Horizontal arrows showing flow: "1. Query" from User to Embedder, "2. Embed (5ms)" to HAT Index, "3. Beam Search (0.51ms)", "4. Top-K Context" to Ollama, "5. Generate", "6. Response" back to User. Bottom: "~500ms end-to-end". Clean UML-style, time flows downward.
```

---

## 10. Memory File Format Diagram

**Use:** Developer documentation

**Prompt:**
```
Technical diagram on dark background (#0d1117). Vertical stack of rectangles representing .hat file structure. Top section (small): "Header (4 bytes)". Middle section: "JSON Metadata". Bottom section (large): "Serialized Sessions → Documents → Chunks". Byte offset markers on left side. Title: ".hat binary format". Note: "Portable across models". Binary/hex aesthetic with clean lines.
```

---

## 11. Benchmark Results Dashboard

**Use:** Benchmark section, paper results

**Prompt:**
```
Technical dashboard on dark background (#0d1117). 2x2 grid of panels. Top-left: bar chart titled "Query Latency" showing bars for 100K, 200K, 375K chunks. Top-right: line chart titled "Scaling Curve" showing tokens vs time. Bottom-left: large stat display "375K chunks indexed in 37s". Bottom-right: table showing model comparison (gemma3:1b 1413x, phi4 706x, llama3.2 1413x). Clean data dashboard style.
```

---

## 12. Local-First Privacy Diagram

**Use:** Privacy messaging, README

**Prompt:**
```
Technical diagram on dark background (#0d1117). Center: computer icon labeled "Your machine". Inside dotted circle boundary: database icon labeled "Memory", brain icon labeled "LLM", document icons labeled "Data". Outside boundary: cloud icons with red X marks, labeled "No cloud required". Bottom tagline: "100% local, 100% private". Clean iconographic style, cyan accents.
```

---

## 13. Model Compatibility Matrix

**Use:** Compatibility section

**Prompt:**
```
Technical hub diagram on dark background (#0d1117). Central node labeled "HAT Index". Six surrounding nodes connected by cyan lines: "gemma3:1b (1413x)", "phi4 (706x)", "llama3.2 (1413x)", "mistral (1413x)", "qwen (706x)", "any model...". All connections highlighted to show universal compatibility. Clean network/hub diagram style.
```

---

## Generation Notes

**Naming Convention:**
```
fig01_hat_architecture.png
fig02_tree_structure.png
fig03_beam_search.png
fig04_complexity_comparison.png
fig05_accuracy_comparison.png
fig06_centroid_update.png
fig07_context_extension.png
fig08_system_components.png
fig09_query_flow.png
fig10_file_format.png
fig11_benchmark_dashboard.png
fig12_local_privacy.png
fig13_model_compatibility.png
```

**Dimensions:**
- README figures: 1200x600px (2:1 ratio)
- Paper figures: 1600x900px (16:9 ratio)
- Social/thumbnail: 1280x720px (16:9 ratio)

Place all images in `/docs/images/` directory.
