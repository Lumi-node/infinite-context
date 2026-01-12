#!/usr/bin/env python3
"""
Infinite Context - Quickstart Example

This shows how to give any local LLM unlimited memory in under 20 lines of code.

Prerequisites:
    pip install infinite-context
    ollama serve
    ollama pull gemma3:1b
"""

from infinite_context import InfiniteContext

# Initialize with your preferred Ollama model
ctx = InfiniteContext(model="gemma3:1b")

# Add information to memory (automatically embedded and indexed)
ctx.add("My name is Alex and I'm a software engineer.")
ctx.add("I'm working on a quantum computing project.")
ctx.add("The experiment yesterday showed 47% improvement in coherence time.")
ctx.add("We're using superconducting qubits at 15 millikelvin.")

# Start a new topic
ctx.new_topic()
ctx.add("For lunch today I had a great burrito from the food truck.")
ctx.add("The salsa verde was particularly good.")

# Chat - the model retrieves relevant context automatically
print("Question: What were the quantum experiment results?")
response = ctx.chat("What were the quantum experiment results?")
print(f"Answer: {response}\n")

print("Question: What did I have for lunch?")
response = ctx.chat("What did I have for lunch?")
print(f"Answer: {response}\n")

# Save memory for later
ctx.save("my_memory.hat")
print("Memory saved to my_memory.hat")

# Show stats
stats = ctx.stats()
print(f"\nMemory stats:")
print(f"  Chunks: {stats['chunks']}")
print(f"  Sessions: {stats['sessions']}")
print(f"  Documents: {stats['documents']}")
print(f"  Model: {stats['model']}")
