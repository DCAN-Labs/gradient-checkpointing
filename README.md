# gradient-checkpointing
A Deep Dive into Memory-Compute Trade-offs

  The project could demonstrate:

  1. Basic Implementation - Build gradient checkpointing from scratch in PyTorch, showing how to selectively not store activations
  and recompute during backward pass
  2. Benchmarking Suite - Compare memory usage and training time across:
    - Standard backprop (O(n) memory, O(n) compute)
    - Full checkpointing (O(1) memory, O(nÂ²) compute)
    - Selective checkpointing (O(ân) memory, O(nân) compute)
  3. Optimal Checkpoint Selection - Implement dynamic programming algorithm to find optimal checkpoint locations given memory budget
  4. Architecture-Specific Strategies - Show how different architectures (ResNet skip connections, Transformer attention, U-Net
  encoders) require different checkpointing strategies
  5. Profiling Visualizations - Create memory timeline plots showing peak memory reduction and compute overhead

  This would make a great educational repository demonstrating a fundamental technique used in training large models like GPT-4,
  completely independent of medical imaging specifics.
