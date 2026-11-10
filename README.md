# Model Mechanic

A small experiment in repairing neural networks without retraining them.

I started this after reading [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). The question that interested me was whether an LLM could suggest a small edit to a model's weights after seeing a specific failure, and how we could tell if that edit was actually safe to use.

I'm starting with a simple 2D robot simulation and a small PyTorch policy network. I'll compare three things: leaving the model alone, fine-tuning it on failure examples, and testing an LLM-proposed weight edit.

The important test is whether an edit fixes the original problem **without making other situations worse**.

## Current status

Planning the first experiment. No results yet.

## First milestone

- [ ] Build a simple 2D environment
- [ ] Train a small policy network
- [ ] Record one reproducible failure
- [ ] Try one constrained weight edit
- [ ] Compare behaviour before and after on separate test scenarios

## Reference

[Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
