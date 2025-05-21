
# 🔄 Learning Heuristics for TopSpin Using BWA* and Deep Learning

This project implements and analyzes heuristic learning methods for solving the **TopSpin puzzle** using both classical search and deep learning techniques.

## 🧩 Problem Description: TopSpin(n, k)

TopSpin is a combinatorial puzzle where `n` numbered disks are arranged in a circle. At each step:
- The whole circle can be rotated clockwise or counterclockwise.
- A fixed-size segment of `k` consecutive disks can be flipped.

The goal is to reach the sorted configuration `[1, 2, ..., n]` from a given permutation using the smallest number of moves. Each move has the same cost.

---

## 📚 Project Components

### 🔹 1. Puzzle Implementation
A class `TopSpinState` models the puzzle:
- Generates all valid successor states.
- Detects goal states.
- Supports state encoding and comparison.

### 🔹 2. Search Algorithm: Batch Weighted A* (BWA*)
A parallel version of Weighted A* that:
- Expands a **batch of B nodes** at each step.
- Uses a weighted priority `f(n) = g(n) + w ⋅ h(n)`.
- Supports batching for efficient GPU-based heuristic evaluation.

### 🔹 3. Heuristic Learning
Two methods for training neural network heuristics:
- **Bootstrapping**: Learns from the length of paths found by BWA*.
- **One-step Bellman update**: Trains by minimizing the error between a state and the best successor.

Both methods use PyTorch and are trained on random states generated from goal configurations.

### 🔹 4. Empirical Evaluation
The learned heuristics are evaluated using BWA* under different parameters (weight `W`, batch size `B`), and compared to a baseline gap-counting heuristic.

Metrics reported:
- Average runtime
- Average path length
- Average number of node expansions

---

## 🧠 Technologies Used

* PyTorch
* NumPy
* Object-oriented programming
* A\*/Weighted A\*/Batching heuristics
* Neural network training and evaluation

---

## 📝 Files Overview

| File               | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `topspin.py`       | TopSpin puzzle logic and state transitions           |
| `BWAS.py`          | Batch Weighted A\* search implementation             |
| `heuristics.py`    | Heuristic classes: base, Bellman, bootstrap, learned |
| `training.py`      | Neural training procedures for Bellman & bootstrap   |
| `main.py`          | Main runner script for executing search              |
| `requirements.txt` | Python dependencies                                  |

---

## 🏆 Highlights

* Combines **search algorithms** with **deep learning**.
* Implements two training regimes for heuristics.
* Parallel search with batching optimized for GPU evaluation.
* Rigorous empirical evaluation and comparison.

---


## 📊 Results Summary

| W | B   | Heuristic           | Avg. Runtime | Avg. Path Length | Avg. Expansions |
|---|-----|---------------------|--------------|------------------|-----------------|
| 2 | 1   | Basic               | 0.17         | 17.78            | 5160.56         |
| 2 | 1   | Learned-Bellman     | 2.03         | 20.26            | 230.53          |
| 2 | 1   | Learned-Bootstrap   | 0.06         | 23.49            | 516.96          |
| 2 | 100 | Basic               | 0.09         | 17.73            | 5655.11         |
| 2 | 100 | Learned-Bellman     | 1.45         | 17.91            | 1402.41         |
| 2 | 100 | Learned-Bootstrap   | 0.04         | 18.93            | 1516.11         |
| 5 | 1   | Basic               | 0.1          | 20.82            | 415.96          |
| 5 | 1   | Learned-Bellman     | 2.79         | 24.51            | 351.82          |
| 5 | 1   | Learned-Bootstrap   | 0.02         | 30.15            | 238.23          |
| 5 | 100 | Basic               | 0.02         | 18.53            | 1513.51         |
| 5 | 100 | Learned-Bellman     | 1.25         | 17.94            | 1400.91         |
| 5 | 100 | Learned-Bootstrap   | 0.03         | 18.96            | 1501.01         |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/MoriaCohen8/TopSpin-Heuristic-Learning.git
cd TopSpin-Heuristic-Learning/searchAssignment
````

### 2. Setup Virtual Environment (Python 3.10)

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Run Experiments

```bash
python main.py
```

---
