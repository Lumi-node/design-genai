# DesignGenAI

<div class="hero">
  <h1 style="text-align: center;">DesignGenAI</h1>
  <p style="text-align: center; font-size: 1.2em;">Generates high-fidelity, structurally valid designs from semantic type embeddings using ANE-accelerated neural networks.</p>
</div>

---

## ✨ Features

DesignGenAI bridges the gap between abstract design intent and concrete, functional UI code. Our system leverages advanced neural architectures optimized for rapid, high-quality design synthesis.

<div class="row">
  <div class="col-md-4">
    <div class="card">
      <div class="card-body text-center">
        <i class="fas fa-database fa-3x text-primary mb-3"></i>
        <h5 class="card-title">Semantic Data Conversion</h5>
        <p class="card-text">Automatically transforms 50-100 existing design JSONs into structured, labeled training datasets for robust learning.</p>
      </div>
    </div>
  </div>
  <div class="col-md-4">
    <div class="card">
      <div class="card-body text-center">
        <i class="fas fa-brain fa-3x text-success mb-3"></i>
        <h5 class="card-title">ANE-Accelerated Modeling</h5>
        <p class="card-text">Utilizes specialized ANE-accelerated neural networks to map design type embeddings directly to component JSON vectors.</p>
      </div>
    </div>
  </div>
  <div class="col-md-4">
    <div class="card">
      <div class="card-body text-center">
        <i class="fas fa-code fa-3x text-warning mb-3"></i>
        <h5 class="card-title">High-Fidelity Generation</h5>
        <p class="card-text">Generates distinct, valid HTML pages, achieving over 80% structural validity from model output.</p>
      </div>
    </div>
  </div>
</div>

---

## 🚀 Quick Start

Getting started with DesignGenAI is straightforward. Clone the repository and install the necessary dependencies.

<div class="alert alert-info">
  <strong>Installation Command:</strong>
  <pre><code>pip install designgenai[all]</code></pre>
</div>

<div class="d-flex justify-content-center mt-4">
  <a href="/getting-started/" class="btn btn-primary btn-lg me-3">
    <i class="fas fa-arrow-right me-2"></i> Get Started Now
  </a>
  <a href="/api/" class="btn btn-outline-secondary btn-lg">
    View API Docs
  </a>
</div>

---

## 📚 Documentation

Explore the core components of the DesignGenAI pipeline:

*   **Dataset Builder:** Learn how `dataset_builder.py` prepares your initial design corpus.
*   **Model Architecture:** Dive into `model.py` to understand the 2-layer embedding mapping.
*   **Training Pipeline:** Review `train.py` and the integration with `ane_trainer`.
*   **Generation Flow:** See how `generate.py` synthesizes and converts designs into HTML.