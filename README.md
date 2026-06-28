# Deep Learning Next-Word Prediction Engine

An end-to-end natural language processing (NLP) framework that utilizes a stacked, bidirectional Recurrent Neural Network (RNN) to learn structural language syntax and dynamically predict subsequent words based on preceding text contexts.

## 🚀 Key Highlights & Metrics
* **Vocabulary Size:** 8,201 unique tokens trained on a robust linguistic corpus.
* **Dataset Scale:** Engineered from 9,566 complete sentences compiling 96,314 n-gram training tensor sequence arrays.
* **Model Validation Performance:** Achieved a **14.05% validation accuracy** utilizing a 20% validation split.
* **Mathematical Justification:** Outperforms an un-trained random guessing baseline ($\frac{1}{8201} \approx 0.012\%$) by **over 1,150x (approx. 1,200x)**, ensuring highly optimized semantic predictions.

## 🛠️ System Architecture

### 1. Preprocessing Pipeline
* **Text Tokenization:** Standardizes inputs into lowercase formats and utilizes a Keras Tokenizer mapping to an Out-Of-Vocabulary (`<unkn>`) indicator token.
* **N-Gram Token Generation:** Generates progressive word combinations split down vector columns into structural token indices.
* **Pre-Padding Matrix:** Standardizes incoming array lengths using pre-padding methods up to a maximum sentence threshold of 18 tokens.

### 2. Deep Learning Network Topography
The system utilizes a specialized, highly regularized multi-layer `Sequential` model topology:
* **Embedding Layer:** Compresses high-dimensional vocabulary indices into a dense, semantic **256-dimensional space**.
* **Stacked Bidirectional LSTMs:** Leverages dual-layer sequence learning to maintain long-range forward and backward contextual relationships (Layer 1: 256 units with sequence return; Layer 2: 100 units).
* **Regularization Mechanics:** Implements **LayerNormalization** and a **20% Dropout rate** across hidden nodes to prevent vanishing gradients and overfitting.
* **Softmax Output Classifier:** Maps dense layers to a vocabulary probability array to isolate the next correct syntax token.

## ⚙️ Optimization & Training Configurations
* **Loss Function:** Categorical Cross-Entropy.
* **Optimizer:** Adam Optimizer (Adaptive Moment Estimation).
* **Metrics:** Tracks primary Accuracy along with `TopKCategoricalAccuracy` (evaluated at $k=3$).
* **EarlyStopping Monitor:** Configured with a patience parameter of 5 epochs tracking `val_accuracy` to instantly restore optimal network weights.

## 🔮 Generative Text Inference Mode
The inference script features advanced execution modes to avoid repetitive text loop traps:
* **Greedy Search:** Instantly snaps to the single token holding the maximum mathematical probability.
* **Stochastic Temperature Scaling:** Applies a smooth logarithmic temperature scaling modifier ($T \in [0.1, 1.0]$) over raw prediction outputs. This adjusts probability distribution shapes before applying a random weighted choice to achieve creative, non-repetitive phrase generation.

## 🧰 Built With
* **Python**
* **TensorFlow & Keras** (Sequential API, Layers, Callbacks)
* **NumPy**
* **Matplotlib** (Loss and Accuracy curve plot tracking)
* **Google Colab Environment**
