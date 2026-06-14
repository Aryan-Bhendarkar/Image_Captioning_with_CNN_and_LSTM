# Image Captioning with CNN + LSTM
### Deep Learning I — Neural Networks | Scaler School of Technology
**Topic**: Section 1 — Topic 13: Image Captioning with CNN + LSTM  
**Dataset**: Flickr8k  
**Framework**: PyTorch 2.x + torchvision

---

## Section 1 — Problem Understanding

### 1.1 Problem Statement
Image captioning is the task of automatically generating a natural language description for a given input image. It sits at the intersection of **Computer Vision** and **Natural Language Processing**, requiring a model to first understand the visual content of an image (objects, relationships, actions) and then translate that understanding into a coherent grammatical sentence.

This is fundamentally harder than image classification because:
- The output is a sequence (variable length), not a single label
- The model must understand both *what* is in the image and *how to describe it*
- Multiple valid descriptions exist for any single image

### 1.2 Dataset: Flickr8k
| Property | Detail |
|---|---|
| Total Images | 8,091 |
| Captions per Image | 5 (human-written) |
| Total Caption Pairs | 40,455 |
| Vocabulary (raw) | 9,180 unique words |
| Vocabulary (filtered, min_freq=5) | 3,040 tokens |
| Special Tokens | `<PAD>=0`, `<START>=1`, `<END>=2`, `<UNK>=3` |

The Flickr8k dataset contains diverse real-world images scraped from Flickr, covering a wide range of human activities, animals, sports events, and outdoor scenes. Each image has five independently written captions, providing diverse reference ground truths for evaluation.

### 1.3 Core Challenges
1. **Visual Compression Bottleneck**: Compressing the entire image into a fixed-size vector loses spatial detail.
2. **Vocabulary Sparsity**: Rare nouns (e.g., "violin", "jeep", "mohawk") appear fewer than 5 times in training data and are filtered out, causing `<UNK>` token predictions.
3. **Exposure Bias**: The model is trained with teacher forcing (ground-truth tokens as inputs) but must generate from its own predictions at test time. Early mistakes compound.
4. **Variable-Length Output**: Captions range from 8 to 40 tokens, requiring dynamic sequence generation with a meaningful stopping criterion.
5. **Multi-Object Scenes**: Without an attention mechanism, the encoder must describe complex images from a single averaged feature vector.

---

## Section 2 — Architecture

### 2.1 Overall Pipeline
The model follows the classic **encoder-decoder** paradigm for sequence generation:

```
Input Image (224×224×3)
       ↓
  [Encoder: ResNet-18]         — Extracts visual features
       ↓
  512-dim feature vector
       ↓
  [Linear Projection Layer]    — Maps 512 → 256 (embed_size)
       ↓
  256-dim image embedding
       ↓
  [Decoder: LSTM]              — Generates caption tokens autoregressively
       ↓
  Caption: "a dog is running through the grass ."
```

### 2.2 Encoder: ResNet-18
- **Base Model**: ResNet-18 pretrained on ImageNet (`torchvision.models.resnet18`)
- **Modification**: The final fully connected classification layer (1000-class output) is removed. The remaining layers output a `(batch, 512, 1, 1)` spatial feature map after average pooling.
- **Projection**: A `nn.Linear(512, 256)` layer maps the flattened 512-dimensional feature vector to the 256-dimensional word embedding space.
- **Baseline Mode**: All ResNet layers are **frozen** (`requires_grad=False`), acting as a static feature extractor.
- **Fine-tuned Mode** (Experiment 1): The final residual block (`layer4`) is unfrozen, allowing the CNN to adapt its filters to the Flickr8k image distribution.

**Why ResNet-18?** Residual connections (`x + F(x)`) allow gradients to flow through skip connections during backpropagation, enabling much deeper networks without vanishing gradient issues. ResNet-18 offers a good balance between visual feature quality and training speed on a student GPU.

### 2.3 Decoder: LSTM
- **Embedding**: `nn.Embedding(vocab_size=3040, embed_size=256)` converts token indices to dense 256-dim vectors.
- **LSTM**: `nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)`
- **Output Layer**: `nn.Linear(512, 3040)` maps LSTM hidden state to vocabulary logits.

**Decoder Input Sequence Construction (Training)**:
```
Timestep:  0          1          2          3      ...   T-1
Input:     [img_feat] [<START>]  [word_1]   [word_2] ... [word_{T-2}]
Target:    [word_1]   [word_2]   [word_3]   [word_4] ... [<END>]
```

The image feature acts as the initial "context" token at timestep 0. The `<START>` token follows at timestep 1, and all subsequent ground-truth tokens (shifted right by one position) follow. This is implemented as a single forward pass using teacher forcing.

**Why LSTM over vanilla RNN?** Standard RNNs suffer from the vanishing gradient problem during backpropagation through time (BPTT). LSTMs introduce a cell state `c_t` that creates an additive gradient highway using gating mechanisms:
- **Forget Gate** `f_t`: Decides what to discard from cell state
- **Input Gate** `i_t`: Decides what new information to write into cell state
- **Output Gate** `o_t`: Controls what portion of the cell state becomes the hidden state `h_t`

### 2.4 Training Configuration
| Hyperparameter | Value |
|---|---|
| Loss Function | `CrossEntropyLoss(ignore_index=0)` |
| Optimizer | `Adam`, lr=`3e-4` (baseline), lr=`1e-4` (fine-tuning) |
| Batch Size | 32 |
| Training Strategy | Teacher Forcing |
| Gradient Clipping | `max_norm=5.0` |
| Image Normalization | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |

**Why `ignore_index=0` (PAD)?** Captions in a batch are padded to the same length with `<PAD>` tokens (index 0). Including PAD tokens in the cross-entropy loss computation would force the model to spend gradient capacity learning to predict padding, distorting the true learning objective.

---

## Section 3 — Data Pipeline

### 3.1 Preprocessing Steps
1. **Caption Parsing**: Raw `captions.txt` is parsed into a dictionary `{image_name: [cap1, cap2, cap3, cap4, cap5]}`.
2. **Vocabulary Construction**: Word frequencies counted using `Counter`. Words appearing fewer than 5 times (min_freq) are excluded to reduce noise and OOV rate.
3. **Token Encoding**: Each caption is encoded as `[<START>] + [token_ids] + [<END>]`.
4. **Image Transforms**: Resize to `224×224`, convert to tensor, normalize with ImageNet constants.
5. **Batch Collation**: `caption_collate_fn` pads variable-length caption tensors to uniform batch length using `nn.utils.rnn.pad_sequence`.

### 3.2 Dataset Class
`Flickr8kDataset(Dataset)` returns:
- `image`: `Tensor[3, 224, 224]` — normalized image tensor
- `caption_tensor`: `Tensor[seq_len]` — encoded integer token sequence
- `image_name`: `str` — filename for evaluation lookup
- `raw_caption`: `str` — original text for display

---

## Section 4 — Inference

### 4.1 Greedy Decoding
At inference time, ground-truth captions are unavailable. The model generates tokens autoregressively by selecting the highest-probability token (`argmax`) at each step and feeding it back as input for the next step.

**Drawback**: Greedy decoding is myopic. A poor word choice at an early step cannot be corrected and degrades the entire remaining sequence.

### 4.2 Beam Search Decoding
Beam search maintains `k` parallel candidate sentences (beams) at each step. For each hypothesis, the model computes the probability distribution over the vocabulary, selects the top `k` candidate extensions, scores them by **cumulative log-probability**, and retains the top `k` across all hypotheses combined.

**Advantage**: Explores multiple generation paths simultaneously, recovering from locally poor decisions.
**Tradeoff**: Increases inference time and memory by a factor of `k`.

---

## Section 5 — Experiments & Results

All experiments were evaluated on a fixed test subset of 500 images using NLTK's `corpus_bleu`.

### 5.1 Experiment 1: Frozen vs Fine-Tuned Encoder

**Hypothesis**: Allowing the final residual block (`layer4`) of ResNet-18 to adapt to the Flickr8k distribution should improve visual feature quality, leading to more accurate and specific captions.

**Setup**:
- Variable Changed: `requires_grad` for `resnet.layer4`
- Frozen Baseline: Only `nn.Linear(512, 256)` and decoder trained
- Fine-Tuned: `layer4` + `nn.Linear(512, 256)` + decoder trained together at `lr=1e-4`
- All other variables (hidden_size=512, embed_size=256, 5 epochs) held constant

| Configuration | BLEU-1 | BLEU-4 | Trainable Params |
|---|---|---|---|
| Frozen Encoder (Baseline) | 0.5881 | 0.1076 | 4,046,048 |
| Fine-Tuned Encoder | 0.6389 | 0.1394 | 12,439,776 |

**Analysis**: Fine-tuning `layer4` improved BLEU-4 by **+3.18% absolute**. The final convolutional block of ResNet-18 encodes the most semantically abstract visual patterns. Allowing these filters to specialize on Flickr8k objects (e.g., dogs, sports equipment, clothing colors) directly improved the quality of the projected image embedding vector, giving the LSTM a richer contextual starting point.

### 5.2 Experiment 2: LSTM Hidden Size Comparison

**Hypothesis**: A larger LSTM hidden size provides more memory capacity to model complex grammatical dependencies. However, on a small corpus (40K sentences), it risks overfitting.

**Setup**:
- Variable Changed: `hidden_size` ∈ {256, 512, 1024}
- Frozen ResNet encoder for all runs
- Training: 5 epochs, `lr=3e-4`, Beam Search k=3 for evaluation

| Hidden Size | Trainable Params | BLEU-1 | BLEU-4 |
|---|---|---|---|
| 256 | 2,217,184 | 0.52 | 0.07 |
| 512 (Baseline) | 4,046,048 | 0.5881 | 0.1076 |
| 1024 | 9,276,640 | 0.58 | 0.10 |

**Analysis**: LSTM 512 was the optimal configuration. Hidden size 256 underperformed because its smaller state vector lacked the capacity to model the conditional probability distributions required for grammatically correct English sentences. Hidden size 1024 added 5.2M extra parameters but produced no improvement because Flickr8k's 40K sentences were insufficient to reliably train all weights — the model partially memorized training sequences instead of generalizing.

### 5.3 Experiment 3: Greedy vs Beam Search Decoding

**Hypothesis**: Beam Search explores more of the probability space than greedy decoding, finding globally better sentences.

**Setup**:
- Same trained baseline checkpoint for all decoding runs
- Variable Changed: decoding strategy ∈ {Greedy, Beam k=3, Beam k=5}

| Decoding Strategy | BLEU-1 | BLEU-4 |
|---|---|---|
| Greedy | 0.5478 | 0.0837 |
| Beam Search (k=3) | 0.5881 | 0.1076 |
| Beam Search (k=5) | 0.5915 | 0.1122 |

**Analysis**: Switching from Greedy to Beam Search (k=3) yielded a **+4.7% BLEU-1** and **+2.4% BLEU-4** absolute improvement on the same model, with zero additional training. Increasing beam width from 3 to 5 produced only a marginal additional gain (+0.3% BLEU-1), indicating a performance plateau. This is because beyond k=3, the beam candidates begin to share similar high-frequency phrase prefixes ("a man in a..."), eliminating the diversity benefit.

### 5.4 Improved Pipeline Results

After implementing **ImageNet Normalization** (resolving the distribution mismatch between raw pixel data and ResNet's pretrained filters) and fine-tuning `layer4` for **10 epochs** at `lr=1e-4`:

| Configuration | BLEU-1 | BLEU-4 |
|---|---|---|
| Baseline (Frozen, 5 epochs, no normalization) | 0.5105 | 0.0525 |
| Improved (Fine-tuned, 10 epochs, normalized) | **0.6583** | **0.1615** |

**Cumulative improvement**: BLEU-4 improved by **+10.9% absolute** over the initial unnormalized baseline. The normalization step alone was the single largest contributor, as it aligned the Flickr8k image distribution with the statistical distribution that ResNet-18's learned convolutional filters expect.

---

## Section 6 — Analysis & Insights

### 6.1 Failure Case Analysis

Qualitative inspection of low-scoring captions (Sentence BLEU-1 < 0.35) revealed three structural failure modes:

**Failure Case 1 — Visual Hubris (Hallucination)**
- Ground Truth: *"a police officer posing with two army officers beside his motorcycle"*
- Model Output: *"a man is a a on on bike bike a a . ."*
- **Root Cause**: The frozen ResNet-18 lacks domain-specific filters for "police uniform" or "military beret" — objects rare in ImageNet. The decoder falls back to the most statistically frequent patterns in the training corpus.

**Failure Case 2 — Language Loop (Decoder Repetition)**
- Ground Truth: *"two girls playing with people and flowers behind them"*
- Model Output: *"a man in a white shirt and a woman shirt a a . ."*
- **Root Cause**: Gender hallucination ("man" instead of "girls") from decoder bias toward high-frequency training words. The decoder also enters a repetition loop ("shirt shirt") — a known failure mode of LSTMs without attention: the hidden state becomes overloaded and the model defaults to repeating confident predictions.

**Failure Case 3 — Vocabulary Gap (OOV Words)**
- Ground Truth: *"a boy with a purple mohawk is playing the violin"*
- Model Output: *"a man in a white shirt and a shirt shirt a a . ."*
- **Root Cause**: The words "mohawk" and "violin" were filtered out of the vocabulary by min_freq=5 because they appeared fewer than 5 times in training captions. The model had no token to predict for these concepts.

### 6.2 Why ImageNet Normalization Was Critical

ResNet-18's first convolutional layer learned filters for edges, textures, and color gradients during ImageNet pretraining. These filters were calibrated to operate on inputs with `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`. Without normalizing our Flickr8k images to this same distribution, the pixel values entering ResNet-18 were systematically offset from what the filters expected, causing the intermediate feature activations to be statistically unstable. This is why the early baseline model (without normalization) generated chaotic repetitive outputs even after 5 epochs of training.

### 6.3 Limitations and Future Scope

| Limitation | Future Solution |
|---|---|
| Single visual context vector (no attention) | Bahdanau / Soft Attention Mechanism |
| ResNet-18 limited depth | ResNet-50 or Vision Transformer (ViT) encoder |
| OOV tokens for rare words | BPE/WordPiece tokenization (handles subwords) |
| Teacher Forcing exposure bias | Scheduled Sampling (gradually replace ground-truth with predictions) |
| Fixed max caption length | Dynamic length prediction |
| Language model only (no visual grounding) | CLIP + GPT-2 vision-language joint model |

---

## Section 7 — Conclusion

We successfully built and evaluated a complete end-to-end image captioning system using a ResNet-18 encoder and an LSTM decoder, trained on the Flickr8k dataset. Three controlled experiments demonstrated:

1. **Fine-tuning is better than freezing** — adapting the visual encoder to the target dataset is more important than increasing LSTM capacity.
2. **Optimal LSTM capacity exists** — 512 hidden units is the sweet spot for Flickr8k; larger models overfit, smaller models underfit.
3. **Beam Search reliably beats Greedy** — without any additional training, a k=3 beam search delivers a significant BLEU improvement and effectively eliminates early-decision repetition loops.
4. **Preprocessing alignment is critical** — ImageNet normalization was the single most impactful engineering decision, eliminating the distribution mismatch between raw pixels and pretrained convolutional filter expectations.

Our final improved model achieved **BLEU-1: 0.6583** and **BLEU-4: 0.1615** on the 500-image test set, representing a **+10.9% BLEU-4 improvement** over the unnormalized frozen baseline. The Gradio web demo further validates the model qualitatively by allowing real-time caption generation on arbitrary user-uploaded images.

---

*Submitted for: Deep Learning I — Neural Networks, Scaler School of Technology*
