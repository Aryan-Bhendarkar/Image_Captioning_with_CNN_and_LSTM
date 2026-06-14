# Image Captioning with CNN + LSTM (Flickr8k)

An end-to-end Deep Learning pipeline that generates textual descriptions for input images. The model is built using PyTorch and trained on the Flickr8k dataset, bridging computer vision and natural language processing.

## Core Architecture
* **Encoder (Vision)**: Pre-trained **ResNet-18** with the final classification layer removed. It extracts high-level spatial visual features (512-dimensional vector) from $224 \times 224$ images.
* **Projection Layer**: A linear layer (`nn.Linear`) that maps the 512-dimensional CNN output to a 256-dimensional space, matching the text embedding size.
* **Decoder (Language)**: An **LSTM** network that takes the projected image feature vector as its first timestep input, and then sequentially predicts the next tokens autoregressively.

---

## Controlled Experiments & Performance Metrics

We evaluated the model on a test split of 500 images using the standard translation evaluation metric **BLEU** (Bilingual Evaluation Understudy).

### Key Findings
1. **Encoder Fine-tuning**: Unfreezing `layer4` (the final residual block) of ResNet-18 allowed the convolutional filters to adapt to Flickr8k image distributions, raising the test **BLEU-4** score from **10.76% to 13.94%**.
2. **LSTM Capacity**: Testing hidden sizes of 256, 512, and 1024 showed that **512 is the optimal hidden size**. 256 suffered from undercapacity, while 1024 overfit the small dataset text corpus (~40,000 sentences) and did not yield performance gains.
3. **Decoding Strategy**: **Beam Search (k=3)** yielded a massive **+4.7% absolute BLEU-1 increase** over Greedy decoding by exploring multiple parallel generation paths and avoiding early local-optima traps.

### Final Improved Pipeline
By implementing **ImageNet Normalization** (matching the pretrained ResNet distribution) and unfreezing the encoder's last block for **10 epochs** of training:
* **BLEU-1**: **65.83%** (Baseline: 58.81%)
* **BLEU-4**: **16.15%** (Baseline: 10.76%)

---

## Key Engineering Insights for Recruiters

* **Preprocessing Alignment**: The single largest performance boost came from adding ImageNet normalization. This resolved a critical distribution mismatch between the raw Flickr8k pixels and the pre-trained ResNet weights.
* **Exposure Bias**: We trained using **Teacher Forcing** (stable and fast convergence), but designed the greedy and beam search inference loops to autoregressively feed predicted tokens back as inputs, handling the train-test distribution shift.
* **Failure Modes**: Qualitative analysis showed the model struggles with complex multi-object scenes (e.g., horses near a fire) or rare vocabulary words (resulting in `<UNK>` tokens). This is due to the lack of an attention mechanism and visual compression bottleneck, representing a perfect future scope for **Transformer-based (Vision-Language) architectures**.

---

## Interactive Gradio Demo
The project includes a web-based user interface built using **Gradio**. Users can upload any image to see real-time predictions compared side-by-side using Greedy Decoding, Beam Search (k=3), and Beam Search (k=5).
