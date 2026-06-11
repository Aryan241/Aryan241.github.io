---
title: "This Blog Is All You Need!"
date: 2026-06-12
permalink: /blog/2026/06/this-blog-is-all-you-need/
tags:
  - machine learning
  - transformers
mathjax: true
---

With the use of Large Language Models (LLMs) at an all-time high, you must have sat and wondered about what exactly is going on inside them, and how exactly they generate text one word after another. An LLM is essentially a big, complex mathematical function. Instead of predicting a single next word with absolute certainty, it assigns a probability to every possible next word. To make the dialogue sound natural, the system occasionally picks less likely words at random. This means that even though the underlying mathematics are deterministic, the same prompt can yield a completely different answer every time.

### How do these models learn to predict the next word so accurately?

They do it by processing a huge chunk of data from the internet. You can think of this training process as tuning the dials on a massive machine. The behavior of an LLM is entirely governed by continuous numerical values called *parameters (or weights)*. Changing these weights changes the word probabilities. What makes these models “large” is that they contain hundreds of billions of these parameters.

Nobody sets these dials by hand. The process follows a specific lifecycle:

1. **Random Initialization:** The parameters start completely random, outputting pure gibberish.
2. **The Feedback Loop:** The model is fed trillions of training examples (ranging from a few words to thousands). It tries to predict the final word of a sequence.
3. **Backpropagation:** An algorithm compares the model’s guess to the true word and slightly tweaks the parameters to make the correct choice more likely next time.

This process is called *pre-training*. Because the goal of mimicking internet text is different from being a helpful assistant, the model later undergoes *Reinforcement Learning with Human Feedback (RLHF)*, where human workers flag bad responses to further refine the parameters.

### The Shift to Transformers

The sheer scale of computation required for this training is mind-boggling, which was only possible because of GPUs. Before transformers existed, the standard way to process a sentence was with a *Recurrent Neural Network (RNN)*. An RNN reads one word at a time, left to right, updating a hidden state as it goes.

This created a real problem for anything involving *long-range dependencies*. A pronoun at position 40 might refer to a noun at position 5. An RNN had to carry that noun’s meaning through 35 intermediate steps, each one slightly smearing the signal. Furthermore, the gradients during training had the same problem in reverse, they either vanished to nothing or exploded.

Everything changed in 2017 when a team at Google introduced the *Transformer architecture*. Instead of reading text from start to finish, Transformers soak in the entire passage all at once, in parallel. Instead of passing information through a chain, attention lets every word look directly at every other word all at once, in a single operation.

---

## The Transformer Architecture

Follow along the architecture while I explain what each component does. The left half of the model is called the *Encoder*, and the right half is called the *Decoder*.

![The Transformer Architecture](/images/transformer_architecture.png)

*Figure 1: The Transformer Architecture Block Diagram*

---

## The Encoder

### 1. Input Embedding

Any sentence fed into the architecture is first assigned an *Input ID*, which is a number corresponding to the position of that specific token in the vocabulary.

For example, in the sentence *“Your Cat Is A Lovely Cat”*, every single word is assigned a scalar input ID (say 105 or 6089), pointing to its position in the vocabulary. Each word is further represented as an *Embedding Vector*. Think of these vectors lying in a high-dimensional space, where the directions they settle in carry semantic meaning. For instance, words like “tower”, “lighthouse”, and “tall” will lie near each other. By common practice, we define the embedding vector size as:

$$d_{\text{model}} = 512$$

### 2. Positional Encoding

Apart from representing the semantic meaning of a word, the model needs to know its structural position in the sentence—which words are close and which are far away. Because Transformers process everything in parallel, we must inject this sequence order explicitly via a learned or fixed *Positional Encoding*.

We use sine and cosine functions at different frequencies to compute the position embedding vector (also of size 512):

$$PE_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

$$PE_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

This is computed once and reused for every sentence during training and inference.

### 3. Self-Attention: Queries, Keys, and Values

Self-attention allows the model to relate words to each other, letting them interact and pass context along.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Words “talk” to one another by mapping their embeddings into three distinct roles:

* **Query (Q):** A question that each word embedding asks. For example, a noun like “Cat” might ask, *“Is there an adjective in front of me?”*. It is calculated by multiplying the word embedding by a trainable weight matrix \(W_Q\) . This projects it into a smaller dimensional space (e.g., \(d_k = 128\) ).
* **Key (K):** The bridge that answers a query when they closely align. It is computed by multiplying the embedding by a trainable weight matrix \(W_K\) .
* **Value (V):** The actual contextual content added to the word embedding if a key matches a query. It is calculated using the weight matrix \(W_V\) .

To measure how well a key matches a query, we take the dot product between each possible key-query pair to create the *Attention Pattern*. Higher dot products correspond to higher correlations.

![Attention Pattern Matrix](/images/attention_pattern.png)

*Figure 2: Attention Pattern*

These attention scores are multiplied by the value vectors and added column-wise to compute the delta embeddings, which are then combined with the original embeddings to yield context-rich vectors. This entire loop constitutes a *Single Head of Attention*.

### Why Multi-Head Attention?

Language is too complex for a single question. If the word “Cat” looks at a sentence, it needs to ask multiple questions simultaneously:

* *“Where is the adjective describing me?”*
* *“What verb am I performing?”*
* *“Does a pronoun later in the sentence refer back to me?”*

If we only use one set of \(W_Q\), \(W_K\), \(W_V\) matrices, the model is forced to average all these questions together, resulting in mediocre performance. *Multi-Head Attention* solves this by asking multiple questions in parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

![Multi-Head Attention Workflow](/images/mha_workflow.jpg)

*Figure 3: MH-A WorkFlow*

When we stack our sequence tokens together, they form an input matrix of size \((\text{seq}, d_{\text{model}})\), which is (6, 512) for our sample sentence. Instead of performing a small projection, the input matrix is multiplied by massive \((512, 512)\) global projection matrices (\(W_Q\) , \(W_K\) , \(W_V\) ). The resulting matrices (Q’, K’, V’) are sliced into h=4 parallel heads:

$$d_k = \frac{d_{\text{model}}}{h} = \frac{512}{4} = 128$$

This gives us 4 unique heads isolating their own semantic spaces to track distinct properties simultaneously.

To pass this to the next neural network layer, we concatenate the 4 head columns back into a single unified matrix \(H\) of size (6, 512). To let these isolated context blocks interact, we perform a final multiplication with an *Output Projection Matrix* (\(W^O\)) of size (512, 512). The final embedding for the word “cat” is no longer a static dictionary vector; it is a context-aware 512-dimensional vector that simultaneously knows it is “lovely”, belongs to “your”, and links to the structural relationships in the sentence.

### Why the heck do we divide by \(\sqrt{d_k}\)?

Because we are working in a massive 128-dimensional space, calculating the dot products \(Q \times K^T\) pushes our values into extreme, massive numbers. When these giant numbers hit the softmax function, it gets pushed into its flattest regions, forcefully yielding an extreme output (1.0 for the highest score, 0.0 for everything else).

Mathematically, the gradient of a flat softmax curve is virtually zero. During training, backpropagation hits a wall, gradients vanish, and the model completely stops learning. Dividing by \(\sqrt{128}\) scales the variance back down to 1, keeping the inputs to softmax small, the attention maps smooth, and the gradients healthy.

### Layer Normalization (Add &amp; Norm)

Right after adding our residual/skip connection, the flowchart hits *Layer Normalization (LayerNorm)*. This step rescales our numerical values to prevent them from drifting out of control as they move deeper into the network:

$$\text{LayerNorm}(x) = \gamma \left( \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \right) + \beta$$

Unlike batch normalization, LayerNorm isolates one word at a time and calculates the mean (\(\mu\)) and variance (\(\sigma^2\)) across all 512 channels of that single word’s embedding vector. It standardizes the vector, forcing the features of the word “cat” to reset to a stable baseline with a mean of 0 and variance of 1.

To keep the model from being too restricted, LayerNorm introduces two highly adjustable, learnable parameters: Gamma (\(\gamma\)) for scaling and Beta (\(\beta\)) for shifting. The Transformer tweaks these through backpropagation, maintaining numerical stability across the architecture while re-introducing precise, controlled fluctuations where it needs them to master context.

---

## The Decoder

Let's look at the “Decoder” now. The Decoder uses the same embedding and positional encoding setup, but processes data through unique foundational blocks:

### 1. Masked Multi-Head Attention

Our goal here is to make the model *causal*, meaning the output at a certain position can only depend on words at previous positions. The model must not be able to see future words during generation. To achieve this, we replace the attention scores above the principal diagonal in the attention pattern with \(-\infty\) before applying the softmax function, which effectively forces those future weights to 0.

### 2. Cross-Attention

The output then passes into a *Cross-Attention block*. Here, the *Keys (K) and Values (V)* come directly from the Encoder’s final output, while the *Queries (Q)* are streamed from the Decoder’s lower layers. This allows the generated targets to look back directly at the source context sequence.

Finally, the data is pushed through a fully-connected *Feed-Forward Network (FFN)*, projected via a *Linear Layer*, and normalized through a *Softmax* function to compute next-token probabilities.

---

## Inference vs. Training Lifecycle

Let’s look at a translation task translating *“I love you very much”* into Italian (*“Ti amo molto”*), as it outlines the differences clearly and highlights the foundational application of the original transformer architecture.

### Training (Parallelized Pass)

During training, the architecture operates in a single, highly parallelized step.

![Training Flowchart](/images/training_flowchart.jpg)

*Figure 4: Training Flowchart*

* The full English source sentence wrapped in `<SOS>` and `<EOS>` tokens is converted into input embeddings, combined with positional encodings, and sent to the encoder all at once into a rich contextual matrix capturing how every word interacts with every other word.
* Simultaneously, the target Italian sequence is fed into the decoder. Because it is the training phase, the model doesn't need to guess word-by-word; we use *Teacher Forcing* and pass the entire target sequence through the decoder in a single pass.
* The decoder uses its inputs as Queries, matching them against the Keys and Values streamed directly from the encoder.
* Output logits are computed via a linear layer, and softmax calculates the maximum score to identify correct words. Because the entire sequence is evaluated concurrently, the model learns from mistakes across the whole sentence at a single time step, making training incredibly efficient.

### Inference (Autoregressive Loop)

While training happens all at once, inference (prediction) is a gradual, token-by-token loop.

![Inference Flowchart](/images/inference_flowchart.jpg)

*Figure 5: Inference Flowchart*

1. When you ask the model to translate in real-time, the encoder runs *exactly once* to map out the English sentence; since the source text never changes, we save compute by freezing this output.
2. The decoder, however, must start completely blind, receiving only the `<SOS>` token padded to the sequence length.
3. **Time Step 1:** The decoder processes this single token alongside the encoder's data, applies the softmax layer to the resulting logits, and selects the token with the highest probability (`"ti"`).
4. **Time Step 2:** The model appends the generated word back to the decoder's input, feeding `<SOS> ti` back into the loop to generate `"amo"`.
5. This loop repeats sequentially (`<SOS> ti amo` → `"molto"`) until the model predicts the `<EOS>` token, signaling it to stop.

### Decoding Strategies

* **Greedy Strategy:** This baseline approach of always picking the single highest probability score at each step is efficient but can be short-sighted.
* **Beam Search:** To achieve better generation quality, advanced models use Beam Search. Instead of blindly picking a single word, it keeps track of the top B (beam width) most probable word combinations at every single step, systematically reducing weaker sentence structures while preserving the most contextually accurate paths.

---

That is the end of the blog. Hope you read the blog with attention because “Attention is All You Need”!
