---
title: "This Blog Is All You Need!"
date: 2026-06-12
permalink: /blog/2026/06/this-blog-is-all-you-need/
tags:
  - machine learning
  - transformers
  - jekyll
---

[cite_start]With the use of Large Language Models (LLMs) at an all-time high, you must have sat and wondered about what exactly is going on inside them, and how exactly they generate text one word after another[cite: 2]. [cite_start]An LLM is essentially a big, complex mathematical function[cite: 3]. [cite_start]Instead of predicting a single next word with absolute certainty, it assigns a probability to every possible next word[cite: 4]. [cite_start]To make the dialogue sound natural, the system occasionally picks less likely words at random[cite: 5]. [cite_start]This means that even though the underlying mathematics are deterministic, the same prompt can yield a completely different answer every time[cite: 6].

### How do these models learn to predict the next word so accurately?

[cite_start]They do it by processing a huge chunk of data from the internet[cite: 8]. [cite_start]You can think of this training process as tuning the dials on a massive machine[cite: 9]. [cite_start]The behavior of an LLM is entirely governed by continuous numerical values called **parameters (or weights)**[cite: 10]. [cite_start]Changing these weights changes the word probabilities[cite: 11]. [cite_start]What makes these models "large" is that they contain hundreds of billions of these parameters[cite: 11]. 

[cite_start]Nobody sets these dials by hand[cite: 12]. The process follows a specific lifecycle:

1. [cite_start]**Random Initialization:** The parameters start completely random, outputting pure gibberish[cite: 13].
2. [cite_start]**The Feedback Loop:** The model is fed trillions of training examples (ranging from a few words to thousands)[cite: 14]. [cite_start]It tries to predict the final word of a sequence[cite: 15].
3. [cite_start]**Backpropagation:** An algorithm compares the model's guess to the true word and slightly tweaks the parameters to make the correct choice more likely next time[cite: 16].

[cite_start]This process is called **pre-training**[cite: 17]. [cite_start]Because the goal of mimicking internet text is different from being a helpful assistant, the model later undergoes **Reinforcement Learning with Human Feedback (RLHF)**, where human workers flag bad responses to further refine the parameters[cite: 17]. 

### The Shift to Transformers

[cite_start]The sheer scale of computation required for this training is mind-boggling, which was only possible because of GPUs[cite: 18]. [cite_start]Before transformers existed, the standard way to process a sentence was with a **Recurrent Neural Network (RNN)**[cite: 19]. [cite_start]An RNN reads one word at a time, left to right, updating a hidden state as it goes[cite: 20]. 

[cite_start]This created a real problem for anything involving **long-range dependencies**[cite: 21]. [cite_start]A pronoun at position 40 might refer to a noun at position 5[cite: 22]. [cite_start]An RNN had to carry that noun's meaning through 35 intermediate steps, each one slightly smearing the signal[cite: 22]. [cite_start]Furthermore, the gradients during training had the same problem in reverse—they either vanished to nothing or exploded[cite: 23].

[cite_start]Everything changed in 2017 when a team at Google introduced the **Transformer architecture**[cite: 24]. [cite_start]Instead of reading text from start to finish, Transformers soak in the entire passage all at once, in parallel[cite: 25]. [cite_start]Instead of passing information through a chain, attention lets every word look directly at every other word all at once, in a single operation[cite: 26].

---

## The Transformer Architecture

[cite_start]Follow along the architecture while we explain what each component does[cite: 30]. [cite_start]The left half of the model is called the **Encoder**, and the right half is called the **Decoder**[cite: 31].

![The Transformer Architecture](/images/transformer_architecture.png)
[cite_start]*Figure 1: The Transformer Architecture Block Diagram [cite: 29]*

---

## The Encoder

### 1. Input Embedding
[cite_start]Any sentence fed into the architecture is first assigned an **Input ID**, which is a number corresponding to the position of that specific token in the vocabulary[cite: 35]. 

[cite_start]For example, in the sentence *"Your Cat Is A Lovely Cat"*, every single word is assigned a scalar input ID (say `105` or `6089`), pointing to its position in the vocabulary[cite: 36]. [cite_start]Each word is further represented as an **Embedding Vector**[cite: 37]. [cite_start]Think of these vectors lying in a high-dimensional space, where the directions they settle in carry semantic meaning[cite: 38]. [cite_start]For instance, words like "tower", "lighthouse", and "tall" will lie near each other[cite: 39]. By common practice, we define the embedding vector size as:

[cite_start]$$d_{\text{model}} = 512$$ [cite: 40]

### 2. Positional Encoding
[cite_start]Apart from representing the semantic meaning of a word, the model needs to know its structural position in the sentence—which words are close and which are far away[cite: 42]. [cite_start]Because Transformers process everything in parallel, we must inject this sequence order explicitly via a learned or fixed **Positional Encoding**[cite: 42, 43]. 

[cite_start]We use sine and cosine functions at different frequencies to compute the position embedding vector (also of size 512)[cite: 46]:

[cite_start]$$PE_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$ [cite: 44]

[cite_start]$$PE_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$ [cite: 45]

[cite_start]This is computed once and reused for every sentence during training and inference[cite: 46].

### 3. Self-Attention: Queries, Keys, and Values
[cite_start]Self-attention allows the model to relate words to each other, letting them interact and pass context along[cite: 49, 50]. 

[cite_start]$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$ [cite: 51]

[cite_start]Words "talk" to one another by mapping their embeddings into three distinct roles[cite: 53]:

* [cite_start]**Query ($Q$):** A question that each word embedding asks[cite: 54]. [cite_start]For example, a noun like "Cat" might ask, *"Is there an adjective in front of me?"*[cite: 54]. [cite_start]It is calculated by multiplying the word embedding by a trainable weight matrix $W_Q$[cite: 55]. [cite_start]This projects it into a smaller dimensional space (e.g., $d_k = 128$)[cite: 55].
* [cite_start]**Key ($K$):** The bridge that answers a query when they closely align[cite: 57, 59]. [cite_start]It is computed by multiplying the embedding by a trainable weight matrix $W_K$[cite: 57].
* [cite_start]**Value ($V$):** The actual contextual content added to the word embedding if a key matches a query[cite: 60]. [cite_start]It is calculated using the weight matrix $W_V$[cite: 61].

[cite_start]To measure how well a key matches a query, we take the dot product between each possible key-query pair to create the **Attention Pattern**[cite: 62]. [cite_start]Higher dot products correspond to higher correlations[cite: 63]. 

![Attention Pattern Matrix](/images/attention_pattern.png)
[cite_start]*Figure 2: Softmax-normalized attention scores summing up to 1 across rows [cite: 64]*

[cite_start]These attention scores are multiplied by the value vectors and added column-wise to compute the delta embeddings, which are then combined with the original embeddings to yield context-rich vectors[cite: 65]. [cite_start]This entire loop constitutes a **Single Head of Attention**[cite: 66].

---

### Why Multi-Head Attention?

[cite_start]Language is too complex for a single question[cite: 69]. [cite_start]If the word "Cat" looks at a sentence, it needs to ask multiple questions simultaneously[cite: 69]: 
* [cite_start]*"Where is the adjective describing me?"* [cite: 69]
* [cite_start]*"What verb am I performing?"* [cite: 69]
* [cite_start]*"Does a pronoun later in the sentence refer back to me?"* [cite: 70]

[cite_start]If we only use one set of $W_Q, W_K, W_V$ matrices, the model is forced to average all these questions together, resulting in mediocre performance[cite: 71, 72]. [cite_start]**Multi-Head Attention** solves this by asking multiple questions in parallel[cite: 73]:

[cite_start]$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$ [cite: 74]
[cite_start]$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$ [cite: 75]

![Multi-Head Attention Workflow](/images/mha_workflow.png)
[cite_start]*Figure 3: Multi-Head Attention Matrix Breakdown and Re-projection [cite: 77]*

[cite_start]When we stack our sequence tokens together, they form an input matrix of size $(\text{seq}, d_{\text{model}})$, which is $(6, 512)$ for our sample sentence[cite: 79, 80]. 

[cite_start]Instead of performing a small projection, the input matrix is multiplied by massive $(512, 512)$ global projection matrices ($W_Q, W_K, W_V$)[cite: 83]. [cite_start]The resulting matrices ($Q', K', V'$) are sliced into $h=4$ parallel heads[cite: 82, 84]:

[cite_start]$$d_k = \frac{d_{\text{model}}}{h} = \frac{512}{4} = 128$$ [cite: 85, 86]

[cite_start]This gives us 4 unique heads isolating their own semantic spaces to track distinct properties simultaneously[cite: 87, 88, 89]. 

[cite_start]To pass this to the next neural network layer, we concatenate the 4 head columns back into a single unified matrix $H$ of size $(6, 512)$[cite: 90, 92]. [cite_start]To let these isolated context blocks interact, we perform a final multiplication with an **Output Projection Matrix ($W^O$)** of size $(512, 512)$[cite: 95]. [cite_start]The final embedding for the word "cat" is no longer a static dictionary vector; it is a context-aware 512-dimensional vector that simultaneously knows it is "lovely", belongs to "your", and links to the structural relationships in the sentence[cite: 96, 97].

### Why do we divide by $\sqrt{d_k}$?
[cite_start]Because we are working in a massive 128-dimensional space, calculating the dot products ($Q \times K^T$) pushes our values into extreme, massive numbers[cite: 99]. [cite_start]When these giant numbers hit the softmax function, it gets pushed into its flattest regions, forcefully yielding an extreme output (1.0 for the highest score, 0.0 for everything else)[cite: 100]. 

[cite_start]Mathematically, the gradient of a flat softmax curve is virtually zero[cite: 101]. [cite_start]During training, backpropagation hits a wall, gradients vanish, and the model completely stops learning[cite: 102]. [cite_start]Dividing by $\sqrt{128}$ scales the variance back down to 1, keeping the inputs to softmax small, the attention maps smooth, and the gradients healthy[cite: 103].

### Layer Normalization (Add & Norm)
[cite_start]Right after adding our residual/skip connection, the flowchart hits **Layer Normalization (LayerNorm)**[cite: 106]. [cite_start]This step rescales our numerical values to prevent them from drifting out of control as they move deeper into the network[cite: 106]:

[cite_start]$$\text{LayerNorm}(x) = \gamma \left( \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \right) + \beta$$ [cite: 106]

[cite_start]Unlike batch normalization, LayerNorm isolates one word at a time and calculates the mean ($\mu$) and variance ($\sigma^2$) across all 512 channels of that single word's embedding vector[cite: 107]. [cite_start]It standardizes the vector, forcing the features of the word "cat" to reset to a stable baseline with a mean of 0 and variance of 1[cite: 108]. 

[cite_start]To keep the model from being too restricted, LayerNorm introduces two highly adjustable, learnable parameters—Gamma ($\gamma$) for scaling and Beta ($\beta$) for shifting[cite: 108]. [cite_start]The Transformer tweaks these through backpropagation, maintaining numerical stability across the architecture while re-introducing precise, controlled fluctuations where it needs them to master context[cite: 108, 109].

---

## The Decoder

[cite_start]The Decoder uses the same embedding and positional encoding setup, but processes data through unique foundational blocks[cite: 111]:

### 1. Masked Multi-Head Attention
[cite_start]Our goal here is to make the model **causal**, meaning the output at a certain position can only depend on words at previous positions[cite: 115]. [cite_start]The model must not be able to see future words during generation[cite: 116]. [cite_start]To achieve this, we replace the attention scores above the principal diagonal in the attention pattern with $-\infty$ before applying the softmax function, which effectively forces those future weights to 0[cite: 117].

### 2. Cross-Attention
[cite_start]The output then passes into a **Cross-Attention block**[cite: 112]. [cite_start]Here, the **Keys ($K$) and Values ($V$)** come directly from the Encoder's final output, while the **Queries ($Q$)** are streamed from the Decoder's lower layers[cite: 112]. [cite_start]This allows the generated targets to look back directly at the source context sequence[cite: 112].

[cite_start]Finally, the data is pushed through a fully-connected **Feed-Forward Network (FFN)**, projected via a **Linear Layer**, and normalized through a **Softmax** function to compute next-token probabilities[cite: 113].

---

## Inference vs. Training Lifecycle

[cite_start]Let's look at a translation task translating *"I love you very much"* into Italian (*"Ti amo molto"*), as it outlines the differences clearly[cite: 119, 121].

### Training (Parallelized Pass)
[cite_start]During training, the architecture operates in a single, highly parallelized step[cite: 121]. 

![Training Flowchart](/images/training_flowchart.png)
[cite_start]*Figure 4: Parallel Training Pipeline with Cross-Entropy Loss [cite: 130]*

* [cite_start]The full English source sentence wrapped in `<SOS>` and `<EOS>` tokens is encoded all at once into a rich contextual matrix[cite: 122, 123].
* [cite_start]Simultaneously, the target Italian sequence is fed into the decoder[cite: 124]. [cite_start]Because it is the training phase, we use **Teacher Forcing** and pass the entire target sequence through the decoder in a single pass rather than guessing word-by-word[cite: 125, 126].
* [cite_start]The decoder uses its inputs as Queries, matching them against the Keys and Values streamed from the encoder[cite: 127]. 
* [cite_start]Output logits are computed via a linear layer, and cross-entropy loss evaluates the errors against the target labels simultaneously, making training incredibly efficient[cite: 128, 129, 130].

### Inference (Autoregressive Loop)
[cite_start]While training happens all at once, inference (prediction) is a gradual, token-by-token loop[cite: 132].

![Inference Flowchart](/images/inference_flowchart.png)
[cite_start]*Figure 5: Step-by-Step Autoregressive Token Generation [cite: 142]*

1. [cite_start]The encoder runs **exactly once** to map the English sentence; since the source text never changes, we save compute by freezing this output[cite: 133, 134].
2. [cite_start]The decoder starts blind, receiving only the `<SOS>` token[cite: 135].
3. [cite_start]**Time Step 1:** The decoder processes `<SOS>`, applies the softmax layer, and selects the token with the highest probability (`"ti"`)[cite: 136].
4. [cite_start]**Time Step 2:** The model appends the generated word back to the input, passing `<SOS> ti` to generate `"amo"`[cite: 137].
5. [cite_start]This loop repeats sequentially (`<SOS> ti amo` $\rightarrow$ `"molto"`) until the model predicts the `<EOS>` token, signaling it to stop[cite: 138].

### Decoding Strategies
* [cite_start]**Greedy Strategy:** This baseline approach of always picking the single highest probability score at each step is simple but can be short-sighted[cite: 139, 140].
* [cite_start]**Beam Search:** To achieve better generation quality, advanced models track the top $B$ (beam width) most probable word combinations at every single step, systematically pruning away weaker sentence structures while preserving the most contextually accurate global paths[cite: 140, 141].

[cite_start]Always remember to read your documents with care, because *Attention is All You Need*[cite: 143]!
