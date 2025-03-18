# Transformer Mini Implementation

This file contains the implementation of a simplified **Transformer model**, including its core components such as **embedding, self-attention, multi-layer perceptron (MLP), and logits computation**. The model is designed for **sequence processing tasks**. file can be found `/inference/utils/transfromer.py`
---

## **Classes**  

### **1. Embedding**  
Handles **token and time embeddings** for input sequences.  

#### **Constructor Parameters:**  
- `embedding_dim` (*int*): Dimensionality of the embeddings. **Default = 64**  
- `n_vocal` (*int*): Vocabulary size. **Default = 128**  
- `context_len` (*int*): Length of the input sequence. **Default = 50**  

#### **Methods:**  
- `forward(x)`:  
  - **Input:** `x` (*Tensor of shape (batch, context_len)*)  
  - **Output:** Combined token and time embeddings of shape **(batch, context_len, embedding_dim)**  

---

### **2. Head**  
Implements a **single attention head** for the Transformer.  

#### **Constructor Parameters:**  
- `context_len` (*int*): Length of the input sequence. **Default = 50**  
- `embedding_dim` (*int*): Dimensionality of the embeddings. **Default = 64**  
- `attention_dim` (*int*): Dimensionality of the attention mechanism. **Default = 8**  
- `mode` (*str*): Either `"encoder"` or `"decoder"`. **Default = "encoder"**  

#### **Methods:**  
- `forward(x)`:  
  - **Input:** `x` (*Tensor of shape (batch, context_len, embedding_dim)*)  
  - **Output:** Attention output of shape **(batch, context_len, attention_dim)**  

---

### **3. SelfAttention**  
Implements **multi-head self-attention**.  

#### **Constructor Parameters:**  
- `context_l` (*int*): Length of the input sequence. **Default = 50**  
- `embedding_dim` (*int*): Dimensionality of the embeddings. **Default = 64**  
- `heads` (*int*): Number of attention heads. **Default = 8**  
- `mode` (*str*): Either `"encoder"` or `"decoder"`. **Default = "decoder"**  

#### **Methods:**  
- `forward(x)`:  
  - **Input:** `x` (*Tensor of shape (batch, context_len, embedding_dim)*)  
  - **Output:** Self-attention output of shape **(batch, context_len, embedding_dim)**  

---

### **4. MLP**  
Implements a **feedforward neural network** with **residual connections**.  

#### **Constructor Parameters:**  
- `embedding_dim` (*int*): Dimensionality of the embeddings. **Default = 64**  
- `mlp_multiplier` (*int*): Multiplier for the hidden layer size. **Default = 4**  

#### **Methods:**  
- `forward(x)`:  
  - **Input:** `x` (*Tensor of shape (batch, context_len, embedding_dim)*)  
  - **Output:** MLP output of shape **(batch, context_len, embedding_dim)**  

---

### **5. Logits**  
Computes the **final logits** for classification or prediction.  

#### **Constructor Parameters:**  
- `n_vocal` (*int*): Vocabulary size. **Default = 128**  
- `embedding_dim` (*int*): Dimensionality of the embeddings. **Default = 64**  

#### **Methods:**  
- `forward(x)`:  
  - **Input:** `x` (*Tensor of shape (batch, context_len, embedding_dim)*)  
  - **Output:** Logits of shape **(batch, context_len, n_vocal)**  

---

### **6. TransformerMini**  
Combines all components into a **mini Transformer model**.  

#### **Constructor Parameters:**  
- `context_l` (*int*): Length of the input sequence. **Default = 50**  
- `n_vocal` (*int*): Vocabulary size. **Default = 128**  
- `embedding_dim` (*int*): Dimensionality of the embeddings. **Default = 64**  
- `attention_heads` (*int*): Number of attention heads. **Default = 8**  
- `mode` (*str*): Either `"encoder"` or `"decoder"`. **Default = "decoder"**  

#### **Methods:**  
- `forward(x)`:  
  - **Input:** `x` (*Tensor of shape (batch, context_len)*)  
  - **Output:** Logits of shape **(batch, context_len, n_vocal)**  

---

## **🔹 Key Notes**  
✅ **Residual Connections**: Both **SelfAttention** and **MLP** use residual connections to improve gradient flow.  
✅ **Dropout**: Dropout layers are used throughout the model to prevent overfitting.  
✅ **Masked Attention**: The **Head** class supports masked attention for decoder mode using a **lower triangular matrix (`tril`)**.  

---

## **📌 Extensibility**  
This implementation is **modular** and can be extended for more complex Transformer architectures.
