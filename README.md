# RAG Implementation with Claude: Medicine documentation

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.11.6-blue.svg)
![Langchain Version](https://img.shields.io/badge/Langchain-0.3.x-brightgreen.svg)

## RAG: Enhancing LLMs with External Knowledge

This demonstration showcases how to implement **Retrieval-Augmented Generation (RAG)** with Anthropic's Claude model using Medicine documentation as an example domain. The technique demonstrates how to enhance LLM capabilities by integrating external knowledge sources with the model's inherent abilities.

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a powerful technique that significantly enhances LLM capabilities by:

1. **Supplementing internal knowledge** - RAG augments the model's pre-trained knowledge with external, domain-specific information
2. **Extending knowledge boundaries** - Allows access to information beyond the LLM's training data and cutoff date
3. **Increasing response specificity** - Enables highly focused answers based on authoritative external sources
4. **Integrating seamlessly** - Combines external knowledge retrieval with the LLM's reasoning capabilities

RAG is particularly valuable in specialized domains like medicine and pharmaceuticals, where precise, up-to-date information is critical.

## Technical Implementation

This demonstration implements a RAG pipeline with the following components:

### 1. Document Processing Pipeline

* **Data Source**: Thai pharmaceutical information from PDF document ("ข้อมูลยา 50 ชนิด.pdf")
* **Document Loading**: PyMuPDF extracts text content from the source
* **Text Chunking**: Documents are split into manageable segments with controlled overlap

### 2. Vector Embedding System

* **Embedding Generation**: OpenAI's embedding model converts text chunks into vector representations
* **Vector Storage**: Chroma database provides efficient similarity-based retrieval
* **Semantic Search**: Enables finding relevant information based on meaning, not just keywords

### 3. LLM Integration with Claude

* **Context Assembly**: Relevant document chunks are retrieved and formatted as context
* **Prompt Engineering**: Carefully crafted prompts guide Claude to use the retrieved information
* **Response Generation**: Claude produces answers that combine its reasoning capabilities with the specific retrieved information

## Technologies Used

* **Python 3.11.6**: Core programming language
* **Langchain**: Framework for building LLM applications and RAG pipelines
* **Anthropic Claude**: Large Language Model for response generation
* **OpenAI Embeddings**: Generates vector representations of text
* **Chroma**: Vector database for similarity search
* **PyMuPDF**: PDF processing library

## RAG Benefits Demonstrated

This demonstration highlights several key advantages of RAG:

1. **Knowledge Extension**: Supplements Claude's general knowledge with specific pharmaceutical information
2. **Temporal Extension**: Provides access to information regardless of Claude's training cutoff date
3. **Domain Specialization**: Enhances responses with domain-specific pharmaceutical knowledge
4. **Source Grounding**: Anchors responses in specific reference materials
5. **Thai Language Support**: Shows RAG's effectiveness with non-English content

## Setup 🛠️

1. Clone this project to your repository:

2. Create Virtual Environment (optional but recommended)

3. Activate Virtual Environment (venv) or Select Python Interpreture 📦 
   
```bash
source venv/bin/activate  # On MacOS use this with CMD
venv\Scripts\activate     # On Windows use this with CMD
```

4. Install dependencies ⬇️
```bash
pip install -r requirements.txt
```

5. Configure API key 🔑

```python
# Generate API KEY from Claude and OpenAI website and define as a variable.
os.environ["ANTHROPIC_API_KEY"] =  "YOUR_API_KEYS"
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEYS"
```


6. Run the analysis ▶️

```bash
python main.py
```



## RAG Technical Process Flow

1. **Document Ingestion**: 
   * Source Medicine documentation is loaded and processed
   * Text is extracted and split into semantic chunks

2. **Knowledge Base Creation**:
   * Text chunks are converted to vector embeddings
   * Embeddings are stored in the vector database

3. **Query Processing**:
   * User question is converted to the same vector space
   * Similarity search identifies relevant information chunks

4. **Contextual Response Generation**:
   * Retrieved chunks form a knowledge context
   * Claude generates responses using this context plus its own capabilities

This implementation demonstrates how RAG effectively bridges the gap between general LLM knowledge and specialized domain expertise.

## Advanced RAG Techniques

The demonstration includes several RAG optimization strategies:

* **Chunk Overlap Control**: Ensures semantic coherence between document segments
* **Relevance Parameters**: Configurable k-parameter for controlling retrieval quantity
* **Context Window Management**: Optimizes use of Claude's context window
* **Prompt Engineering**: Specialized prompts that guide Claude to properly utilize the retrieved context

## Future Technique Enhancements

* Query decomposition for complex questions
* Re-ranking of retrieved documents
* Hybrid retrieval combining sparse and dense vectors
* Multi-stage retrieval pipelines
* Self-critique and retrieval refinement

## License

This demonstration is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

**A technical demonstration of RAG implementation with Claude using Medicine documentation.**
