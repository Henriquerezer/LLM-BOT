### Assistente Conversacional com RAG e Busca Semântica (Em andamento)

Este projeto tem como objetivo o desenvolvimento de um assistente conversacional baseado em Recuperação Aumentada por Geração (RAG) para consulta de informações extraídas de livros técnicos. A solução utiliza técnicas avançadas de Processamento de Linguagem Natural (NLP) para fornecer respostas contextualizadas e referenciadas, garantindo que o usuário tenha acesso à fonte original da informação.

### Tecnologias Utilizadas
- **ChromaDB**: Banco de dados vetorial utilizado para armazenar embeddings das informações extraídas dos livros. Permite uma busca eficiente por similaridade semântica.
- **Ollama**: Ferramenta de inferência de modelos de linguagem de grande escala (LLM), utilizada para gerar respostas contextualizadas.
- **RAG (Retrieval-Augmented Generation)**: Combina busca e geração de texto para aprimorar a precisão das respostas.
- **Busca Semântica**: Implementada via embeddings, permitindo localizar informações relevantes nos textos mesmo quando a consulta do usuário não coincide exatamente com os termos originais.

### Funcionamento
1. **Processamento dos Livros**: Os textos dos livros são extraídos e segmentados em trechos significativos.
2. **Geração de Embeddings**: Cada trecho é transformado em um vetor numérico utilizando modelos de embedding, permitindo a busca por similaridade.
3. **Armazenamento no ChromaDB**: Os vetores são armazenados em um banco de dados vetorial para recuperação eficiente.
4. **Consulta do Usuário**: Quando uma pergunta é feita, a busca semântica retorna os trechos mais relevantes.
5. **Geração da Resposta**: O modelo de linguagem (via Ollama) elabora uma resposta utilizando as informações recuperadas.
6. **Referência das Fontes**: Cada resposta inclui a página, o título do livro e o autor da informação utilizada.
7. Integração com Interface via Streamlit (Em Desenvolvimento) O objetivo final é oferecer uma interface conversacional intuitiva, permitindo interações fluidas e naturais com o assistente. A integração com Streamlit possibilitará um chatbot funcional e acessível.

### Exemplo de Implementação
#### Carregamento e Indexação dos Dados no ChromaDB
```python
from chromadb import Client
from sentence_transformers import SentenceTransformer

# Inicializa o ChromaDB
client = Client()
db = client.get_or_create_collection(name="livros_tecnicos")

# Modelo para gerar embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def indexar_texto(texto, pagina, livro, autor):
    embedding = embedding_model.encode(texto).tolist()
    db.add(
        documents=[texto],
        metadatas=[{"pagina": pagina, "livro": livro, "autor": autor}],
        embeddings=[embedding]
    )
```

#### Consulta e Recuperação de Informação
```python
def buscar_resposta(consulta):
    consulta_embedding = embedding_model.encode(consulta).tolist()
    resultados = db.query(embedding=consulta_embedding, n_results=3)
    
    resposta = ""
    for doc in resultados['documents']:
        resposta += f"{doc['document']} (Página: {doc['metadata']['pagina']}, Livro: {doc['metadata']['livro']}, Autor: {doc['metadata']['autor']})\n\n"
    
    return resposta
```

### Desafios e Complexidade
- **Segmentação Otimizada**: Determinar o tamanho adequado dos trechos armazenados para maximizar a recuperação de informação relevante.
- **Eficiência na Busca Semântica**: Ajuste dos embeddings e da métrica de similaridade para reduzir falsos positivos e garantir alta precisão.
- **Integração com Modelos de Linguagem**: Ajuste fino da combinação entre busca e geração de texto para evitar alucinações e garantir que a resposta esteja ancorada em dados confiáveis.

A implementação deste projeto exige a combinação de diversas técnicas avançadas de NLP, banco de dados vetorial e ajuste de modelos de linguagem, garantindo uma solução robusta e confiável para consultas baseadas em livros técnicos.

### Como Clonar o Repositório
Para testar ou contribuir com o projeto, siga os passos abaixo:
```bash
git clone https://github.com/usuario/repo_assistente_rag.git
cd repo_assistente_rag
pip install -r requirements.txt
python main.py
```

