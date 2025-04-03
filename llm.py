#%%
import ollama
import chromadb
import hashlib
import json
import time

def calcular_hash(texto):
    """Gera um hash do texto para evitar duplicação."""
    return hashlib.md5(texto.encode()).hexdigest()

# Conectar ao banco ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("livros_llm")

def gerar_embedding(texto):
    """Gera embeddings usando o modelo de embedding no Ollama."""
    inicio = time.time()
    resposta = ollama.embeddings(model="nomic-embed-text", prompt=texto)
    print(f"Tempo para gerar embedding: {time.time() - inicio:.2f} segundos")
    return resposta["embedding"]

def adicionar_ao_banco(dados_pdf):
    """Adiciona dados extraídos dos PDFs ao banco vetorial."""
    titulo = dados_pdf["titulo"]
    autor = dados_pdf["autor"]

    for pagina in dados_pdf["paginas"]:
        texto = pagina["texto"]
        pagina_num = pagina["pagina"]
        tem_imagem = pagina["tem_imagem"]

        if tem_imagem:
            texto += "\n[Esta página contém uma imagem.]"

        embedding = gerar_embedding(texto)
        doc_id = calcular_hash(texto)

        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{
                "titulo": titulo,
                "autor": autor,
                "pagina": pagina_num,
                "texto": texto,
                "tem_imagem": tem_imagem
            }]
        )

def buscar_no_banco(pergunta, top_k=3):
    """Busca os documentos mais relevantes no banco vetorial."""
    inicio = time.time()
    embedding_pergunta = gerar_embedding(pergunta)
    resultados = collection.query(
        query_embeddings=[embedding_pergunta],
        n_results=top_k
    )
    print(f"Tempo para buscar no banco: {time.time() - inicio:.2f} segundos")

    contexto = ""
    paginas = set()
    for r in resultados["metadatas"][0]:
        contexto += f"\nPágina {r['pagina']} ({r['titulo']} - {r['autor']}):\n{r['texto']}\n"
        paginas.add(r["pagina"])
    
    return contexto, sorted(paginas)

def responder_pergunta(pergunta, mostrar_raciocinio=True):
    """Gera uma resposta usando a LLM com base nas informações do RAG."""
    inicio = time.time()
    contexto, paginas = buscar_no_banco(pergunta)
    resultados = collection.query(
        query_embeddings=[gerar_embedding(pergunta)],
        n_results=3
    )

    instrucao_raciocinio = """
    Por favor, mostre seu processo de raciocínio usando as tags <think> antes de dar a resposta final.
    """ if mostrar_raciocinio else ""

    prompt = f"""
    Você é um assistente de Henrique Rezer, especializado em LLMs e MLLMs.
    {instrucao_raciocinio}
    Responda com base nas informações fornecidas:
    {contexto}
    
    Pergunta: {pergunta}
    Resposta:
    """
    resposta = ollama.chat(model="deepseek-r1:14b", messages=[{"role": "user", "content": prompt}])
    print(f"Tempo para gerar resposta: {time.time() - inicio:.2f} segundos")
    
    # Gerar referências
    referencias = []
    for r in resultados["metadatas"][0]:
        referencias.append(f"{r['titulo']} - {r['autor']} (Página {r['pagina']})")
    
    return resposta["message"]["content"], referencias

# Loop interativo para perguntas
print("Assistente de LLMs e MLLMs ativado! Digite sua pergunta ou 'sair' para encerrar.")
print("Use 'raciocinio on/off' para controlar a exibição do processo de raciocínio.")
mostrar_raciocinio = True

while True:
    pergunta = input("\nPergunta: ")
    if pergunta.lower() == "sair":
        break
    elif pergunta.lower() == "raciocinio on":
        mostrar_raciocinio = True
        print("Exibição do raciocínio ativada!")
        continue
    elif pergunta.lower() == "raciocinio off":
        mostrar_raciocinio = False
        print("Exibição do raciocínio desativada!")
        continue
        
    resposta, referencias = responder_pergunta(pergunta, mostrar_raciocinio)
    print(f"\nResposta: {resposta}")
    print(f"\nReferências: {', '.join(referencias)}")




# %%

# Achei o processo um pouco lento, vou tentar colocar prints para ver o que está demorando

# posso mexer no tamanho da resposta ? 

# Qual o limite de tokens que o ollama suporta ? 

# Qual o limite do contexto ?

# Por estar em uma vector store, o contexto ocupa menos memoria / tokens ?