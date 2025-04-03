import fitz  # PyMuPDF
import os
import json
import chromadb
import ollama
import hashlib

# Configurações
PASTA_PDFS = "D:\\BOT_LLM_MLLM\\Books"  # Pasta onde estão os PDFs
DB_PATH = "./chroma_db"  # Caminho para o banco de dados ChromaDB

def extrair_texto_e_imagens_pdf(caminho_pdf):
    """Extrai texto e informações de imagens de um PDF"""
    doc = fitz.open(caminho_pdf)
    titulo = os.path.basename(caminho_pdf).replace(".pdf", "")
    autor = doc.metadata.get("author", "Desconhecido")

    paginas = []
    for num_pagina in range(len(doc)):
        texto = doc[num_pagina].get_text("text")
        imagens = doc[num_pagina].get_images(full=True)
        elementos_graficos = len(doc[num_pagina].get_drawings()) > 0

        tem_imagem = bool(imagens) or elementos_graficos

        paginas.append({
            "pagina": num_pagina + 1,
            "texto": texto,
            "tem_imagem": tem_imagem,
            "quantidade_imagens": len(imagens),
            "tem_graficos": elementos_graficos
        })

    return {"titulo": titulo, "autor": autor, "paginas": paginas}

def calcular_hash(conteudo):
    """Gera um hash único a partir de bytes ou texto"""
    if isinstance(conteudo, str):
        return hashlib.md5(conteudo.encode('utf-8')).hexdigest()
    elif isinstance(conteudo, bytes):
        return hashlib.md5(conteudo).hexdigest()
    else:
        raise ValueError("Tipo de entrada não suportado. Use str ou bytes.")

def gerar_embedding(texto):
    """Gera embedding usando o modelo nomic-embed-text do Ollama"""
    response = ollama.embeddings(model="nomic-embed-text", prompt=texto)
    return response["embedding"]

def carregar_pdfs_processados(collection):
    """Retorna uma lista de hashes de PDFs já processados"""
    documentos = collection.get()
    return set(documentos['ids']) if documentos['ids'] else set()

def processar_pasta_pdfs(pasta, chroma_client):
    """Processa todos os PDFs em uma pasta, ignorando os já processados"""
    collection = chroma_client.get_or_create_collection("livros_llm")
    hashes_processados = carregar_pdfs_processados(collection)
    
    pdfs_encontrados = 0
    pdfs_ja_processados = 0
    pdfs_processados = 0
    
    # Percorrer todos os arquivos na pasta
    for arquivo in os.listdir(pasta):
        if arquivo.lower().endswith('.pdf'):
            caminho_pdf = os.path.join(pasta, arquivo)
            pdfs_encontrados += 1
            
            # Gerar hash único do arquivo completo para verificar se já foi processado
            with open(caminho_pdf, 'rb') as f:
                hash_arquivo = calcular_hash(f.read())
            
            if hash_arquivo in hashes_processados:
                print(f"PDF {arquivo} já foi processado anteriormente. Ignorando...")
                pdfs_ja_processados += 1
                continue
            
            print(f"\nProcessando PDF: {arquivo}")
            try:
                # Extrair texto e informações
                dados_pdf = extrair_texto_e_imagens_pdf(caminho_pdf)
                
                # Adicionar ao banco
                adicionar_ao_banco(dados_pdf, collection, arquivo)
                pdfs_processados += 1
                
                # Salvar hash do PDF processado
                hashes_processados.add(hash_arquivo)
                
            except Exception as e:
                print(f"Erro ao processar {arquivo}: {str(e)}")
                continue
    
    print("\nResumo do processamento:")
    print(f"Total de PDFs encontrados: {pdfs_encontrados}")
    print(f"PDFs já processados (ignorados): {pdfs_ja_processados}")
    print(f"Novos PDFs processados: {pdfs_processados}")

def adicionar_ao_banco(dados_pdf, collection, nome_arquivo):
    """Adiciona os textos do PDF ao banco vetorial"""
    titulo = dados_pdf["titulo"]
    autor = dados_pdf["autor"]
    
    total_paginas = len(dados_pdf["paginas"])
    paginas_processadas = 0
    paginas_ignoradas = 0
    paginas_com_erro = 0

    print(f"Iniciando processamento do livro: {nome_arquivo}")
    print(f"Total de páginas: {total_paginas}")

    for pagina in dados_pdf["paginas"]:
        try:
            texto = pagina["texto"]
            pagina_num = pagina["pagina"]
            tem_imagem = pagina["tem_imagem"]

            if not texto.strip():
                print(f"Página {pagina_num}: Ignorada (texto vazio)")
                paginas_ignoradas += 1
                continue

            if tem_imagem:
                texto += "\n[Esta página contém uma imagem.]"

            embedding = gerar_embedding(texto)
            
            if not embedding:
                print(f"Página {pagina_num}: Erro ao gerar embedding")
                paginas_com_erro += 1
                continue

            doc_id = calcular_hash(texto)

            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[{
                    "titulo": titulo,
                    "autor": autor,
                    "pagina": pagina_num,
                    "texto": texto,
                    "tem_imagem": tem_imagem,
                    "nome_arquivo": nome_arquivo
                }]
            )
            print(f"Página {pagina_num}: Embedding criado com sucesso")
            paginas_processadas += 1

        except Exception as e:
            print(f"Página {pagina_num}: Erro ao processar - {str(e)}")
            paginas_com_erro += 1
            continue

    print(f"Resumo do {nome_arquivo}:")
    print(f"Páginas processadas: {paginas_processadas}")
    print(f"Páginas ignoradas: {paginas_ignoradas}")
    print(f"Páginas com erro: {paginas_com_erro}")

if __name__ == "__main__":
    # Inicializar ChromaDB
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    # Processar todos os PDFs na pasta
    processar_pasta_pdfs(PASTA_PDFS, chroma_client)
    
    print("Processamento concluído!")