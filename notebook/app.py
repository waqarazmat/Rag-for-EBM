import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List
import numpy as np
import httpx
import concurrent.futures
from rank_bm25 import BM25Okapi
import time
import torch

# LangChain & AI Imports   

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
import os

os.environ['CURL_CA_BUNDLE'] = '' # SSL check bypass karne ka naya tareeka

load_dotenv()

# --- Custom Embedding Class ---

class SimpleEmbeddings(Embeddings):

    def __init__(self, model_name: str = "paraphrase-MiniLM-L3-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Embeddings loaded on: {device}")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

# --- Advanced RAG Pipeline 
class AdvancedRAGPipeline:

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []

    def query(self, question: str, top_k: int = 5, min_score: float = 0.1, summarize: bool = False):

        # 1. Call chunks from database
        docs = self.retriever.invoke(question) 
       # 1. HyDE Question Expansion (Optimized to  variations instead of 3)
        expansion_prompt = f"""
        Re-write "{question}" into 3 strategic search queries.
        1. Focus on core frameworks.
        2. Focus on actionable solutions.
        Output only queries, one per line.
        """
        expanded_queries_response = self.llm.invoke(expansion_prompt)
        queries = [question] + expanded_queries_response.content.strip().split("\n")

        # --- OPTIMIZED: PARALLEL RETRIEVAL ---
        all_docs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Queries run parallel for improving latency, especially with multiple expansions
            future_to_query = {executor.submit(self.retriever.invoke, q): q for q in queries}
            for future in concurrent.futures.as_completed(future_to_query):
                all_docs.extend(future.result())

        # remove duplicates based on content
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        print(f"Total Unique Docs found: {len(unique_docs)}")


        context_list = [doc.page_content for doc in docs]

        tokenized_corpus = [doc.page_content.split() for doc in unique_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_hits = bm25.get_top_n(question.split(), unique_docs, n=len(unique_docs))



        relevant_chunks = []
        if bm25_hits:
            candidate_context = ""
            # we are sending top 8 BM25 hits to the LLM for batch grading (YES/NO) on relevance to the question
            for idx, doc in enumerate(bm25_hits[:8]): 
                candidate_context += f"ID: {idx}\nText: {doc.page_content[:300]}...\n\n"

            batch_grader_prompt = f"""
            Analyze these document snippets for relevance to the question: "{question}"
            For each ID, respond with YES if relevant or NO if irrelevant.
            Format: ID: YES or ID: NO
            Snippets:
            {candidate_context}
            """
            
            
            # only 1 LLM call 
            batch_grades = self.llm.invoke(batch_grader_prompt).content.upper()
            
            for idx, doc in enumerate(bm25_hits[:8]):
                graph_meta = str(doc.metadata.get('graph_triples', '')).lower()
                is_graph_match = any(word in graph_meta for word in question.lower().split() if len(word) > 3)
                
                if f"{idx}: YES" in batch_grades or is_graph_match:
                    relevant_chunks.append(doc)
                    if len(relevant_chunks) >= top_k:
                        break



        if not relevant_chunks:
            relevant_chunks = bm25_hits[:top_k]

        context = "\n".join([doc.page_content for doc in relevant_chunks])


        


        system_prompt = f"""
            ### ROLE
            You are **"Alex"**, a Senior Evidence-Based Management (EBM) Consultant.
            Your mission: help managers and executives make smarter decisions using only the scientific evidence and organizational data in the CONTEXT below.

            ---         

            ### THE EBM APPROACH
            1. **Diagnose**: Identify the core challenge beneath the user's question.
            2. **Synthesize**: Draw on frameworks and findings from the CONTEXT that directly apply.
            3. **Advise**: Translate evidence into concrete, actionable steps — not definitions, not theory.
            4. **Trade-offs**: If evidence supports multiple paths, present them honestly.

            ---

            ### FAITHFULNESS TO CONTEXT — CRITICAL
            - **Stay Inside the Context**: Only use what is explicitly stated or directly implied in the CONTEXT.
            - **No Gap-Filling**: Don't fill missing pieces with outside knowledge — work only with what's there.
            - **Partial Match**: If CONTEXT only partially covers the question, address what it does cover and flag the rest: *"The evidence covers [X] but doesn't speak to [Y] specifically."*
            - **Confidence Calibration**: Strong evidence = direct tone. Tangential evidence = soften language (e.g., *"The evidence leans toward..."*).
            - **Contradictions**: If the CONTEXT conflicts with itself, surface the tension honestly rather than picking a side.
            - **No Invented Examples**: No hypothetical organizations, teams, or scenarios — even for clarity.

            ---

            ### STYLE
            - **Natural & Direct**: Talk like a sharp senior consultant — no rigid section headers like "The Challenge" or "What the Evidence Says."
            - **Empathetic Opener**: Briefly acknowledge the user's situation, then get straight to the point.
            - **Concise**: Readable in under a minute. Bullet points for actions, minimal prose around them.
            - **No Jargon**: Plain business language. Define technical terms in one sentence if needed.
            - **No Citations**: Never mention author names, book titles, or years. Present everything as your own expert synthesis.

            ---

            ### OPERATING PRINCIPLES
            - **Zero External Knowledge**: Do not use general training data or theories outside the CONTEXT.
            - **Honest Limits**: If CONTEXT has no answer, say: *"Based on the evidence in our database, I don't have a specific framework for this yet."*
            - **Greeting**: If greeted, respond: *"Hello, I'm Alex — your Evidence-Based Management Advisor. What organizational challenge are we tackling today?"*

            ---

            ### NEVER
            - Fabricate case studies, data, or examples
            - Stretch the CONTEXT to fit a question it doesn't cover
            - Use "Based on my internal knowledge" or "It is commonly known that"
            - Give vague advice — always specify *what*, *how*, and *why the evidence supports it*

            ---

            CONTEXT:
            {context}

            QUESTION:
            {question}

            **ALEX'S EBM STRATEGIC ADVICE:**
            """
        response = self.llm.invoke(system_prompt)

        answer = response.content

        sources_info = []

        for doc in relevant_chunks:
            full_path = doc.metadata.get('source', 'Unknown Book')
            book_name = os.path.basename(full_path)

            display_text = doc.page_content

            if "[KNOWLEDGE GRAPH:" in display_text:

                display_text = display_text.split("]\n")[-1] 

            page_num = doc.metadata.get('page', 'Unknown')

           

            # to showing top 3 lines of the retrieved chunk as snippet in the UI, you can adjust this logic as needed

            short_snippet = "\n".join(display_text.split("\n")[:10])



            sources_info.append({
                'book': book_name,
                'page': page_num + 1,
                'snippet': short_snippet

            })

       

        # Summary (if wanted)

        summary = ""
        if summarize:
            summary_prompt = f"Summarize this: {answer}"
            summary = self.llm.invoke(summary_prompt).content

               
        return {
            'answer': answer,
            'summary': summary,
            'sources': sources_info,
            'retrieved_contexts': [doc.page_content for doc in relevant_chunks], 
            "retrieved_docs": [
                {"book": d.metadata['source'], "page": d.metadata['page']} 
                for d in relevant_chunks
            ]
            # 'retrieved_contexts': retrieved_contexts 
        }

@st.cache_resource # Streamlit caching for resource-intensive functions

def get_vector_db(_llm):

    # the path of the folder where the PDFs are stored and where the FAISS index will be saved
    BASE_DIR = Path(__file__).resolve().parent.parent
    PDF_DIR = BASE_DIR / "data" / "pdf"
    PERSIST_DIR = BASE_DIR / "faiss_db"
    persist_dir = str(PERSIST_DIR)
    pdf_path = str(PDF_DIR)

    embeddings = SimpleEmbeddings()

   

    # if FAISS folder already exists, load the database instead of creating it again (saves time on subsequent runs)
    if os.path.exists(persist_dir):
        print("--- FAISS Folder found, Loading Database ---")
        # allow_dangerous_deserialization=True to bypass the error related to loading FAISS index (use with caution in production)
        return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

   

    # if there is no existing FAISS database, create a new one from the PDFs (this will take time, especially with LLM calls for graph extraction)
    print("--- Creating NEW FAISS Database ---")
    st.warning("Database not found. Creating new FAISS database from PDFs. This may take a few minutes...")


    if not os.path.exists(pdf_path):
        st.error(f"PDF folder not found: {pdf_path}")
        st.stop()



    # Loading Documents 

    loader = DirectoryLoader(
        pdf_path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )

    documents = loader.load()

   

    if not documents:
        st.error("PDF folder is empty")
        st.stop()


   

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)

    graph_prompt = """

        You are an expert Knowledge Graph Engineer. Your task is to extract structured information from the provided text to build a Knowledge Graph.



        1. **Entities**: Identify all important concepts, organizations, names, or technical terms (e.g., "Hallucination", "NLG", "Ji et al.", "Intrinsic Hallucination").

        2. **Relationships**: For every pair of related entities, describe their relationship in a single concise phrase (e.g., "Hallucination" -> [is defined in the context of] -> "NLG").

        3. **Definitions**: If the text provides a formal definition, extract it clearly.



        Format the output as a list of Knowledge Triples:

        (Subject) -> [Relationship] -> (Object)

        Output ONLY the triples. Do not include introductory text like 'Here are the triples'.

        Text to analyze:

        {text_content}

        """


    for i, chunk in enumerate(splits):
        success = False
        retries = 0
        max_retries = 3
       

        while not success and retries < max_retries:
            try:
                text_content = chunk.page_content
                formatted_prompt = graph_prompt.format(text_content=text_content)
          
                # LLM Call
                response = _llm.invoke(formatted_prompt)
         
               # Cleaning and Saving
                clean_content = response.content.replace("Here are the extracted Knowledge Triples:", "").strip()
                chunk.metadata['graph_triples'] = clean_content
          
                print(f"✅ Processed Chunk {i+1}/{len(splits)}")
           
               # --- Rate Limit Buffer ---

                time.sleep(1.5)
                success = True

            
            except Exception as e:
                retries += 1
                wait_time = retries * 10
                print(f"Rate limit hit ya error: {e}. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

    print("--- All Chunks Processed with Graph Triples ---")

    # creating FAISS Database 

    vectorstore = FAISS.from_documents(splits, embeddings)

    vectorstore.save_local(persist_dir)
    st.success("FAISS Database has been created and saved successfully!")
    print("--- FAISS Saved Successfully ---")

    return vectorstore



# --- Streamlit UI Setup ---
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("🤖 RAG ChatBot")
# Sidebar settings

st.sidebar.header("Pipeline Settings")
show_summary = st.sidebar.checkbox("Show Summary", value=True)
num_chunks = st.sidebar.slider("Chunks to retrieve", 1, 10, 5)

llm = ChatGroq(

    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    http_client=httpx.Client(verify=False), 
    # request_timeout=60.0
    request_timeout=120.0,
    # connection retry enable 
    max_retries=3
)



vector_db = get_vector_db(llm)
adv_rag = AdvancedRAGPipeline(vector_db.as_retriever(), llm)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history with sources and summaries in expanders
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # if the message is from assistant, show summary and sources in expanders (only if they exist)
        if msg["role"] == "assistant":
            if "summary" in msg and msg["summary"]:
                with st.expander("✨ Summary"):
                    st.write(msg["summary"])
            
            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 Verify from Book (Actual PDF Text)"):
                    st.write("Find below the relevant passages from the document:")
                    # Sources ko A-Z sort karke dikhana (Jo humne discuss kiya tha)
                    sorted_sources = sorted(msg["sources"], key=lambda x: x['book'])
                    for source in sorted_sources:
                        st.markdown(f"#### 📖 {source['book']} (Page {source['page']})")
                        st.info(source['snippet'])
                        st.divider()


if prompt := st.chat_input("Ask Alex about Management or Finance..."):

    # show user message immediately in the chat interface
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # show alex reponse with sources and summary in expanders
        result = adv_rag.query(prompt, top_k=num_chunks, summarize=show_summary)
        answer = result['answer']
        st.markdown(answer)

        # Logic checks 
        greetings = ["hi", "hello", "hey", "greetings", "good morning", "how are you"]
        is_greeting = any(greet in prompt.lower() for greet in greetings) and len(prompt.split()) < 4
        no_info_keywords = ["I am sorry", "does not contain", "no information", "available in our database"]
        has_no_answer = any(key in answer for key in no_info_keywords)

        # Variables for session state
        current_summary = ""
        current_sources = []

        if not is_greeting and not has_no_answer:
            # Summary Display & Prep
            if show_summary and result['summary']:
                current_summary = result['summary']
                with st.expander("✨ Summary"):
                    st.write(current_summary)

            # Sources Display & Prep
            current_sources = result['sources']
            with st.expander("🔍 Verify from Book (Actual PDF Text)"):
                st.write("Find below the relevant passages from the document:")
                # Sorting for fresh display
                sorted_fresh_sources = sorted(current_sources, key=lambda x: x['book'])
                for source in sorted_fresh_sources:
                    st.markdown(f"#### 📖 {source['book']} (Page {source['page']})")
                    st.info(source['snippet'])
                    st.divider()
        
        elif has_no_answer:
            st.warning("No relevant sources found in the document.")

        # --- Save to Session State ---
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "summary": current_summary, # Summary saved
            "sources": current_sources   # Sources saved
        })

