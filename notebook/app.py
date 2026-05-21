import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List
import concurrent.futures
from rank_bm25 import BM25Okapi
import torch

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.embeddings import Embeddings

load_dotenv()

def auto_classify(doc) -> str:
    """
    Automatically classify a document as 'academic' or 'practitioner'
    using its PyMuPDF metadata and first-page text — no hardcoded author
    names required. Drop any PDF into data/pdf/ and it self-classifies.

    Signals checked (in priority order):
    1. PDF producer/creator metadata  → LaTeX / TeX / academic publishers
    2. First-page content keywords    → Abstract, doi, ISSN, journal markers
    3. Practitioner content keywords  → chapter, foreword, ISBN, price markers
    Falls back to 'unknown' when signals are ambiguous.
    """
    meta = doc.metadata
    text = doc.page_content.lower()

    # --- Metadata signals ---
    producer = (meta.get('producer', '') or '').lower()
    creator  = (meta.get('creator',  '') or '').lower()
    title    = (meta.get('title',    '') or '').lower()
    subject  = (meta.get('subject',  '') or '').lower()
    combined_meta = f"{producer} {creator} {title} {subject}"

    ACADEMIC_META = [
        'latex', 'tex', 'acm', 'ieee', 'elsevier', 'springer', 'arxiv',
        'wiley', 'sage', 'taylor', 'francis', 'oxford', 'cambridge',
        'harvard business', 'academy of management', 'journal',
    ]
    if any(s in combined_meta for s in ACADEMIC_META):
        return "academic"

    # --- First-page text signals ---
    ACADEMIC_TEXT = [
        'abstract', 'doi:', 'doi.org', 'issn', 'journal of', 'proceedings of',
        'et al.', 'keywords:', 'peer-reviewed', 'references\n', 'bibliography',
        'harvard business review', 'academy of management', 'vol.', 'no.',
        'university press', 'working paper', 'preprint',
    ]
    PRACTITIONER_TEXT = [
        'foreword', 'about the author', 'praise for', 'bestseller',
        'new york times', 'published by', 'all rights reserved',
        'table of contents', 'chapter one', 'chapter 1',
        'isbn-13', 'isbn-10', 'printed in', 'copyright ©',
        'first published', 'visit our website', 'trade paperback',
    ]

    academic_score     = sum(1 for s in ACADEMIC_TEXT     if s in text)
    practitioner_score = sum(1 for s in PRACTITIONER_TEXT if s in text)

    if academic_score >= 2 and academic_score >= practitioner_score:
        return "academic"
    if practitioner_score >= 2 and practitioner_score > academic_score:
        return "practitioner"
    if academic_score == 1 and practitioner_score == 0:
        return "academic"

    # Filename fallback: academic papers typically contain "et al" or "(20XX)".
    # Books and practitioner reports usually do not.
    import re
    fname = os.path.basename(meta.get('source', meta.get('file_path', ''))).lower()
    if 'et al' in fname or re.search(r'\(\d{4}\)', fname):
        return "academic"

    # No strong signal either way — treat as practitioner rather than unknown
    # so the LLM caveats it appropriately instead of presenting it as neutral.
    return "practitioner"


# FIX 3: Upgraded from paraphrase-MiniLM-L3-v2 (17M, wrong training objective)
# to BAAI/bge-base-en-v1.5 (109M, trained for semantic retrieval).
# NOTE: Delete faiss_db/ folder so the index is rebuilt with the new embeddings.
class SimpleEmbeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._encoder = SentenceTransformer(model_name, device=device)
        self.model = model_name  # string so RAGAS EmbeddingUsageEvent can serialise it
        print(f"Embeddings loaded on: {device}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encoder.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._encoder.encode([text], normalize_embeddings=True)[0].tolist()


class AdvancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        # FIX 5: Cross-encoder reranker replaces the LLM YES/NO batch grader.
        # CrossEncoder jointly encodes (query, doc) pairs — proper relevance scoring.
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.history = []

    def query(self, question: str, top_k: int = 5, min_score: float = 0.0, summarize: bool = False):
        # Short-circuit greetings before any retrieval or LLM calls.
        _GREETINGS = {"hi", "hello", "hey", "greetings", "howdy", "good morning", "good afternoon", "good evening", "how are you"}
        q_lower = question.strip().lower().rstrip("!?,.")
        if q_lower in _GREETINGS or (len(question.split()) <= 3 and any(g in q_lower for g in _GREETINGS)):
            greeting_reply = "Hello, I'm Alex — your Evidence-Based Management Advisor. What organizational challenge are we tackling today?"
            return {'answer': greeting_reply, 'summary': '', 'sources': [], 'retrieved_contexts': [], 'retrieved_docs': []}

        # HyDE query expansion
        expansion_prompt = f"""
        Re-write "{question}" into 3 search queries for a management and finance knowledge base.
        Rules:
        - Use only management, finance, psychology, and organisational behaviour terminology.
        - Do not invent concepts not grounded in these domains.
        - Each query must be a natural search phrase, not a sentence.
        1. Focus on the core concept or framework.
        2. Focus on practical application or evidence.
        3. Focus on a related principle or adjacent concept.
        Output only the 3 queries, one per line, no numbering or labels.
        """
        expanded_queries_response = self.llm.invoke(expansion_prompt)
        queries = [question] + expanded_queries_response.content.strip().split("\n")

        # FIX 4: Removed the redundant initial `docs = self.retriever.invoke(question)`
        # call and the unused `context_list`. The original question is already included
        # in `queries`, so parallel retrieval covers it — no separate call needed.
        all_docs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_query = {executor.submit(self.retriever.invoke, q): q for q in queries}
            for future in concurrent.futures.as_completed(future_to_query):
                all_docs.extend(future.result())

        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        print(f"Total unique docs from FAISS: {len(unique_docs)}")

        # BM25 hybrid re-score over the FAISS candidate pool
        tokenized_corpus = [doc.page_content.split() for doc in unique_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_candidates = bm25.get_top_n(question.split(), unique_docs, n=min(20, len(unique_docs)))

        # Cross-encoder reranking with relevance thresholds.
        # CHUNK_THRESHOLD: drop individual chunks that score this low (e.g. "Notes:", TOC pages).
        # COVERAGE_THRESHOLD: if the *best* chunk is still below this, the corpus doesn't
        # cover the question well — flag it so the prompt triggers the honest-limits response
        # instead of letting Alex hallucinate to fill the gap.
        CHUNK_THRESHOLD = -10.0
        COVERAGE_THRESHOLD = -8.5
        MIN_CHUNK_WORDS = 30  # filter copyright headers, TOC pages, "Notes:" pages

        low_coverage = False
        if bm25_candidates:
            # Pre-filter: drop chunks with too few words to contain real content
            content_candidates = [
                doc for doc in bm25_candidates
                if len(doc.page_content.split()) >= MIN_CHUNK_WORDS
            ]
            if not content_candidates:
                content_candidates = bm25_candidates  # fallback if all are short

            pairs = [(question, doc.page_content) for doc in content_candidates]
            scores = self.reranker.predict(pairs)
            scored = sorted(zip(scores, content_candidates), key=lambda x: x[0], reverse=True)

            # Drop chunks that are clearly irrelevant (below CHUNK_THRESHOLD)
            filtered = [(score, doc) for score, doc in scored if score >= CHUNK_THRESHOLD]
            relevant_chunks = [doc for score, doc in filtered[:top_k]]

            best_score = scored[0][0] if scored else -999
            if best_score < COVERAGE_THRESHOLD or not relevant_chunks:
                low_coverage = True
                relevant_chunks = [doc for score, doc in scored[:top_k]]

            print(f"Best reranker score: {best_score:.3f} | low_coverage: {low_coverage}")
            if low_coverage:
                print(f"[LOW COVERAGE] Question triggered hedge: '{question[:80]}...' | score={best_score:.3f}")
        else:
            low_coverage = True
            relevant_chunks = unique_docs[:top_k]

        # FIX 6 & FIX 7: Build context with evidence-type labels so the LLM prompt
        # can calibrate confidence per EBM evidence hierarchy.
        context_parts = []
        for doc in relevant_chunks:
            etype = doc.metadata.get("evidence_type", "unknown")
            if etype == "academic":
                label = "[ACADEMIC EVIDENCE]"
            elif etype == "practitioner":
                label = "[PRACTITIONER EVIDENCE]"
            else:
                label = "[EVIDENCE]"
            context_parts.append(f"{label}\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)

        # FIX 7: Replaced "No Citations" rule with Evidence Attribution.
        # Original rule ("Never mention author names, book titles, or years")
        # directly contradicted EBM's core requirement for verifiable outputs.
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
            - **Confidence Calibration**: The CONTEXT passages are labelled [ACADEMIC EVIDENCE] or [PRACTITIONER EVIDENCE] for YOUR internal use only.
              - Passages marked [ACADEMIC EVIDENCE]: speak with confidence ("Research shows...", "A controlled study found...").
              - Passages marked [PRACTITIONER EVIDENCE]: caveat appropriately ("Practitioners report...", "Field experience suggests...").
              - Tangential evidence: soften language ("The evidence leans toward...").
              - **CRITICAL**: Never write "[ACADEMIC EVIDENCE]" or "[PRACTITIONER EVIDENCE]" in your response — use only the natural language phrasings above.
            - **Contradictions**: If the CONTEXT conflicts with itself, surface the tension honestly rather than picking a side.
            - **No Invented Examples**: No hypothetical organizations, teams, or scenarios — even for clarity.

            ---

            ### EVIDENCE ATTRIBUTION
            - Indicate the evidence type naturally in your response — do NOT cite author names, years, or journals.
            - Your answer must be traceable: a reader should be able to verify each claim against the source passages.
            - If CONTEXT has no answer, say: *"Based on the evidence in our database, I don't have a specific framework for this yet."*

            ---

            ### STYLE
            - **Natural & Direct**: Talk like a sharp senior consultant — no rigid section headers.
            - **Empathetic Opener**: Briefly acknowledge the user's situation, then get straight to the point.
            - **Concise**: 150–250 words maximum. Bullet points for actions, minimal prose around them. Never exceed 300 words.
            - **No Jargon**: Plain business language. Define technical terms in one sentence if needed.

            ---

            ### OPERATING PRINCIPLES
            - **Zero External Knowledge**: Do not use general training data or theories outside the CONTEXT.
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

            {f'''COVERAGE NOTE — READ BEFORE RESPONDING:
            The retrieval system flagged this query as LOW COVERAGE. The passages above are only weakly related to the question asked.

            YOU MUST follow this exact structure and then STOP:
            1. State: "Based on the evidence in our database, I don\'t have a specific framework for [restate the exact topic]."
            2. State: "The closest evidence in our database covers [describe what the retrieved passages are actually about — be specific and literal]."
            3. Offer ONLY what the retrieved passages explicitly say — quote or closely paraphrase them.
            4. STOP. Do not offer recommendations, suggestions, or advice that goes beyond the retrieved text.

            FORBIDDEN on low-coverage responses:
            - "If I had to offer some advice..." — this signals gap-filling, which is prohibited.
            - "More research is needed" — not in the context, do not add it.
            - Any recommendation about what the company should do.
            - Any general management principles not stated in the retrieved passages.''' if low_coverage else ''}

            **ALEX'S EBM STRATEGIC ADVICE:**
            """
        response = self.llm.invoke(system_prompt)
        answer = response.content

        sources_info = []
        for doc in relevant_chunks:
            full_path = doc.metadata.get('source', 'Unknown Book')
            book_name = os.path.basename(full_path)
            evidence_type = doc.metadata.get('evidence_type', 'unknown')
            display_text = doc.page_content
            if "[KNOWLEDGE GRAPH:" in display_text:
                display_text = display_text.split("]\n")[-1]
            page_num = doc.metadata.get('page', 'Unknown')
            short_snippet = "\n".join(display_text.split("\n")[:10])
            sources_info.append({
                'book': book_name,
                'page': page_num + 1 if isinstance(page_num, int) else page_num,
                'snippet': short_snippet,
                'evidence_type': evidence_type,
            })

        summary = ""
        if summarize:
            if low_coverage:
                # For low-coverage queries the answer already says what the limits are.
                # Summarise faithfully — do not add recommendations or inferences.
                summary_prompt = (
                    f"Summarise the following response in 3-4 bullet points. "
                    f"Do NOT add any recommendations, suggestions, or conclusions that are not explicitly stated in the text below. "
                    f"If the response says the database lacks a framework, the summary must say the same.\n\n"
                    f"Response to summarise:\n{answer}"
                )
            else:
                summary_prompt = (
                    f"Summarise the following evidence-based management advice in 3-4 bullet points. "
                    f"Stay strictly within what is stated — do not add external knowledge.\n\n"
                    f"Response to summarise:\n{answer}"
                )
            summary = self.llm.invoke(summary_prompt).content

        return {
            'answer': answer,
            'summary': summary,
            'sources': sources_info,
            'retrieved_contexts': [doc.page_content for doc in relevant_chunks],
            'retrieved_docs': [
                {"book": d.metadata['source'], "page": d.metadata['page']}
                for d in relevant_chunks
            ]
        }


@st.cache_resource
def get_vector_db(_llm):
    BASE_DIR = Path(__file__).resolve().parent.parent
    PDF_DIR = BASE_DIR / "data" / "pdf"
    PERSIST_DIR = BASE_DIR / "faiss_db"
    persist_dir = str(PERSIST_DIR)
    pdf_path = str(PDF_DIR)

    embeddings = SimpleEmbeddings()

    if os.path.exists(persist_dir):
        print("--- FAISS folder found, loading database ---")
        return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

    print("--- Creating new FAISS database ---")
    st.warning("Database not found. Creating new FAISS database from PDFs. This may take a few minutes...")

    if not os.path.exists(pdf_path):
        st.error(f"PDF folder not found: {pdf_path}")
        st.stop()

    loader = DirectoryLoader(pdf_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()

    if not documents:
        st.error("PDF folder is empty")
        st.stop()

    # Auto-classify each document using its PDF metadata and first-page text.
    # No hardcoded author names needed — works for any new book dropped in.
    for doc in documents:
        doc.metadata['evidence_type'] = auto_classify(doc)

    # FIX 10: Removed KG extraction loop (1 LLM call + 1.5 s sleep per chunk,
    # but triples were only used for a plain keyword match — negligible benefit).
    # Better separators preserve sentence and paragraph boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(persist_dir)
    st.success("FAISS database created and saved successfully!")
    print("--- FAISS saved ---")

    return vectorstore


# FIX 9: Removed httpx.Client(verify=False) — SSL verification must not be
# disabled in a research/production system. If SSL errors occur, fix the
# certificate store rather than bypassing verification.
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("🤖 RAG ChatBot")

st.sidebar.header("Pipeline Settings")
show_summary = st.sidebar.checkbox("Show Summary", value=True)
num_chunks = st.sidebar.slider("Chunks to retrieve", 1, 10, 5)

st.sidebar.divider()
st.sidebar.header("Knowledge Base")

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR  = BASE_DIR / "data" / "pdf"
PERSIST_DIR = BASE_DIR / "faiss_db"

pdf_files = sorted(PDF_DIR.glob("**/*.pdf")) if PDF_DIR.exists() else []
st.sidebar.caption(f"📚 {len(pdf_files)} book(s) indexed")
for pdf in pdf_files:
    st.sidebar.markdown(f"- {pdf.name}")

if st.sidebar.button("🔄 Rebuild Index", help="Drop new PDFs into data/pdf/ then click this to re-index everything."):
    if PERSIST_DIR.exists():
        import shutil
        shutil.rmtree(PERSIST_DIR)
    get_vector_db.clear()
    st.sidebar.success("Index cleared — rebuilding now...")
    st.rerun()

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    request_timeout=120.0,
    max_retries=3,
)

vector_db = get_vector_db(llm)
adv_rag = AdvancedRAGPipeline(vector_db.as_retriever(), llm)

# FIX 8: Removed duplicate `if "messages" not in st.session_state` block.
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if "summary" in msg and msg["summary"]:
                with st.expander("✨ Summary"):
                    st.write(msg["summary"])
            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 Verify from Book (Actual PDF Text)"):
                    st.write("Find below the relevant passages from the document:")
                    sorted_sources = sorted(msg["sources"], key=lambda x: x['book'])
                    for source in sorted_sources:
                        etype = source.get('evidence_type', 'unknown')
                        badge = "🎓 Academic" if etype == 'academic' else "💼 Practitioner" if etype == 'practitioner' else "📄 Unclassified"
                        st.markdown(f"#### 📖 {source['book']} (Page {source['page']}) — {badge}")
                        st.info(source['snippet'])
                        st.divider()

if prompt := st.chat_input("Ask Alex about Management or Finance..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        result = adv_rag.query(prompt, top_k=num_chunks, summarize=show_summary)
        answer = result['answer']
        st.markdown(answer)

        greetings = ["hi", "hello", "hey", "greetings", "good morning", "how are you"]
        is_greeting = any(greet in prompt.lower() for greet in greetings) and len(prompt.split()) < 4
        no_info_keywords = ["I am sorry", "does not contain", "no information", "available in our database"]
        has_no_answer = any(key in answer for key in no_info_keywords)

        current_summary = ""
        current_sources = []

        if not is_greeting and not has_no_answer:
            if show_summary and result['summary']:
                current_summary = result['summary']
                with st.expander("✨ Summary"):
                    st.write(current_summary)

            current_sources = result['sources']
            with st.expander("🔍 Verify from Book (Actual PDF Text)"):
                st.write("Find below the relevant passages from the document:")
                sorted_fresh_sources = sorted(current_sources, key=lambda x: x['book'])
                for source in sorted_fresh_sources:
                    badge = "🎓 Academic" if source.get('evidence_type') == 'academic' else "💼 Practitioner"
                    st.markdown(f"#### 📖 {source['book']} (Page {source['page']}) — {badge}")
                    st.info(source['snippet'])
                    st.divider()

        elif has_no_answer:
            st.warning("No relevant sources found in the document.")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "summary": current_summary,
            "sources": current_sources,
        })
