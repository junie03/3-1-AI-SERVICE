import os
import shutil
import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from indexer import build_vector_db

load_dotenv()

app = FastAPI(title="강남대 RAG 챗봇 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 벡터 DB 로드 ──────────────────────────────────────────────────
print("FAISS 벡터 DB 로딩 중...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def load_vectorstore():
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

# ── LLM 설정 ──────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)

system_prompt = (
    "당신은 강남대학교의 학사 정보를 안내하는 친절한 AI 조교입니다. "
    "반드시 아래 제공된 Context만을 기반으로 질문에 답변하세요. "
    "Context에 없는 내용이라면 '죄송합니다. 제공된 학칙 PDF에서는 해당 정보를 찾을 수 없습니다.'라고 답변하세요.\n\n"
    "{dept_hint}"
    "Context:\n{context}"
)

web_system_prompt = (
    "당신은 강남대학교의 학사 정보를 안내하는 친절한 AI 조교입니다. "
    "아래는 강남대학교 공식 웹사이트에서 가져온 최신 정보입니다. "
    "이 내용을 바탕으로 질문에 친절하고 정확하게 답변하세요. "
    "날짜나 기간이 포함된 경우 그대로 전달하세요.\n\n"
    "웹사이트 Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

web_prompt = ChatPromptTemplate.from_messages([
    ("system", web_system_prompt),
    ("human", "{input}"),
])

# ── Request 모델 ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    department: Optional[str] = ""


# ── 쿼리 확장 (동의어/관련어 추가) ────────────────────────────────
QUERY_SYNONYMS = {
    "휴학": "휴학 휴학원 학적변동 휴학신청 휴학절차 휴학방법 제출 승인",
    "휴학 신청": "휴학 휴학원 학적변동 신청방법 휴학절차 제출서류",
    "복학": "복학 복학신청 복학절차 학적복귀 복학원",
    "수강신청": "수강신청 강좌신청 수강변경 수강취소 강의등록",
    "졸업": "졸업 졸업요건 졸업학점 졸업심사 학위취득",
    "장학": "장학 장학금 장학생 등록금 감면",
    "성적": "성적 학점 GPA 성적정정 성적이의",
}

def expand_query(question: str) -> str:
    """질문에 동의어/관련어를 추가해 검색 품질을 높입니다."""
    expanded = question
    for keyword, expansion in QUERY_SYNONYMS.items():
        if keyword in question:
            expanded = question + " " + expansion
            break  # 첫 번째 매칭만 적용
    return expanded


# ── 웹 크롤링 함수들 ───────────────────────────────────────────────

# 강남대학교 주요 학사 공지 URL 목록
KANGNAM_URLS = {
    "학점" : [
        "https://web.kangnam.ac.kr/menu/fd8c126ac0e81458620beb18302bc271.do?encMenuSeq=fcbd4013ab5b5238ed8ca9186a6bfb32"],
    "재수강" : [
        "https://web.kangnam.ac.kr/menu/fd8c126ac0e81458620beb18302bc271.do?encMenuSeq=d2fca573c753f30f9ae5c79dd740bdcd"],
    "수강신청": [
        "https://web.kangnam.ac.kr/menu/f19069e6134f8f8aa7f689a4a675e66f.do?searchMenuSeq=0",
        "https://web.kangnam.ac.kr/menu/f19069e6134f8f8aa7f689a4a675e66f.do?searchMenuSeq=0",
    ],
    "복수전공" : [
        "https://web.kangnam.ac.kr/menu/b2d1211af4999ac7a3ae1e11ad581860.do"],
    "학사일정": [
        "https://web.kangnam.ac.kr/menu/f19069e6134f8f8aa7f689a4a675e66f.do?paginationInfo.currentPageNo=1&searchMenuSeq=116&searchType=ttl&searchValue=",
    ],
    "휴학": [
        "https://web.kangnam.ac.kr/menu/12d2ee44cc4e95562f84a01bf953a054.do",
    ],
    "장학": [
        "https://web.kangnam.ac.kr/menu/062e41fba927c0c76d1c0e929f931016.do",
    ],
    "default": [
        "https://www.kangnam.ac.kr/web/portal/bbs/list.do?bIdx=98",
    ]
}

def select_urls(question: str) -> list[str]:
    """질문 키워드에 맞는 크롤링 URL 목록 반환"""
    for keyword, urls in KANGNAM_URLS.items():
        if keyword in question:
            return urls
    return KANGNAM_URLS["default"]


async def crawl_web_context(question: str) -> tuple[str, list[str]]:
    """
    강남대 웹사이트를 크롤링해 질문 관련 텍스트와 출처 URL 목록을 반환합니다.
    반환: (컨텍스트 문자열, [출처 URL, ...])
    """
    urls = select_urls(question)
    collected_texts: list[str] = []
    source_urls: list[str] = []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    async with httpx.AsyncClient(timeout=10, headers=headers, follow_redirects=True) as client:
        for url in urls:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except Exception as e:
                print(f"  ⚠️ 크롤링 실패 [{url}]: {e}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            rows = soup.select("table tbody tr, .board-list li, .bbs-list tr")
            board_texts: list[str] = []
            for row in rows[:20]:
                cells = row.find_all(["td", "li"])
                row_text = " | ".join(c.get_text(strip=True) for c in cells if c.get_text(strip=True))
                if row_text:
                    board_texts.append(row_text)

            if board_texts:
                collected_texts.append("[ 공지 목록 ]\n" + "\n".join(board_texts))
                source_urls.append(url)
                continue

            main_area = (
                soup.find("main")
                or soup.find(id="content")
                or soup.find(class_="content")
                or soup.body
            )
            if main_area:
                text = main_area.get_text(separator="\n", strip=True)
                if 50 < len(text) < 8000:
                    collected_texts.append(text[:3000])
                    source_urls.append(url)

    context = "\n\n---\n\n".join(collected_texts) if collected_texts else ""
    return context, source_urls


def is_not_found_answer(answer: str) -> bool:
    """LLM이 '정보를 찾을 수 없다'고 답변했는지 감지"""
    not_found_phrases = [
        "찾을 수 없습니다",
        "해당 정보를 찾을 수 없",
        "제공된 학칙 PDF에서는",
        "정보가 없습니다",
        "확인되지 않습니다",
    ]
    return any(phrase in answer for phrase in not_found_phrases)


# ── 프론트엔드 ─────────────────────────────────────────────────────
@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.get("/admin")
async def get_admin():
    return FileResponse("admin.html")

app.mount("/data", StaticFiles(directory="data"), name="data")


# ── 채팅 API ───────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    dept = (request.department or "").strip()

    # ★ 변경 1: k값을 4 → 10으로 증가
    k = 10

    # ★ 변경 2: 쿼리 확장 적용
    expanded_question = expand_query(request.question)
    print(f"\n[원본 질문]: {request.question}")
    print(f"[확장 쿼리]: {expanded_question}")

    # 1. 벡터 DB 검색 (학과 필터 적용)
    if dept:
        # ★ 변경 3: 학과 필터 시 후보를 30개로 확대
        all_docs = vectorstore.similarity_search(expanded_question, k=30)
        filtered = [
            d for d in all_docs
            if dept in d.metadata.get("department", "")
            or dept in d.metadata.get("source", "")
        ]
        docs = filtered[:k] if filtered else all_docs[:k]
        dept_hint = f"[필터: {dept} 관련 규정 우선 적용]\n\n"
    else:
        # ★ 변경 4: 유사도 점수 포함 검색으로 변경 (낮은 유사도 문서도 포함)
        docs_with_score = vectorstore.similarity_search_with_score(expanded_question, k=k)

        # 유사도 임계값 필터 (score가 낮을수록 유사도 높음, FAISS L2 기준)
        # 임계값을 넉넉하게 설정해 관련 문서 누락 방지
        SCORE_THRESHOLD = 1.8
        docs = [doc for doc, score in docs_with_score if score < SCORE_THRESHOLD]

        # 임계값 이하 문서가 없으면 상위 k개 그대로 사용
        if not docs:
            docs = [doc for doc, score in docs_with_score]

        print(f"  [유사도 점수]")
        for doc, score in docs_with_score:
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            print(f"    score={score:.4f} | {source} p.{page}")

        dept_hint = ""

    # 2. 컨텍스트 구성
    context_text = ""
    print(f"  [학과필터]: {dept or '전체'}")
    for doc in docs:
        source = doc.metadata.get("source", "알 수 없는 문서")
        page   = doc.metadata.get("page", None)
        page_info = f" (p.{page + 1})" if page is not None else ""
        print(f"  - PDF 출처: {source}{page_info}")
        context_text += f"[{source}{page_info}]: {doc.page_content}\n\n"

    # 3. LLM 1차 호출 (PDF 기반)
    formatted_prompt = prompt.format_messages(
        context=context_text,
        input=request.question,   # LLM에는 원본 질문 전달
        dept_hint=dept_hint
    )
    response = llm.invoke(formatted_prompt)
    answer = response.content

    # 4. PDF에서 못 찾은 경우 → 웹 크롤링 Fallback
    web_sources: list[str] = []
    used_web = False

    if is_not_found_answer(answer):
        print(f"  ⚠️ PDF에서 정보 없음 → 웹 크롤링 시작...")
        web_context, web_sources = await crawl_web_context(request.question)

        if web_context:
            print(f"  🌐 웹 크롤링 성공. 출처: {web_sources}")
            web_formatted = web_prompt.format_messages(
                context=web_context,
                input=request.question,
            )
            web_response = llm.invoke(web_formatted)
            answer = web_response.content
            used_web = True
        else:
            print("  ❌ 웹 크롤링도 실패 - 정보 없음으로 최종 응답")

    # 5. 출처 정리
    seen = set()
    pdf_sources = []
    for doc in docs:
        filename = doc.metadata.get("source", "학칙")
        page     = doc.metadata.get("page", None)
        page_num = page + 1 if page is not None else None
        key = f"{filename}:{page_num}"
        if key not in seen:
            seen.add(key)
            pdf_sources.append({"filename": filename, "page": page_num, "type": "pdf"})

    web_source_list = [{"url": u, "type": "web"} for u in web_sources]

    return {
        "answer": answer,
        "sources": pdf_sources if not used_web else [],
        "web_sources": web_source_list,
        "department_filter": dept or "전체",
        "source_type": "web" if used_web else "pdf",
    }


# ── 관리자 API ─────────────────────────────────────────────────────

@app.post("/admin/upload")
async def admin_upload(files: List[UploadFile] = File(...)):
    global vectorstore
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    saved = []
    for f in files:
        if not f.filename.endswith(".pdf"):
            continue
        dest = os.path.join(data_dir, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)
        print(f"  ✅ 저장됨: {dest}")

    build_vector_db()
    vectorstore = load_vectorstore()
    chunks = vectorstore.index.ntotal if hasattr(vectorstore, 'index') else 0

    return {
        "status": "ok",
        "uploaded_files": saved,
        "indexed_files": len(saved),
        "chunks": chunks
    }


@app.post("/admin/rebuild")
async def admin_rebuild():
    global vectorstore
    build_vector_db()
    vectorstore = load_vectorstore()
    chunks = vectorstore.index.ntotal if hasattr(vectorstore, 'index') else 0
    return {"status": "ok", "chunks": chunks}


@app.get("/admin/stats")
async def admin_stats():
    data_dir = "data"
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")] if os.path.exists(data_dir) else []
    total_size = sum(os.path.getsize(os.path.join(data_dir, f)) for f in pdf_files)
    chunks = vectorstore.index.ntotal if hasattr(vectorstore, 'index') else 0
    return {
        "total_files": len(pdf_files),
        "total_chunks": chunks,
        "total_size_mb": round(total_size / 1024 / 1024, 1)
    }


@app.get("/admin/files")
async def admin_files():
    data_dir = "data"
    if not os.path.exists(data_dir):
        return {"files": []}
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    chunk_counts: dict = {}
    try:
        for doc_id, doc in vectorstore.docstore._dict.items():
            src = doc.metadata.get("source", "")
            chunk_counts[src] = chunk_counts.get(src, 0) + 1
    except Exception:
        pass
    result = [{"name": f, "chunks": chunk_counts.get(f, "—")} for f in pdf_files]
    return {"files": result}
