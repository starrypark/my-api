# backend.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, AsyncGenerator
import os
import io
from uuid import uuid4
import base64

# ---------- LangChain ----------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

# ---------- PDF ----------
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# =============================
# 1) 키 로드
# =============================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 없으면 api_key.txt에서 읽어오기
if not OPENAI_API_KEY and os.path.exists("api_key.txt"):
    with open("api_key.txt", "r", encoding="utf-8") as f:
        OPENAI_API_KEY = f.read().strip()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY   # 이후에도 환경변수처럼 접근 가능

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY가 설정되어 있지 않습니다. "
        "api_key.txt 파일을 만들거나, .env/환경변수에 추가하세요."
    )


# =============================
# 2) FastAPI
# =============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://jaeminbag12.shinyapps.io",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:3838",   # Shiny 로컬 (RStudio 기본 포트)
        "http://localhost:3838",
        "http://127.0.0.1:7173",   # Shiny 로컬 (RStudio 기본 포트)
        "http://localhost:7173",
    ],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# =============================
# 3) LLM & 체인
# =============================
llm = ChatOpenAI(model="gpt-4o", temperature=0)

DEFAULT_SYSTEM = "You are a helpful assistant. Keep answers concise and accurate."

prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{question}"),
])

parsed_chain = prompt | llm | StrOutputParser()

# 세션별 메모리 (데모: 인메모리)
_SESSION_STORES: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _SESSION_STORES:
        _SESSION_STORES[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORES[session_id]

# 메모리 얹은 체인
chain_with_memory = RunnableWithMessageHistory(
    parsed_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# =============================
# 4) 업로드 파일 저장소 (데모용)
# =============================
# {"id": {"filename":..., "content_type":..., "text":..., "bytes":..., "size": int}}
_FILE_STORE: Dict[str, Dict] = {}

# =============================
# 5) 스키마
# =============================
class ChatIn(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    system_prompt: Optional[str] = None
    file_id: Optional[str] = None  # 업로드 파일 참조용

# =============================
# 6) 헬스체크/에코
# =============================
@app.get("/ping")
def ping():
    return {"ok": True}

@app.post("/echo")
def echo(body: ChatIn):
    return {"answer": f"echo: {body.question}"}

# =============================
# 7) 파일 업로드
# =============================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    허용된 타입만 업로드. PDF는 텍스트 추출, 이미지는 바이트 보관.
    """
    allowed = {
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/webp",
    }
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"허용되지 않은 파일 형식입니다: {file.content_type}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    file_id = str(uuid4())
    rec = {
        "filename": file.filename,
        "content_type": file.content_type,
        "bytes": None,
        "text": None,
        "size": len(data),
    }

    if file.content_type == "application/pdf":
        if PdfReader is None:
            raise HTTPException(status_code=500, detail="PDF 파서를 사용할 수 없습니다. pypdf를 설치하세요: pip install pypdf")
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = []
            for p in reader.pages:
                # extract_text가 None일 수 있어 방어
                pages.append(p.extract_text() or "")
            rec["text"] = "\n".join(pages).strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF 파싱 실패: {e}")
    else:
        # 이미지 등은 바이트로 보관하여 추후 비전 모델에 전달
        rec["bytes"] = data

    _FILE_STORE[file_id] = rec
    return {
        "ok": True,
        "file_id": file_id,
        "filename": rec["filename"],
        "content_type": rec["content_type"],
        "size": rec["size"]
    }

# =============================
# 8) 비스트리밍 대화 (/chat)
# =============================
@app.post("/chat")
def chat(body: ChatIn):
    if os.environ.get("OPENAI_API_KEY", "") == "":
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY가 설정되어 있지 않습니다. PowerShell: $env:OPENAI_API_KEY=\"...\""
        )

    try:
        session_id = body.session_id or "default"
        system_prompt = body.system_prompt or DEFAULT_SYSTEM

        # 업로드 파일 컨텍스트 주입
        file_context = ""
        if body.file_id:
            rec = _FILE_STORE.get(body.file_id)
            if not rec:
                raise HTTPException(status_code=400, detail="유효하지 않은 file_id 입니다.")

            # PDF면 텍스트 스니펫, 이미지면 메타정보만 주입
            if rec["content_type"] == "application/pdf" and rec.get("text"):
                snippet = rec["text"][:6000]  # 너무 길면 앞부분만 사용 (운영: 청크/검색 권장)
                file_context = f"\n\n[Attached PDF excerpt]\n{snippet}\n"
            elif rec["bytes"] is not None:
                file_context = f"\n\n[Attached image: {rec['filename']} ({rec['content_type']}, {rec['size']} bytes)]"

        text = chain_with_memory.invoke(
            {"question": body.question + file_context, "system_prompt": system_prompt},
            config={"configurable": {"session_id": session_id}},
        )
        return {"answer": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================
# 9) 스트리밍 대화 (/chat_stream)
# =============================
@app.post("/chat_stream")
async def chat_stream(body: ChatIn):
    """
    chunked text 스트리밍. Content-Type: text/plain
    """
    if os.environ.get("OPENAI_API_KEY", "") == "":
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY가 설정되어 있지 않습니다. PowerShell: $env:OPENAI_API_KEY=\"...\""
        )

    session_id = body.session_id or "default"
    system_prompt = body.system_prompt or DEFAULT_SYSTEM

    # 업로드 파일 컨텍스트 주입
    file_context = ""
    if body.file_id:
        rec = _FILE_STORE.get(body.file_id)
        if not rec:
            raise HTTPException(status_code=400, detail="유효하지 않은 file_id 입니다.")
        if rec["content_type"] == "application/pdf" and rec.get("text"):
            snippet = rec["text"][:6000]
            file_context = f"\n\n[Attached PDF excerpt]\n{snippet}\n"
        elif rec["bytes"] is not None:
            file_context = f"\n\n[Attached image: {rec['filename']} ({rec['content_type']}, {rec['size']} bytes)]"

    async def token_generator() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in chain_with_memory.astream(
                {"question": body.question + file_context, "system_prompt": system_prompt},
                config={"configurable": {"session_id": session_id}},
            ):
                yield (chunk).encode("utf-8")
        except Exception as e:
            yield f"\n[STREAM ERROR] {str(e)}".encode("utf-8")

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")

# =============================
# 10) 이미지 분석 (멀티모달) 예시 엔드포인트
# =============================
@app.post("/analyze_image")
def analyze_image(file_id: str, question: str = "이 이미지를 설명해줘."):
    """
    멀티모달 호출 연결용 예시.
    LangChain OpenAI 멀티모달 메시지 포맷은 버전에 따라 달라질 수 있으므로,
    여기서는 base64 data URL로 이미지를 넣는 구조를 개념적으로 보여준다.
    """
    rec = _FILE_STORE.get(file_id)
    if not rec or not rec["bytes"] or not rec["content_type"].startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 file_id를 전달하세요.")

    # base64 data URL (개념 예시)
    b64 = base64.b64encode(rec["bytes"]).decode()
    image_data_url = f"data:{rec['content_type']};base64,{b64}"

    # LangChain의 ChatOpenAI 멀티모달 입력은 버전에 따라 다음과 같은 형식이 가능:
    # messages = [
    #     {"role": "system", "content": DEFAULT_SYSTEM},
    #     {"role": "user", "content": [
    #         {"type": "text", "text": question},
    #         {"type": "image_url", "image_url": image_data_url}
    #     ]},
    # ]
    # resp = llm.invoke(messages)
    # return {"answer": resp}

    # 최소 동작 보장(배선 전): 이미지 수신 확인만 반환
    return {
        "ok": True,
        "note": "멀티모달 호출 배선 전 상태입니다. LangChain/오픈AI 멀티모달 메시지 포맷에 맞춰 llm.invoke를 연결하세요.",
        "filename": rec["filename"],
        "content_type": rec["content_type"],
        "size": rec["size"],
        "question": question
    }

# =============================
# 11) 메모리 리셋
# =============================
@app.post("/reset_memory")
def reset_memory(body: ChatIn):
    sid = body.session_id or "default"
    if sid in _SESSION_STORES:
        del _SESSION_STORES[sid]
    return {"ok": True, "session_id": sid, "message": "memory cleared"}

# =============================
# 12) 샘플 홈 UI (업로드 포함)
# =============================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>FastAPI Chat</title>
</head>
<body style="font-family:sans-serif; max-width:820px; margin:40px auto;">
<h2>FastAPI Chat (Memory + Stream + Upload)</h2>

<div style="margin:8px 0">
  <input id="sid" placeholder="session_id (e.g., user1)" style="width:49%">
  <input id="sys" placeholder="system_prompt (optional)" style="width:49%">
</div>

<div style="margin:8px 0">
  <input type="file" id="fileInput">
  <button id="btnUpload">Upload</button>
  <span id="uploadInfo" style="margin-left:8px; color:#555"></span>
</div>

<div id="log" style="border:1px solid #ddd; padding:10px; height:320px; overflow:auto;"></div>
<textarea id="q" rows="3" style="width:100%; margin-top:10px;" placeholder="질문"></textarea>

<div style="margin-top:8px; display:flex; gap:8px; flex-wrap: wrap;">
  <button id="btn">Send (non-stream)</button>
  <button id="btnStream">Send (stream)</button>
  <button id="reset">Reset Memory</button>
  <button id="analyzeImage">Analyze Image</button>
</div>

<script>
const log = document.getElementById('log');
const q = document.getElementById('q');
const sid = document.getElementById('sid');
const sys = document.getElementById('sys');
const fileInput = document.getElementById('fileInput');
const uploadInfo = document.getElementById('uploadInfo');

let lastFileId = null;

function println(html) {
  log.innerHTML += "<div>" + html + "</div>";
  log.scrollTop = log.scrollHeight;
}

// 업로드
document.getElementById('btnUpload').onclick = async () => {
  if(!fileInput.files[0]) {
    alert("파일을 선택하세요.");
    return;
  }
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);

  try {
    const r = await fetch("/upload", { method: "POST", body: fd });
    const data = await r.json();
    if(!r.ok) throw new Error(data.detail || r.statusText);
    lastFileId = data.file_id;
    uploadInfo.textContent = `Uploaded: ${data.filename} (${data.content_type}, ${data.size} bytes), file_id=${data.file_id}`;
    println("<i>Uploaded file: " + data.filename + " (file_id=" + data.file_id + ")</i>");
  } catch (e) {
    uploadInfo.textContent = e.message;
    println("<span style='color:#b00020'><b>Upload Error:</b> " + e.message + "</span>");
  }
};

// 비스트리밍
document.getElementById('btn').onclick = async () => {
  const question = q.value.trim();
  if(!question) return;
  println("<b>You:</b> " + question);
  q.value = "";

  try {
    const r = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        question,
        session_id: sid.value || "default",
        system_prompt: sys.value || null,
        file_id: lastFileId || null
      })
    });
    const data = await r.json();
    if(!r.ok) throw new Error(data.detail || r.statusText);
    println("<span style='color:#1f4c7c'><b>Assistant:</b> " + (data.answer || "") + "</span>");
  } catch (e) {
    println("<span style='color:#b00020'><b>Error:</b> " + e.message + "</span>");
  }
};

// 스트리밍
document.getElementById('btnStream').onclick = async () => {
  const question = q.value.trim();
  if(!question) return;
  println("<b>You:</b> " + question);
  q.value = "";

  const r = await fetch("/chat_stream", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({
      question,
      session_id: sid.value || "default",
      system_prompt: sys.value || null,
      file_id: lastFileId || null
    })
  });

  if(!r.ok || !r.body) {
    const data = await r.json().catch(()=> ({}));
    println("<span style='color:#b00020'><b>Error:</b> " + (data.detail || r.statusText) + "</span>");
    return;
  }

  const reader = r.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let assistantLine = "<span style='color:#1f4c7c'><b>Assistant:</b> ";

  const span = document.createElement("div");
  span.innerHTML = assistantLine;
  log.appendChild(span);

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    assistantLine += chunk;
    span.innerHTML = assistantLine.replace(/\\n/g, "<br>");
    log.scrollTop = log.scrollHeight;
  }

  span.innerHTML = assistantLine + "</span>";
};

// 메모리 리셋
document.getElementById('reset').onclick = async () => {
  try {
    const r = await fetch("/reset_memory", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ question: "", session_id: sid.value || "default" })
    });
    const data = await r.json();
    if(!r.ok) throw new Error(data.detail || r.statusText);
    println("<i>Memory cleared for session_id=" + (sid.value || "default") + "</i>");
  } catch (e) {
    println("<span style='color:#b00020'><b>Error:</b> " + e.message + "</span>");
  }
};

// 이미지 분석 (예시)
document.getElementById('analyzeImage').onclick = async () => {
  if(!lastFileId) {
    alert("먼저 이미지를 업로드하세요.");
    return;
  }
  try {
    const params = new URLSearchParams({ file_id: lastFileId, question: "이 이미지를 설명해줘." });
    const r = await fetch("/analyze_image?" + params.toString(), { method: "POST" });
    const data = await r.json();
    if(!r.ok) throw new Error(data.detail || r.statusText);
    println("<i>Analyze Image:</i> " + JSON.stringify(data));
  } catch (e) {
    println("<span style='color:#b00020'><b>Analyze Error:</b> " + e.message + "</span>");
  }
};
</script>
</body>
</html>
"""
