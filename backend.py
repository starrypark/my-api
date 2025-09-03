## Note : 서버 열기
# uvicorn backend:app --reload --port 8000 --host 127.0.0.1

# backend.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict
import os
import io
import re
from uuid import uuid4
import base64

# ---------- LangChain ----------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

# --- Agent 도구/메시지/유틸 ---
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from pydantic import Field
import datetime, math, ast, time, logging, json

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("agent")


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
# X) Tools (Agent가 필요 시 자동 사용)
# =============================

# (1) 안전 계산기 ---------------------------------------------------------
_ALLOWED_FUNCS = {
    "abs": abs, "round": round, "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos,
    "atan": math.atan, "pow": pow
}
_ALLOWED_NAMES = {**_ALLOWED_FUNCS, "pi": math.pi, "e": math.e}

def _safe_eval(expr: str) -> float:
    # ^를 **로 치환 (수학 표기 지원)
    expr = expr.replace("^", "**")
    # AST 파싱 및 화이트리스트 검증
    node = ast.parse(expr, mode="eval")
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load, ast.Call,
        ast.Name, ast.Pow, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
        ast.USub, ast.UAdd, ast.Constant
    )
    for sub in ast.walk(node):
        if not isinstance(sub, allowed_nodes):
            raise ValueError("unsupported expression")
        if isinstance(sub, ast.Call):
            if not isinstance(sub.func, ast.Name) or sub.func.id not in _ALLOWED_FUNCS:
                raise ValueError("unsupported function")
        if isinstance(sub, ast.Name) and sub.id not in _ALLOWED_NAMES:
            raise ValueError("unsupported identifier")
    return eval(compile(node, "<expr>", "eval"), {"__builtins__": {}}, _ALLOWED_NAMES)

class CalcInput(BaseModel):
    expression: str = Field(..., description="예: 2*(3+4)/sqrt(2)")

@tool("calc", args_schema=CalcInput)
def calc_tool(expression: str) -> str:
    """수식 계산이 필요할 때 사용."""
    try:
        return str(_safe_eval(expression))
    except Exception as e:
        return f"[calc error] {e}"

# (2) 업로드 PDF 간단 검색 -----------------------------------------------
class PDFSearchInput(BaseModel):
    query: str = Field(..., description="키워드 또는 짧은 문장")

@tool("search_pdf", args_schema=PDFSearchInput)
def search_pdf_tool(query: str) -> str:
    """업로드한 PDF 텍스트에서 키워드 주변 스니펫을 찾아 반환."""
    q = query.lower()
    hits = []
    for fid, rec in _FILE_STORE.items():
        if rec.get("content_type") == "application/pdf" and rec.get("text"):
            text = rec["text"]
            low = text.lower()
            idx = low.find(q)
            if idx >= 0:
                s = max(0, idx - 160)
                e = min(len(text), idx + len(q) + 160)
                snippet = text[s:e].replace("\n", " ")
                hits.append(f"[{rec['filename']} | id={fid}] …{snippet}…")
    return "\n".join(hits[:5]) if hits else "검색 결과 없음"

# (3) 현재 시각 ----------------------------------------------------------
@tool("now")
def now_tool() -> str:
    """현재 서버 시각을 문자열로 반환."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# (4) 모의 생존 확률 계산기 -----------------------------------------------
class MockSurvivalInput(BaseModel):
    text: str = Field(..., description="질문/환자 설명 전체 문장")

def _roman_to_int(s: str) -> Optional[int]:
    if not s: return None
    s = s.upper().strip()
    return {"I":1, "II":2, "III":3, "IV":4}.get(s)

# 추출 결과 스키마 정의
class SurvivalFields(BaseModel):
    age: Optional[int] = Field(None, description="나이")
    sex: Optional[int] = Field(None, description="성별: male=0, female=1")
    stage: Optional[int] = Field(None, description="SEER stage: 1=Localized, 2=Regional, 3=Distant, 4=Unknown")
    year: Optional[int] = Field(None, description="n-year survival probability (기간, 달력연도가 아님)")

@tool("mock_survival", args_schema=MockSurvivalInput)
def mock_survival_tool(text: str) -> str:
    """
    A dedicated tool that calculates simulated survival probabilities from text 
    containing patient age, sex, SEER stage, and year.
    You must use this tool exclusively for the calculation. 
    The output should contain only probability values.
    (Medical significance: none, demo only.)

    SEER stage coding:
    Localized = 1
    Regional = 2
    Distant = 3
    Unknown = 4

    sex coding : male = 0, female = 1
    """

    # LLM에 필드 추출 요청
    extraction_prompt = f"""
    Extract survival fields from the text below.
    Respond strictly in JSON format with keys: age, sex, stage, year.
    
    - age: integer
    - sex: male=0, female=1
    - stage: 1=Localized, 2=Regional, 3=Distant, 4=Unknown
    - year: survival duration in years (not calendar year)

    Text: {text}
    """

    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    try:
        parsed = SurvivalFields.parse_raw(response.content)
    except Exception:
        # LLM이 JSON을 못 주면 fallback
        return "probability=N/A"

    age = parsed.age or 0
    stage = parsed.stage or 0
    sex = parsed.sex or 0
    year = parsed.year or 0

    prob = (age + stage + sex + year) / 100
    return f"{prob:.3f}"
    
# Tool 정의
TOOLS = [calc_tool, search_pdf_tool, now_tool, mock_survival_tool]

# =============================
# X) Agent 실행 함수 (tool-calling)
# =============================
def run_agent(question: str, system_prompt: str, session_id: str, file_context: str = "", debug: bool = False):
    hist = get_session_history(session_id)
    llm_tools = llm.bind_tools(TOOLS)

    # 디버그용 트레이스 버퍼
    trace = {
        "session_id": session_id,
        "question": question,
        "tool_calls": [],   # 각 툴 호출 기록
        "rounds": 0,        # 모델-툴 왕복 라운드 수
    }

    def _truncate(obj, n=500):
        try:
            s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            s = str(obj)
        return s if len(s) <= n else s[:n] + "...<truncated>"
        
    def _sanitize_preview(tool_name: str, result: str) -> str:
        if tool_name == "mock_survival":
            try:
                m = re.search(r'(-?\d+(?:\.\d+)?)', str(result))
                if m:
                    return f"probability={m.group(1)}"
                else:
                    return "probability=N/A"
            except Exception:
                return "probability=N/A"
        return str(result)


    messages = [SystemMessage(content=system_prompt)]
    messages.extend(hist.messages)
    messages.append(HumanMessage(content=question + file_context))

    t0 = time.time()
    ai = llm_tools.invoke(messages)
    tool_calls = getattr(ai, "tool_calls", []) or []

    while tool_calls:
        trace["rounds"] += 1
        tool_msgs = []
        for tc in tool_calls:
            name = tc.get("name")
            args = tc.get("args", {})
            call_id = tc.get("id")

            start = time.time()
            try:
                tool_fn = next((t for t in TOOLS if t.name == name), None)
                if not tool_fn:
                    result = f"[unknown tool: {name}]"
                else:
                    result = tool_fn.invoke(args)
                ok = True
            except Exception as e:
                result = f"[tool error] {e}"
                ok = False
            dur = time.time() - start

            # 로그 & 트레이스 기록
            preview = _sanitize_preview(name, result)
            LOGGER.info(f"[AGENT] tool={name} ok={ok} dur={dur:.3f}s args={args} result={_truncate(preview, 200)}")
            trace["tool_calls"].append({
                "tool": name,
                "ok": ok,
                "duration_s": round(dur, 3),
                "args": args,  # 필요하면 args도 마스킹 가능
                "result_preview": _truncate(preview, 500),
            })

            tool_msgs.append(ToolMessage(content=str(result), name=name, tool_call_id=call_id))

        messages.append(ai)
        messages.extend(tool_msgs)
        ai = llm_tools.invoke(messages)
        tool_calls = getattr(ai, "tool_calls", []) or []

    total = time.time() - t0

    # 히스토리 업데이트
    hist.add_user_message(question + file_context)
    hist.add_ai_message(ai.content or "")

    # 마지막 요약 로그
    LOGGER.info(f"[AGENT] answer_len={len(ai.content or '')} rounds={trace['rounds']} total={total:.3f}s")

    if debug:
        trace["answer_preview"] = _truncate(ai.content or "", 800)
        trace["total_duration_s"] = round(total, 3)
        return (ai.content or ""), trace
    else:
        return (ai.content or ""), None


# =============================
# 3) LLM & 체인
# =============================
llm = ChatOpenAI(model="gpt-5", temperature=0)

DEFAULT_SYSTEM = (
    "You are a helpful assistant. Keep answers concise and accurate. "
    "Use tools only when they help answer better (e.g., math calculation, searching uploaded PDFs, getting current time). "
    "If tools are not needed, answer directly."
)

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
    debug: Optional[bool] = False

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

        answer, trace = run_agent(
          body.question, system_prompt, session_id, file_context=file_context, debug=bool(body.debug)
)
        resp = {"answer": answer}
        if trace is not None:
            resp["debug"] = trace
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
<h2>FastAPI Chat (Memory + Upload)</h2>

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
  <button id="btn">Send</button>
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
  if (!question) return;
  println("<b>You:</b> " + question);
  q.value = "";

  try {  // ✅ try 복구
    const r = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        question,
        session_id: sid.value || "default",
        system_prompt: sys.value || null,
        file_id: lastFileId || null,
        debug: true                    // ✅ 툴 콜링 트레이스 요청
      })
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);

    // 답변 출력
    println("<span style='color:#1f4c7c'><b>Assistant:</b> " + (data.answer || "") + "</span>");

    // ✅ 툴 사용 뱃지/요약 표시
    if (data.debug && Array.isArray(data.debug.tool_calls)) {
      const used = data.debug.tool_calls.map(t => t.tool).filter(Boolean);
      const info = used.length ? used.join(", ") : "none";
      const rounds = data.debug.rounds ?? 0;
      const total = data.debug.total_duration_s ?? "?";
      println(
        `<div style="color:#666;font-size:12px;margin:-6px 0 8px 0">
           ↳ tools used: ${info} · rounds: ${rounds} · total: ${total}s
         </div>`
      );

      // (선택) 상세 내역 펼쳐보기
      // println("<pre style='background:#f7f7f7;border:1px dashed #ccc;padding:6px;font-size:12px'>"
      //   + JSON.stringify(data.debug.tool_calls, null, 2) + "</pre>");
    }
  } catch (e) {
    println("<span style='color:#b00020'><b>Error:</b> " + e.message + "</span>");
  }
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
