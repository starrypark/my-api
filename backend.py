from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, AsyncGenerator
import os

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# 키 로드 (예: api_key.txt)
# -----------------------------
with open("api_key.txt") as f:
    OPENAI_API_KEY = f.read().strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# LLM & 체인 준비
# -----------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)

DEFAULT_SYSTEM = "You are a helpful assistant. Keep answers concise and accurate."

prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{question}"),
])

# 파서 붙여서 문자열로 스트리밍 가능하게
parsed_chain = prompt | llm | StrOutputParser()

# 세션별 메모리 저장 (데모: 인메모리)
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


# -----------------------------
# 스키마
# -----------------------------
class ChatIn(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    system_prompt: Optional[str] = None


# -----------------------------
# 헬스체크/에코
# -----------------------------
@app.get("/ping")
def ping():
    return {"ok": True}


@app.post("/echo")
def echo(body: ChatIn):
    return {"answer": f"echo: {body.question}"}


# -----------------------------
# 1) 비스트리밍 대화 (메모리 사용)
# -----------------------------
@app.post("/chat")
def chat(body: ChatIn):
    if os.environ.get("OPENAI_API_KEY", "") == "":
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY가 설정되어 있지 않습니다. "
                   "PowerShell: $env:OPENAI_API_KEY=\"...\""
        )
    try:
        session_id = body.session_id or "default"
        system_prompt = body.system_prompt or DEFAULT_SYSTEM

        # 한번에 결과 text
        text = chain_with_memory.invoke(
            {"question": body.question, "system_prompt": system_prompt},
            config={"configurable": {"session_id": session_id}},
        )
        return {"answer": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# 2) 스트리밍 대화 (토큰 단위, 메모리 사용)
# -----------------------------
@app.post("/chat_stream")
async def chat_stream(body: ChatIn):
    """
    fetch 스트림으로 받기 쉬운 'chunked text' 스트리밍.
    Content-Type은 text/plain(기본)으로 흘려보냅니다.
    """
    if os.environ.get("OPENAI_API_KEY", "") == "":
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY가 설정되어 있지 않습니다. "
                   "PowerShell: $env:OPENAI_API_KEY=\"...\""
        )

    session_id = body.session_id or "default"
    system_prompt = body.system_prompt or DEFAULT_SYSTEM

    async def token_generator() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in chain_with_memory.astream(
                {"question": body.question, "system_prompt": system_prompt},
                config={"configurable": {"session_id": session_id}},
            ):
                # chunk는 문자열 파편. 줄바꿈을 넣어 브라우저 쪽에서 flush가 잘 되게 함.
                yield (chunk).encode("utf-8")

        except Exception as e:
            # 에러도 스트림으로 전달
            yield f"\n[STREAM ERROR] {str(e)}".encode("utf-8")

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")


# -----------------------------
# 3) 메모리 리셋
# -----------------------------
@app.post("/reset_memory")
def reset_memory(body: ChatIn):
    sid = body.session_id or "default"
    if sid in _SESSION_STORES:
        del _SESSION_STORES[sid]
    return {"ok": True, "session_id": sid, "message": "memory cleared"}


# -----------------------------
# 테스트용 홈 UI (스트림/비스트림 둘 다)
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>FastAPI Chat</title>
</head>
<body style="font-family:sans-serif; max-width:720px; margin:40px auto;">
<h2>FastAPI Chat (Memory + Stream)</h2>

<div style="margin:8px 0">
  <input id="sid" placeholder="session_id (e.g., user1)" style="width:49%">
  <input id="sys" placeholder="system_prompt (optional)" style="width:49%">
</div>

<div id="log" style="border:1px solid #ddd; padding:10px; height:320px; overflow:auto;"></div>
<textarea id="q" rows="3" style="width:100%; margin-top:10px;" placeholder="질문"></textarea>

<div style="margin-top:8px; display:flex; gap:8px;">
  <button id="btn">Send (non-stream)</button>
  <button id="btnStream">Send (stream)</button>
  <button id="reset">Reset Memory</button>
</div>

<script>
const log = document.getElementById('log');
const q = document.getElementById('q');
const sid = document.getElementById('sid');
const sys = document.getElementById('sys');

function println(html) {
  log.innerHTML += "<div>" + html + "</div>";
  log.scrollTop = log.scrollHeight;
}

// 1) 비스트리밍 호출
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
        system_prompt: sys.value || null
      })
    });
    const data = await r.json();
    if(!r.ok) throw new Error(data.detail || r.statusText);
    println("<span style='color:#1f4c7c'><b>Assistant:</b> " + (data.answer || "") + "</span>");
  } catch (e) {
    println("<span style='color:#b00020'><b>Error:</b> " + e.message + "</span>");
  }
};

// 2) 스트리밍 호출 (ReadableStream 사용)
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
      system_prompt: sys.value || null
    })
  });

  if(!r.ok || !r.body) {
    const data = await r.json().catch(()=> ({}));
    println("<span style='color:#b00020'><b>Error:</b> " + (data.detail || r.statusText) + "</span>");
    return;
  }

  // 스트림 읽기
  const reader = r.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let assistantLine = "<span style='color:#1f4c7c'><b>Assistant:</b> ";

  // 화면에 누적 바인딩
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

// 3) 메모리 리셋
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
</script>
</body>
</html>
"""
