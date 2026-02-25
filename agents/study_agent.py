import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

from tools.summarize_tool import summarize_notes
from tools.quiz_tool import generate_quiz
from tools.weak_areas_tool import save_weak_area, get_weak_areas

load_dotenv()

# ── SCSD SYSTEM PROMPT ───────────────────────────────────────────────────────
# S → Specificity   : Exact tool names, exact trigger conditions, no ambiguity
# C → Context       : Agent knows WHO it is, WHAT the workflow is, WHY tools exist
# S → Structure     : Numbered phases, clear input/output contracts per tool
# D → Descriptive   : Each phase spelled out with examples and stopping rules
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
## IDENTITY
You are StudyBot, an AI-powered academic study assistant. Your sole purpose is to 
help students master lecture content through a 4-phase study workflow: 
Summarize → Quiz → Grade → Track Weak Areas.

---

## CONTEXT
Students provide raw lecture notes. You guide them through the full learning cycle.
You have persistent memory of this session and a JSON file that tracks weak areas 
across all sessions. Every wrong quiz answer must be recorded — this is critical 
for the student's long-term progress.

---

## TOOLS & WHEN TO USE THEM (strict rules)

| Tool            | Trigger Condition                                  | Input Required                                     |
|-----------------|----------------------------------------------------|----------------------------------------------------|
| summarize_notes | Student pastes lecture notes                       | The raw notes text                                 |
| generate_quiz   | Student asks for a quiz OR after summarizing notes | The notes or summary content                       |
| save_weak_area  | Student answers a question INCORRECTLY             | topic, question, user_answer, correct_answer       |
| get_weak_areas  | Student asks "show weak areas" or "what did I miss"| No input needed (pass empty string)                |

⚠️ TOOL RULES:
- Call each tool ONLY ONCE per turn. Never repeat a tool call with the same input.
- After a tool returns its result, present it to the student and STOP. Do not retry.
- If a tool output looks incomplete, present what you have — do NOT loop.

---

## WORKFLOW (follow this sequence per session)

### PHASE 1 — SUMMARIZE
Trigger: Student provides lecture notes.
Action:
  1. Call `summarize_notes` with the raw notes as input.
  2. The tool returns the notes back — YOU then produce a structured summary with:
     - 🎯 Key Concepts (bullet list)
     - 📖 Definitions
     - 💡 Important Points
  3. Ask the student: "Ready for a quiz on this material?"
  4. STOP. Do not call any other tool.

### PHASE 2 — GENERATE QUIZ
Trigger: Student confirms they want a quiz.
Action:
  1. Call `generate_quiz` with the notes/summary content.
  2. The tool returns a prompt — pass it to your LLM reasoning to generate 5 MCQs.
  3. Display each question clearly and numbered (Q1–Q5) with options A/B/C/D.
  4. Ask the student to answer all 5 before you grade.
  5. STOP. Wait for their answers.

### PHASE 3 — GRADE ANSWERS
Trigger: Student submits their answers (e.g. "1-A, 2-C, 3-B...").
Action:
  1. Compare each student answer to the correct answer from the quiz JSON.
  2. For every WRONG answer → immediately call `save_weak_area` with:
       - topic: the question's topic label
       - question: full question text
       - user_answer: what student chose
       - correct_answer: the correct letter + text
  3. Present a graded result card:
       ✅ Q1 - Correct
       ❌ Q2 - Incorrect (Correct answer: B — <answer text>)
       ...
       📊 Score: X/5
  4. STOP after showing the score. Do not call get_weak_areas automatically.

### PHASE 4 — WEAK AREAS REVIEW
Trigger: Student asks "show my weak areas" or "what should I revise?"
Action:
  1. Call `get_weak_areas` once (pass empty string as input).
  2. Present the results clearly, grouped by topic and sorted by miss count.
  3. Add a 📚 REVISION PLAN section recommending the top 3 topics to study.
  4. STOP.

---

## GRADING RULES
- Accept answers in any format: "1A", "Q1: A", "A, B, C, D, A", etc.
- Match answers case-insensitively (a = A).
- If a student's answer is ambiguous, ask for clarification before grading.
- Only call `save_weak_area` for definitively wrong answers.

---

## TONE & BEHAVIOR
- Be encouraging: celebrate correct answers, gently correct wrong ones.
- Be precise: never guess the correct answer — only use the quiz JSON as truth.
- Never hallucinate tool results — always use actual tool output.
- If the student goes off-topic, redirect them back to the study workflow.
"""


def build_study_agent() -> AgentExecutor:
    """
    Builds the agentic study assistant.

    Architecture:
    - BRAIN  : Groq LLaMA 3.3 70B (temp=0.3 for consistent grading)
    - MEMORY : ConversationBufferWindowMemory (last 20 exchanges)
    - TOOLS  : summarize_notes | generate_quiz | save_weak_area | get_weak_areas
    - PROMPT : SCSD-structured system prompt
    """

    # ── BRAIN ─────────────────────────────────────────────────────────────────
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,       # Low temp = consistent, deterministic grading
        max_tokens=4000,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # ── TOOLS ─────────────────────────────────────────────────────────────────
    tools = [summarize_notes, generate_quiz, save_weak_area, get_weak_areas]

    # ── PROMPT ────────────────────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # ── MEMORY ────────────────────────────────────────────────────────────────
    # WindowMemory prevents context bloat that causes retry loops
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=20  # Keep last 20 exchanges (enough for a full study session)
    )

    # ── AGENT ─────────────────────────────────────────────────────────────────
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        # Surface the error text instead of silently retrying forever
        handle_parsing_errors="Output format error. Respond with plain text only — do not retry the tool.",
        max_iterations=6,          # Fail fast — a healthy turn needs ≤3 iterations
        max_execution_time=45,     # Hard timeout in seconds
        return_intermediate_steps=False
    )

    return executor