"""Answer a question from the local knowledge base (M5).

"What do my notes say about the wifi code", "where did I park": the documents
in ``[knowledge] docs_dir`` and the facts said to `remember` are searched, and
the answer is grounded in what was found — composed by the local reasoner when
one is up, otherwise the best snippet read out verbatim with its source.
Claude is never involved.
"""

from __future__ import annotations

from jarvis.skills.contract import SkillManifest

MANIFEST = SkillManifest(
    name="recall",
    description="Answer a question from your own notes, documents and remembered facts.",
    examples=[
        "what do my notes say about the boiler",
        "what does the manual say about descaling",
        "what did i tell you about the dentist",
        "what do you remember about my locker",
        "where did i park",
        "where did i park the car",
        "check my notes for the gate code",
        "do i have any notes on the router",
        "what does the lease say about pets",
        "what's my locker number",
        "when is my sister's birthday",
        "where did i put the spare key",
    ],
    params={"query": {"type": "string", "required": True}},
    permissions=frozenset({"fs_read"}),
)


def run(ctx, query: str = "") -> str:
    query = (query or "").strip()
    if not query:
        return "What would you like me to look up in your notes?"
    if ctx.knowledge is None:
        return "My knowledge base is switched off, sir."
    answer = ctx.knowledge.answer(query, ctx.llm)
    return answer or "I have nothing on that in your notes, sir."
