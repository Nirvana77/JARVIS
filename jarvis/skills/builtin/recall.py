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
        "what do my notes say about the wifi code",
        "what does the manual say about descaling",
        "what did i tell you about the meeting",
        "what do you remember about my locker",
        "where did i park",
        "check my notes for the door code",
        "do i have any notes on the router",
        "what did i say about the dentist",
        "what does the lease say about pets",
        "look in my notes for the recipe",
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
