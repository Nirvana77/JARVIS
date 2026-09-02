import os

import anthropic
from dotenv import load_dotenv

# Default to Anthropic's most capable model. Override with the ANTHROPIC_MODEL
# env var if you want a cheaper/faster one (e.g. "claude-sonnet-5").
DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = (
	"You are JARVIS, a spoken voice assistant. Answer in a brief, natural, "
	"conversational tone that sounds right when read aloud. Keep replies to a "
	"few sentences. Do not use markdown, bullet lists, or code blocks unless "
	"the user explicitly asks for them."
)

client = None
model = DEFAULT_MODEL


def ask_claude(prompt):
	"""Send a single free-form question to Claude and return the spoken answer."""
	response = client.messages.create(
		model=model,
		max_tokens=1024,
		thinking={"type": "adaptive"},
		system=SYSTEM_PROMPT,
		messages=[{"role": "user", "content": prompt}],
	)
	return "".join(
		block.text for block in response.content if block.type == "text"
	).strip()


def init():
	global client, model
	load_dotenv()
	model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
	# The SDK reads ANTHROPIC_API_KEY from the environment on its own; fall back
	# to the legacy `api_key` name from older .env files if that's all there is.
	api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("api_key")
	# Identity-linked API keys must name the workspace they act in.
	workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")
	headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
	kwargs = {"default_headers": headers} if headers else {}
	if api_key:
		kwargs["api_key"] = api_key
	client = anthropic.Anthropic(**kwargs)
