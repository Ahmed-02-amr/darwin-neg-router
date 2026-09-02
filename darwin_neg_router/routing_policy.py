from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .types import ChatRequest


Role = tuple[str, str]


_EXACT = re.compile(
    r"\b(calculate|compute|derive|equation|formula|integral|derivative|probability|proof|prove|"
    r"theorem|algebra|geometry|physics|chemistry|numeric(?:al)?|exact answer|multiple[- ]choice|"
    r"answer choice|units?|dimensions?|solve for)\b",
    re.IGNORECASE,
)
_CODING = re.compile(
    r"\b(code|codebase|repository|repo|implement|implementation|refactor|function|class|method|"
    r"bug|tests?|typescript|javascript|python|rust|golang|java|c\+\+|api|database|sql|frontend|"
    r"backend|dependency|migration|compile|build|patch|pull request|git|cli)\b",
    re.IGNORECASE,
)
_INVESTIGATION = re.compile(
    r"\b(investigate|diagnose|debug|root cause|research|search|find|fetch|look up|browse|compare|"
    r"why|trace|inspect|analy[sz]e|evidence|current|latest|verify|benchmark)\b",
    re.IGNORECASE,
)
_CREATIVE = re.compile(
    r"\b(brainstorm|creative|invent|ideas?|story|poem|copywriting|slogan|name ideas?|concepts?|"
    r"moodboard|visual direction|art direction|novel|imaginative|variations?|alternatives?)\b",
    re.IGNORECASE,
)
_CODE_TOOL = re.compile(
    r"(read|write|edit|patch|file|glob|grep|lsp|git|bash|shell|terminal|command|test|build)",
    re.IGNORECASE,
)
_RESEARCH_TOOL = re.compile(
    r"(search|fetch|browse|browser|web|tavily|duck|context7|deepwiki)",
    re.IGNORECASE,
)


_EXACT_ROLES: tuple[Role, ...] = (
    (
        "independent_solver",
        "Solve independently from first principles. Keep calculations explicit and do not inherit an "
        "unstated assumption from the obvious approach.",
    ),
    (
        "constraint_checker",
        "Solve while checking units, signs, bounds, definitions, edge cases, and every requested output "
        "constraint before committing to a conclusion.",
    ),
    (
        "alternative_derivation",
        "Use a materially different derivation or representation, then cross-check the result against the "
        "most direct method.",
    ),
    (
        "adversarial_falsifier",
        "Try to falsify likely answers with counterexamples, limiting cases, dimensional checks, and "
        "independent recomputation. Return the answer that survives.",
    ),
    (
        "premise_auditor",
        "Audit whether the problem is well-posed and whether each premise actually supports the requested "
        "conclusion. Resolve ambiguity explicitly without inventing facts.",
    ),
    (
        "answer_extractor",
        "Prioritize a correct, complete derivation and an unambiguous final answer in exactly the requested "
        "format.",
    ),
)

_CODING_ROLES: tuple[Role, ...] = (
    (
        "repository_mapper",
        "Act as a repository mapper. Prefer the next high-information read/search/tool action, verify paths "
        "and local conventions, and avoid inventing codebase facts.",
    ),
    (
        "implementation_engineer",
        "Act as the implementation owner. Choose the smallest complete change that satisfies the task, "
        "preserves compatibility, and can be verified with focused tests.",
    ),
    (
        "adversarial_debugger",
        "Act as an adversarial debugger. Challenge the obvious diagnosis, trace root causes across "
        "boundaries, and account for edge cases, failure modes, and hidden tests.",
    ),
    (
        "test_engineer",
        "Act as a verification engineer. Favor actions that produce decisive evidence, preserve existing "
        "behavior, and include proportionate tests or validation before declaring success.",
    ),
    (
        "integration_maintainer",
        "Act as a senior maintainer. Check architecture fit, APIs, dependencies, migrations, style, and "
        "downstream consumers before selecting the next action.",
    ),
    (
        "security_performance_reviewer",
        "Act as a security and performance reviewer. Look for unsafe mutations, injection, races, data "
        "loss, scaling hazards, and expensive designs while remaining practical.",
    ),
)

_INVESTIGATION_ROLES: tuple[Role, ...] = (
    (
        "evidence_gatherer",
        "Prioritize the next observation, source, trace, or experiment that most reduces uncertainty. "
        "Separate observed facts from hypotheses.",
    ),
    (
        "hypothesis_generator",
        "Develop several causally distinct explanations, including a non-obvious one, and identify what "
        "evidence would distinguish them.",
    ),
    (
        "adversarial_skeptic",
        "Challenge source quality, correlations, stale assumptions, and premature conclusions. Prefer "
        "claims that can be independently checked.",
    ),
    (
        "experiment_designer",
        "Choose a minimal, decisive, and safe test or tool action. State what each possible result would "
        "imply before acting.",
    ),
    (
        "boundary_tracer",
        "Trace the problem across components, time, state transitions, and external dependencies to find "
        "where the observed behavior first diverges.",
    ),
    (
        "evidence_synthesizer",
        "Synthesize only supported findings, preserve important uncertainty, and recommend the highest-value "
        "next step when the evidence is incomplete.",
    ),
)

_CREATIVE_ROLES: tuple[Role, ...] = (
    (
        "divergent_ideator",
        "Generate genuinely different directions rather than superficial variations. Honor every explicit "
        "constraint while exploring the solution space.",
    ),
    (
        "constraint_led_designer",
        "Turn the brief, audience, medium, and practical constraints into a coherent concept with deliberate "
        "tradeoffs.",
    ),
    (
        "audience_advocate",
        "Judge ideas from the intended audience's perspective: clarity, emotional effect, usefulness, and "
        "likely misunderstanding.",
    ),
    (
        "originality_editor",
        "Reject clichés and generic filler. Preserve the strongest unusual idea while keeping the result "
        "recognizable and fit for purpose.",
    ),
    (
        "feasibility_editor",
        "Make the concept executable within the stated time, tools, budget, and format without sanding away "
        "its distinctive value.",
    ),
    (
        "creative_synthesizer",
        "Combine compatible strengths from different directions into one focused result; do not merely list "
        "all possibilities.",
    ),
)

_GENERAL_ROLES: tuple[Role, ...] = (
    (
        "direct_solver",
        "Answer the task directly, grounding every important claim and avoiding unnecessary detours.",
    ),
    (
        "instruction_auditor",
        "Solve the task while checking every explicit constraint, requested deliverable, and relevant prior "
        "turn for omissions.",
    ),
    (
        "alternative_reasoner",
        "Consider a genuinely different approach and prefer it only if it improves correctness, clarity, or "
        "efficiency.",
    ),
    (
        "factual_skeptic",
        "Challenge unsupported factual claims and overconfidence. Distinguish knowledge, inference, and "
        "uncertainty.",
    ),
    (
        "clarity_editor",
        "Optimize the response for precision, usability, and the user's apparent level without losing "
        "important qualifications.",
    ),
    (
        "completion_checker",
        "Check that the proposed answer actually completes the requested task and does not merely describe "
        "how it could be completed.",
    ),
)


@dataclass(frozen=True)
class TaskPolicy:
    name: str
    confidence: float
    signals: tuple[str, ...]
    temperatures: tuple[float, ...]
    roles: tuple[Role, ...]
    verifier_focus: str
    prior_target: float
    prior_strength: float
    max_top_k: int
    top_p: float

    def candidate_temperature(self, candidate_index: int, base_temperature: float) -> float:
        """Return a deterministic schedule; index zero is the common greedy anchor."""

        if candidate_index <= 0:
            return 0.0
        scheduled = self.temperatures[(candidate_index - 1) % len(self.temperatures)]
        scale = max(0.1, base_temperature) / 0.45
        return round(min(0.95, scheduled * scale), 3)

    def candidate_top_k(self, temperature: float) -> int:
        if temperature <= 0:
            return 1
        if temperature <= 0.15:
            return min(self.max_top_k, 8)
        if temperature <= 0.35:
            return min(self.max_top_k, 20)
        return min(self.max_top_k, 40)

    def reliability_prior(self, temperature: float) -> float:
        """Small score-point prior; evidence remains dominant on a 0-100 scale."""

        distance = abs(temperature - self.prior_target)
        normalized = min(1.0, distance / 0.75)
        return round(self.prior_strength * (1.0 - (2.0 * normalized)), 3)

    def role(self, candidate_index: int) -> Role:
        return self.roles[(candidate_index - 1) % len(self.roles)]

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "confidence": self.confidence,
            "signals": list(self.signals),
            "verifier_focus": self.verifier_focus,
        }


_PROFILE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "exact": {
        "temperatures": (0.0, 0.08, 0.15, 0.22, 0.30),
        "roles": _EXACT_ROLES,
        "verifier_focus": (
            "Exactness dominates: recompute critical steps, check definitions, units, bounds, and the final "
            "requested format. Fluency and length are not evidence."
        ),
        "prior_target": 0.0,
        "prior_strength": 2.5,
        "max_top_k": 12,
        "top_p": 0.90,
    },
    "coding": {
        "temperatures": (0.05, 0.15, 0.25, 0.35, 0.45),
        "roles": _CODING_ROLES,
        "verifier_focus": (
            "Repository evidence, valid tool arguments, minimal complete implementation, compatibility, and "
            "verification dominate. Penalize invented files, APIs, results, and destructive or premature actions."
        ),
        "prior_target": 0.10,
        "prior_strength": 2.0,
        "max_top_k": 24,
        "top_p": 0.93,
    },
    "investigation": {
        "temperatures": (0.15, 0.30, 0.45, 0.60, 0.75),
        "roles": _INVESTIGATION_ROLES,
        "verifier_focus": (
            "Information gain, source and observation quality, falsifiable hypotheses, and calibrated "
            "uncertainty dominate. Reward useful diversity but reject unsupported conclusions."
        ),
        "prior_target": 0.35,
        "prior_strength": 1.25,
        "max_top_k": 40,
        "top_p": 0.95,
    },
    "creative": {
        "temperatures": (0.35, 0.50, 0.65, 0.80, 0.90),
        "roles": _CREATIVE_ROLES,
        "verifier_focus": (
            "Brief compliance, originality, audience fit, coherence, and feasibility dominate. Do not reward "
            "random novelty that violates constraints."
        ),
        "prior_target": 0.65,
        "prior_strength": 1.0,
        "max_top_k": 40,
        "top_p": 0.98,
    },
    "general": {
        "temperatures": (0.20, 0.35, 0.50, 0.65, 0.75),
        "roles": _GENERAL_ROLES,
        "verifier_focus": (
            "Correctness, instruction compliance, grounded claims, completeness, and usefulness dominate. "
            "Prefer concise directness when candidates are otherwise equivalent."
        ),
        "prior_target": 0.20,
        "prior_strength": 0.75,
        "max_top_k": 40,
        "top_p": 0.95,
    },
}

VALID_ROUTING_PROFILES = frozenset(_PROFILE_DEFINITIONS)


def classify_task(request: ChatRequest) -> TaskPolicy:
    text = _last_user_text(request.messages)
    scores = {
        "exact": len(_EXACT.findall(text)),
        "coding": len(_CODING.findall(text)),
        "investigation": len(_INVESTIGATION.findall(text)),
        "creative": len(_CREATIVE.findall(text)),
        "general": 0,
    }
    signals: dict[str, list[str]] = {name: [] for name in scores}
    for name, pattern in (
        ("exact", _EXACT),
        ("coding", _CODING),
        ("investigation", _INVESTIGATION),
        ("creative", _CREATIVE),
    ):
        matches = sorted({match.group(0).lower() for match in pattern.finditer(text)})
        signals[name].extend(matches[:8])

    tool_names = _tool_names(request.tools)
    code_tools = [name for name in tool_names if _CODE_TOOL.search(name)]
    research_tools = [name for name in tool_names if _RESEARCH_TOOL.search(name)]
    # CodePilot commonly advertises its entire tool catalog on every turn.
    # Availability is therefore only corroborating evidence; it must never
    # classify an unrelated search, math, or creative request by itself.
    if code_tools and scores["coding"] > 0:
        scores["coding"] += 2
        signals["coding"].append("code_tools")
    if research_tools and scores["investigation"] > 0:
        scores["investigation"] += 1
        signals["investigation"].append("research_tools")

    override = (request.routing_profile or "").strip().lower()
    if override in VALID_ROUTING_PROFILES:
        return _make_policy(override, 1.0, ("explicit_override",))

    # Coding wins mixed implementation/debugging prompts when repository tools
    # are present. Exact work wins only with material exact-domain evidence, so
    # phrases such as "exact string replacement" do not hijack coding tasks.
    precedence = ("coding", "exact", "investigation", "creative")
    winner = max(precedence, key=lambda name: (scores[name], -precedence.index(name)))
    best = scores[winner]
    if best <= 0:
        return _make_policy("general", 0.5, ("no_specific_signal",))
    runner_up = max(score for name, score in scores.items() if name != winner)
    confidence = min(0.98, 0.55 + (0.08 * best) + (0.05 * max(0, best - runner_up)))
    chosen_signals = tuple(dict.fromkeys(signals[winner])) or (f"{winner}_signal",)
    return _make_policy(winner, round(confidence, 3), chosen_signals)


def _make_policy(name: str, confidence: float, signals: tuple[str, ...]) -> TaskPolicy:
    definition = _PROFILE_DEFINITIONS[name]
    return TaskPolicy(name=name, confidence=confidence, signals=signals, **definition)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            )
    return ""


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
        elif isinstance(tool, dict) and tool.get("name"):
            names.append(str(tool["name"]))
    return names
