# PhD-Skills for Opencode

This document describes how to use the phd-skills plugin that has been installed for opencode.

## Installation Location

`C:\Users\PLN\.local\share\opencode\skills\phd-skills\`

## How to Use

### Auto-Trigger Skills (Natural Language)

Simply describe what you need in natural language, and opencode will automatically load the appropriate skill:

| Say this... | Skill activates |
|-------------|-----------------|
| "Reproduce this arxiv paper" | **Reproduce** — 7-stage paper reproduction |
| "Why is my loss diverging?" | **Debug** — Evidence-first debugging |
| "Compare run A to baseline" | **Compare** — Same-epoch comparison |
| "Launch a new training run" | **Launch** — Pre-flight checklist |
| "Design an ablation study" | **Experiment Design** |
| "Find related papers on X" | **Literature Research** |
| "Check if my numbers match" | **Paper Verification** |
| "Review my methods section" | **Paper Writing** |
| "Analyze dataset bias" | **Dataset Curation** |
| "Prepare code for release" | **Research Publishing** |
| "What will reviewers ask?" | **Reviewer Defense** |
| "Setup LaTeX for CVPR" | **LaTeX Setup** |

### Direct Commands

If opencode supports structured commands, you can reference:

- **xray** — Audit paper against code and data (5 dimensions)
- **factcheck** — Verify BibTeX entries against DBLP
- **gaps** — Literature gap analysis
- **fortify** — Select ablations + anticipate reviewers
- **setup** — Auto-detection tour + optional extras
- **help** — Show all features

## Skill Structure

```
phd-skills/
├── SKILL.md              # Main overview
├── skill.json            # Metadata
├── skills/               # 12 auto-trigger skills
├── commands/             # 6 slash-commands
├── agents/               # 2 specialized agents
├── references/           # Detailed reference guides
│   └── reproduce/        # 7-stage reproduction methodology
└── scripts/              # 11 guardrail scripts
```

## Guardrails (Auto-Running)

The following guardrails run automatically when triggered:

- **Citation Guard** — Verifies citations when editing .tex/.bib
- **Destructive Path Guard** — Warns before rm/mv on absolute paths
- **Jargon Scrub** — Blocks internal jargon in commits/docs
- **LaTeX Check** — Auto-compiles .tex after edits
- **Launch Inject** — Pre-flight checklist before training jobs
- **Notify** — Routes notifications to ntfy/Slack/email
- **Outbound Reminder** — Warns before sending unverified content
- **Remote Edit Guard** — Blocks in-place SSH edits
- **Save State** — Saves research context before compaction
- **Timezone Scrub** — Warns on mismatched timezone tokens
- **Visual Check** — Reminds to inspect generated plots

## Notes

- All skills are zero-dependency and work over SSH
- Skills auto-detect your setup (timezone, trackers, jargon)
- Every skill builds in verification checkpoints
- Designed for research integrity and mistake prevention
