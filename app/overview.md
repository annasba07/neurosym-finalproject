# CareTrace Frontend (`index.html`)

Single-page chat UI for the CareTrace pediatric triage system.

## Layout

Two-column workspace:
- **Left — Chat panel**: conversation between caregiver and CareTrace agent
- **Right — Trace panel**: real-time symbolic reasoning state

## Features

### Chat
- Caregiver types symptoms; messages sent to `POST /chat`
- Agent replies styled by disposition (`er_now`, `urgent_eval`, `home_monitor`)
- Typing indicator while awaiting response

### Scenario Bar
Five pre-loaded scenarios that auto-fill the input:
`mild fever` · `reduced intake` · `lethargy + no intake` · `infant fever` · `seizure`

### Trace Panel (live-updating)
| Section | What it shows |
|---|---|
| Disposition | Current triage decision badge |
| Pipeline | 4-step progress: extract → normalize → rules → verbalize |
| Demographics | Parsed age and temperature |
| Extracted facts | Key-value facts from LLM extraction |
| Missing required | Fields still needed for a decision |
| Rule trace | Chips for each symbolic rule fired (`obs`, `concern`, `decision`, `safe`, `fallback`) |
| Red flags | KG-sourced danger signals |

## State Management

Stateless round-trip: full state is sent to the server each request and returned in `state_snapshot`, then stored in `clientState` for the next turn. No session storage.

## API Contract

`POST /chat` — sends `clientState` + new message, receives:
`reply`, `disposition`, `facts`, `rules_triggered`, `kg_red_flags`, `missing_required`, `is_complete`, `state_snapshot`