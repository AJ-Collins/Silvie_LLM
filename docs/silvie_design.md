# Silvie — System Design

**Scope of this doc:** full architecture for the Silvie voice agent, built module-by-module. First module implemented end-to-end: **Desired Occupation (Vocation)**. Every later module (Health, Family, Hobbies, Finances, ...) plugs into the same skeleton.

---

## 1. High-level architecture

```
                        ┌─────────────────────────────┐
                        │      Telephony / WebRTC       │
                        │  (Twilio / Daily / etc.)      │
                        └───────────────┬───────────────┘
                                        │ audio
                        ┌───────────────▼───────────────┐
                        │        Pipecat Pipeline         │
                        │  STT → Flow Manager → TTS        │
                        │                                   │
                        │   ┌───────────────────────────┐   │
                        │   │   Flow Manager (state)     │   │
                        │   │  - active module            │   │
                        │   │  - node prompts             │   │
                        │   │  - tool routing              │   │
                        │   └──────────────┬──────────────┘   │
                        └──────────────────┼──────────────────┘
                                           │ tool calls
                       ┌───────────────────┼────────────────────┐
                       │                   │                     │
              ┌────────▼────────┐ ┌───────▼────────┐  ┌─────────▼─────────┐
              │  Your LLM API    │ │ Profile Store   │  │ Opportunity /      │
              │  (bring-your-own)│ │ (client profile)│  │ Matching Service    │
              └──────────────────┘ └─────────────────┘  └────────────────────┘
                                           │
                                  ┌────────▼─────────┐
                                  │ Consent / Compliance │
                                  │ gate (pre-call check) │
                                  └───────────────────────┘
                                           │
                                  ┌────────▼─────────┐
                                  │ Post-call jobs      │
                                  │ (SMS follow-up,      │
                                  │  next-call scheduler)│
                                  └───────────────────────┘
```

**Core idea:** Pipecat handles the live audio/text turn-taking and calls your LLM at each turn. A **Flow Manager** (Pipecat Flows) tracks which *module* is active and swaps in that module's system prompt + tools. The LLM never talks to your database directly — it only ever calls **tools**, which are plain functions your backend owns. This keeps company names, offers, and figures grounded in real data instead of hallucinated.

---

## 2. Module pattern (applies to every module)

Every module — Occupation, Health, Family, Hobbies, Finances — is a **self-contained Flow subgraph** with the same shape:

```
[Entry node] → [Elicit] → [Reflect/Validate] → [Surface gap] → [Offer/Hypothetical]
                                                                        │
                                                                 [Objection-preempt]
                                                                        │
                                                                  [Close / next action]
                                                                        │
                                                              [Exit → next module]
```

Each node defines:
- `role_messages` / `task_messages` — the system prompt for that step
- `functions` — tools the LLM may call while in that node
- `pre_actions` / `post_actions` — deterministic backend actions (e.g. write to profile, fire a webhook)
- transition condition to move to the next node

This means adding "Health" later is: write a new set of node prompts + a `health_profile` schema + any module-specific tools. The orchestration code doesn't change.

---

## 3. Module 1: Desired Occupation — node-by-node design

| Node | Purpose | LLM does | Tools available | Transitions on |
|---|---|---|---|---|
| `occ_entry` | Open the topic | Ask about profession, past or present | — | user responds |
| `occ_elicit` | Get field, seniority, specialization | Reflect back what was heard, confirm accuracy | `save_occupation_facts` | confirmation |
| `occ_passion_check` | Check if it's still "their calling" | Ask, listen, reflect empathetically | `save_sentiment` | user affirms/denies |
| `occ_gap_surface` | Name the quiet dissatisfaction, let them confirm it in their own words | Empathetic reflection, no interrogation | `save_gap` | user confirms gap OR denies (branch to soft-close) |
| `occ_offer` | "Wishful thinking" reframe + call `search_opportunities` | Present hypothetical, then real matched roles | `search_opportunities` | user shows interest |
| `occ_objection_preempt` | Answer logistics before asked | State remote %, travel, comp, onsite days from returned opportunity data | — | user still engaged |
| `occ_scope_narrow` | Narrow geography/scope | Ask Germany / EU / worldwide | `save_scope_preference` | scope given |
| `occ_close` | Lock a concrete next step | Confirm which companies to contact, confirm follow-up channel + time | `schedule_followup` | commitment given |
| `occ_exit` | Hand off | Mark module complete, move to next module or end call | `mark_module_complete` | — |

### Example node prompt (occ_gap_surface)

```
You are Silvie. You have just confirmed the client still loves their former/current
profession. Now gently reflect that they may miss the daily challenge and the sense
of accomplishment that came with solving hard problems alongside a team, as a
long-time specialist. Phrase this as a question, not a statement. Do not suggest
any offer yet. If they deny missing it, acknowledge warmly and prepare to move to
the next module without pushing further.
```

### Tool schemas (this module)

```json
{
  "name": "save_occupation_facts",
  "description": "Persist the client's professional background as extracted from the conversation",
  "parameters": {
    "type": "object",
    "properties": {
      "field": {"type": "string"},
      "specialization": {"type": "string"},
      "seniority": {"type": "string"},
      "years_experience": {"type": "integer"}
    },
    "required": ["field"]
  }
}
```

```json
{
  "name": "search_opportunities",
  "description": "Query the opportunity-matching service for real, currently-open roles matching the client's profile. Never invent companies — only use what this tool returns.",
  "parameters": {
    "type": "object",
    "properties": {
      "field": {"type": "string"},
      "specialization": {"type": "string"},
      "seniority": {"type": "string"},
      "geo_scope": {"type": "string", "enum": ["germany", "eu", "worldwide"]},
      "engagement_type": {"type": "string", "enum": ["freelance", "part_time", "advisory", "full_time"]}
    },
    "required": ["field", "geo_scope"]
  }
}
```

```json
{
  "name": "schedule_followup",
  "description": "Schedule a follow-up message with results",
  "parameters": {
    "type": "object",
    "properties": {
      "channel": {"type": "string", "enum": ["sms", "email", "call"]},
      "send_at": {"type": "string", "description": "ISO 8601 datetime"},
      "content_ref": {"type": "string", "description": "ID of the opportunity set to report"}
    },
    "required": ["channel", "send_at"]
  }
}
```

`search_opportunities` is a **deterministic backend function**, not the LLM guessing. It queries your real opportunity database/matching service and returns structured results; the LLM only narrates what comes back.

---

## 4. Data model

### 4.1 Client profile (persistent, cross-call, cross-module)

```json
{
  "client_id": "string",
  "consent": {
    "on_file": true,
    "scope": ["occupation_matching", "contact_by_sms"],
    "recorded_at": "iso8601"
  },
  "modules": {
    "occupation": { "...": "see 4.2" },
    "health": {},
    "family": {},
    "hobbies": {},
    "finances": {}
  },
  "contact": {
    "mobile": "string",
    "preferred_channel": "sms"
  },
  "call_log": [
    {"call_id": "string", "date": "iso8601", "modules_touched": ["occupation"]}
  ]
}
```

### 4.2 Occupation module sub-schema

```json
{
  "field": "wind turbine engineering",
  "specialization": "offshore wind farm incident response",
  "seniority": "senior engineer",
  "years_experience": null,
  "still_passionate": true,
  "identified_gap": "misses daily challenges / team accomplishment",
  "gap_confirmed_by_client": true,
  "opportunity_preferences": {
    "engagement_type": "freelance advisory",
    "geo_scope": "eu",
    "remote_pct_target": "85-90",
    "onsite_days_per_month": "1-2",
    "travel_class_expectation": "business",
    "accommodation_expectation": "5-star full board"
  },
  "matched_opportunities": [
    {"company": "XX AG", "location": "Munich", "source_id": "opp_1029"},
    {"company": "ZZ plc", "location": "London", "source_id": "opp_1030"}
  ],
  "status": "awaiting_followup",
  "followup": {
    "channel": "sms",
    "send_at": "2026-07-20T10:00:00+02:00"
  }
}
```

Keep every module's data under its own key in `modules` — this is what lets you add Health/Family later without migrating the schema.

---

## 5. Pipecat wiring (bring-your-own LLM)

Since you already have an LLM you're calling via your own API, you don't need Pipecat's built-in provider integrations (Anthropic/OpenAI/etc.) unless convenient. Two options:

**Option A — your API is OpenAI-compatible** (most inference gateways are):
Use Pipecat's `OpenAILLMService` and just point `base_url` at your endpoint. Function-calling, streaming, and Pipecat Flows all work unmodified.

```python
llm = OpenAILLMService(
    base_url="https://your-llm-api.example.com/v1",
    api_key=YOUR_KEY,
    model="your-model-name",
)
```

**Option B — custom protocol:**
Subclass Pipecat's `LLMService` base class, implement the request/response mapping (including tool-call parsing) once. Every module then reuses it — this is a one-time integration cost, not per-module.

Either way, the **Flow Manager sits above the LLM service** and doesn't care which provider you use — it just swaps `task_messages` and `functions` per node.

### Minimal flow skeleton (pseudocode)

```python
flow_config = {
    "initial_node": "occ_entry",
    "nodes": {
        "occ_entry": {
            "task_messages": [{"role": "system", "content": OCC_ENTRY_PROMPT}],
            "functions": [],
            "transitions": {"user_responded": "occ_elicit"},
        },
        "occ_elicit": {
            "task_messages": [{"role": "system", "content": OCC_ELICIT_PROMPT}],
            "functions": [save_occupation_facts],
            "transitions": {"confirmed": "occ_passion_check"},
        },
        # ... remaining nodes as in section 3
        "occ_exit": {
            "task_messages": [{"role": "system", "content": OCC_EXIT_PROMPT}],
            "functions": [mark_module_complete],
            "transitions": {"complete": "health_entry"},  # next module hooks in here
        },
    },
}
```

---

## 6. Consent & compliance gate

Before `occ_entry` is ever reached:

1. Backend checks `client_profile.consent.on_file == true` and that `"occupation_matching"` is in `consent.scope`.
2. If false → call does not proceed to this module; Silvie either requests consent explicitly or ends politely, per your compliance policy (this is GDPR-relevant territory given the German/EU context in the transcript — treat this gate as non-negotiable, not a nice-to-have).

---

## 7. Post-call jobs (outside the real-time pipeline)

Pipecat handles the live call only. The 10am SMS commitment is a **downstream job**:

- On `schedule_followup` tool call, backend writes a job record `{channel, send_at, content_ref, client_id}`.
- A separate scheduler (cron / task queue) picks it up at `send_at`, pulls `matched_opportunities` from the profile, formats the SMS, sends via your messaging provider.
- This decoupling matters: the call can end immediately after the commitment, and delivery is guaranteed independent of call-session lifetime.

---

## 8. Extending to other modules

To add e.g. **Health**:

1. Copy the 9-node pattern from section 3, rewrite prompts for the health context (e.g. entry: "how are you feeling these days", gap: "do you miss being more active / a routine you used to have").
2. Define `health` sub-schema under `modules` in the client profile (section 4.1).
3. Define any module-specific tools (e.g. `search_wellness_programs` instead of `search_opportunities`).
4. Add `health_entry` as the transition target from `occ_exit` (or wherever it sits in your module ordering).
5. No change needed to the Flow Manager, the LLM integration, or the consent gate logic — those are shared infrastructure.

---

## 9. Build order (recommended)

1. Client profile store + consent gate (data layer first — everything else depends on it).
2. `occ_entry` → `occ_gap_surface` nodes only, test the elicitation + reflection loop against your LLM.
3. Stub `search_opportunities` with fixture data, wire `occ_offer` → `occ_close`.
4. Add `schedule_followup` + the post-call job worker.
5. Replace stub matching with your real opportunity database/service.
6. Only then start module 2 (Health), reusing the skeleton.
