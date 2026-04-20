                    ┌──────────────────────────────┐
                    │        HUMAN ENGINEERS       │
                    │  - Initial codebase          │
                    │  - Goals + constraints       │
                    │  - Validation framework      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        SEED IMPROVER         │
                    │ (Initial AGI / LLM Agent)    │
                    │------------------------------│
                    │ • Goal: "Improve yourself"   │
                    │ • Planning + reasoning       │
                    │ • Code generation            │
                    │ • Tool usage                 │
                    └──────────────┬───────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
                 ▼                 ▼                 ▼

        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │ SELF-PROMPT  │   │ CODE ENGINE  │   │ TEST SYSTEM  │
        │ LOOP         │   │              │   │              │
        │--------------│   │--------------│   │--------------│
        │ - Generate   │   │ - Read code  │   │ - Run tests  │
        │   ideas      │   │ - Modify     │   │ - Validate   │
        │ - Plan tasks │   │ - Compile    │   │ - Detect     │
        │ - Iterate    │   │ - Execute    │   │   regressions│
        └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
               │                  │                  │
               └──────────┬───────┴───────┬──────────┘
                          ▼               ▼

                ┌────────────────────────────────────┐
                │     EVALUATION & VALIDATION LOOP   │
                │------------------------------------│
                │ • Performance metrics              │
                │ • Goal alignment                   │
                │ • Safety constraints               │
                │ • Non-regression checks            │
                └──────────────┬─────────────────────┘
                               │
                     PASS      │       FAIL
                ───────────────┼──────────────────
                               │
                               ▼
                ┌──────────────────────────────┐
                │   UPDATED AGENT VERSION (N+1)│
                │------------------------------│
                │ • Improved capabilities      │
                │ • Better reasoning           │
                │ • New tools / subsystems     │
                └──────────────┬───────────────┘
                               │
                               ▼
                      (RECURSIVE LOOP)
                               │
                               ▼

        ┌────────────────────────────────────────────┐
        │     INTELLIGENCE GROWTH (FEEDBACK LOOP)    │
        │--------------------------------------------│
        │ Capability ↑ → Better improvements ↑       │
        │                → Faster iterations ↑       │
        │                → More capability ↑         │
        └────────────────────────────────────────────┘
