# Recipes

Recipe JSON files define what tools are required and which rules run during surgery.

## How a recipe is used

- **app.py** loads the recipe once at startup (e.g. `recipes/trauma_room.json`).
- **preop_required**: list of `{ "tool": "name", "min_count": n }`. The Pre-Op checklist shows one row per tool; a tool is OK if it has been seen (smoothed over the last N samples) with count >= min_count. Tool names should match your YOLO model class names when normalized (lowercase, spaces → underscores).
- **params**: `conf_min` (detection confidence), `stable_seconds` (how long all tools must be present before "Start Surgery" is enabled).
- **intraop_rules**: list of rules. Each rule has:
  - **id**: string (e.g. "r1")
  - **phase**: one of incision, suturing, irrigation, closing (must match PHASES in src.constants)
  - **if_present**: tools that must be detected (count >= 1) for the condition to be true
  - **if_missing**: tools that must NOT be present
  - **hold_seconds**: condition must be true for this long before the rule fires an alert
  - **message**: text shown in the alert

Only rules whose **phase** matches the current Intra-Op phase are evaluated. See **src/rules.py** for the exact evaluation logic (debouncing, one-shot per condition).

## trauma_room.json

Example recipe with 8 preop tools and 5 intraop rules. Adjust **preop_required** tool names to match your `models/best.pt` class names (use normalized form: e.g. `needle_holder` or whatever your model returns when lowercased and spaces replaced by underscores).
