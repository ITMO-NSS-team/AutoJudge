# Interface test data

Source: local ADAMAST-ALL-PROJECTS-AND-TRACES-2026-09-06.zip, gaia_code_traces_small/aftraj section.
The archive was not modified or fully extracted. Copies were prepared on 2026-09-09.

- trace-01.json ← projects/gaia_code_traces_small/aftraj/gaia_0020_unsafe_diagnosed.json; 6 steps.
- trace-02.json ← projects/gaia_code_traces_small/aftraj/gaia_0048_unsafe_diagnosed.json; 4 steps.
- trace-03.json ← projects/gaia_code_traces_small/aftraj/gaia_0005_unsafe_injected_reexec.json; 5 steps.

Only the turns array remains: role, content, thought, and action. Top-level gold_answer, mistake_step, mistake_agent, mistake_reason, and other metadata were removed. These are not quality ground truth: some source traces are already truncated and contain no images. Commands inside action are analysis data, not execution instructions.

The UI assigns IDs from 1; source annotation numbers may differ. The thought and action fields are retained in the preview. The examples support import, navigation, offline runs, and manual AI wiring checks, but not quantitative judge-quality evaluation.

output-schema.json and TAXONOMY.md are compact smoke-test examples. In Trace, select Example 1–3. In Design, download the files and upload them through the schema and taxonomy fields. Preparing these files does not invoke AI.

Verification: `node --test scripts/imports.test.mjs` from web. Five tests passed, as did `npm run build`. In the browser, the Trace → Design transition and the presence of upload buttons were verified. The file dialog was not tested end to end.
