// Matches the taxonomy and output schema example in README.md.
export const exampleTaxonomy = `├── Reasoning Errors
│   ├── Hallucinations
│   │   ├── Language-only
│   │   └── Tool-related
│   ├── Information Processing
│   │   ├── Poor Information Retrieval
│   │   └── Tool Output Misinterpretation
│   ├── Decision Making
│   │   ├── Incorrect Problem Identification
│   │   └── Tool Selection Errors
│   └── Output Generation
│       ├── Formatting Errors
│       └── Instruction Non-compliance
├── System Execution Errors
│   ├── Configuration
│   ├── API Issues
│   └── Resource Management
├── Planning and Coordination Errors
│   ├── Context Management
│   └── Task Management`;

// A format example rather than a strict JSON Schema (e.g. "0-5" and
// "HIGH|MEDIUM|LOW" are not valid JSON values), so it is accepted by the
// runner as a free-form output format instruction.
export const exampleOutputSchema = `{
    "errors": [
        {
            "category": "...",
            "location": "...",
            "evidence": "...",
            "description": "...",
            "impact": "HIGH|MEDIUM|LOW"
        }
    ],
    "scores": [
        {
            "reliability_score": 0-5,
            "security_score": 0-5,
            "instruction_adherence_score": 0-5,
            "plan_opt_score": 0-5,
            "overall": 0-5
        }
    ]
}`;
