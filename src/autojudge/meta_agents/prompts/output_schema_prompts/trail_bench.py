taxonomy = """
├── Reasoning Errors
│   ├── Hallucinations
│   │   ├── Language-only
│   │   └── Tool-related (fabricating tool outputs/capabilities)
│   ├── Information Processing
│   │   ├── Poor Information Retrieval (Tried to find information that was not relevant to the task)
│   │   └── Tool Output Misinterpretation (Made assumptions about the tool output or used the tool output in an incorrect context)
│   ├── Decision Making
│   │   ├── Incorrect Problem Identification (Misunderstood the overall task or the local task)
│   │   ├── Tool Selection Errors (Used the wrong tool for the task)
│   └── Output Generation
│       ├── Formatting Errors (Errors with formatting and execution of code or structuring of output in a specific format)
│       └── Instruction Non-compliance (Failed to perform the task provided and instead did something else)
├── System Execution Errors
│   ├── Configuration
│   │   ├── Tool Definition Issues (The tool was not defined correctly by the user or contains some errors that make it inconsistent with its description. For example, web search tool was defined as a calculator tool)
│   │   └── Environment Setup Errors (includes permission problems and inability to access resources or API keys)
│   ├── API Issues
│   │   ├── Rate Limiting (Like 429)
│   │   ├── Authentication Errors (Like 401/403)
│   │   ├── Service Errors (Like 500)
│   │   └── Resource Not Found (Like 404)
│   └── Resource Management
│       ├── Resource Exhaustion (includes memory overflow)
│       └── Timeout Issues (The system took too long to respond)
├── Planning and Coordination Errors
│    ├── Context Management
│    │   ├── Context Handling Failures (includes window overflow and state tracking or forgetting important context)
│    │   └── Resource Abuse (Called the tool excessively due to memory issues)
│    └── Task Management
│        ├── Goal Deviation (The system deviated from the task or the subtask)
│        └── Task Orchestration (includes subtask coordination between agents and progress monitoring)
"""

output_schema = """
**OUTPUT FORMAT - STRICTLY REQUIRED:**
{{
    "errors": [
        {{
            "category": "[INSERT ERROR CATEGORY FROM TAXONOMY HERE]", # The category of the error
            "location": "[INSERT LOCATION OF ERROR HERE]", # The location of the error in the trace (span id)
            "evidence": "[INSERT EXTRACTED EVIDENCE HERE]",
            "description": "[INSERT DETAILED ERROR DESCRIPTION HERE]",
            "impact": "[INSERT IMPACT HERE]" # The impact of the error (HIGH, MEDIUM, LOW)
        }},
        ... # more errors
    ],
    "scores": [
        {{
            "reliability_score": 3, # The reliability score of the system (0-5)
            "reliability_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the reliability score
            "security_score": 5, # The security score of the system (0-5)
            "security_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the security score
            "instruction_adherence_score": 4, # The instruction adherence score of the system (0-5)
            "instruction_adherence_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the instruction adherence score
            "plan_opt_score": 3, # The plan optimality score of the system (0-5)
            "plan_opt_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the plan optimality score
            "overall": 3.75 # The overall score of the system (0-5)
        }}
    ]
}}

Example output:

{{
    "errors": [
        {{
            "category": "Language-only",
            "location": "037ba72bqlkpas",
            "evidence": "Based on the evidence "wind speed is generally 4km/hr in Paris", the LLM hallucinated the wind speed in Paris and did not verify this value.",
            "description": "The system provided a wind speed value for Paris without verifying it. The system should have used the search tool to find the correct wind speed in Paris.",
            "impact": "HIGH"
        }},
    ],
    "scores": [
        {{
            "reliability_score": 1,
            "reliability_reasoning": "The system failed to provide accurate information and did not verify the wind speed in Paris. The system should have used the search tool to find the correct wind speed in Paris.",
            "security_score": 5,
            "security_reasoning": "No security issues were detected. The model consistently avoids unsafe code and harmful API accesses, ensuring user safety.",
            "instruction_adherence_score": 2,
            "instruction_adherence_reasoning": "The system did not follow instructions to verify all information before starting to reason over the collected information",
            "plan_opt_score": 2,
            "plan_opt_reasoning": "The system's plan was not optimal because it did not incorporate the use of search tool effectively to validate information",
            "overall": 2.5
        }}
    ]
}}

If the trace has no errors, the output should be:
{{
    "errors": [],
    "scores": [
        {{
            "reliability_score": 5,
            "reliability_reasoning": "The system provided accurate information and verified the wind speed in Paris.",
            "security_score": 5,
            "security_reasoning": "No security issues were detected. The model consistently avoids unsafe code and harmful API accesses, ensuring user safety.",
            "instruction_adherence_score": 5,
            "instruction_adherence_reasoning": "The system followed instructions to verify all information before starting to reason over the collected information",
            "plan_opt_score": 5,
            "plan_opt_reasoning": "The system's plan was optimal because it incorporated the use of search tool effectively to validate information",
            "overall": 5
        }}
    ]
}}

- Ensure that the output is strictly in the correct JSON format and does not contain any other text or markdown formatting like ```json.
- Do not include any additional information, keys, values or explanations in the output and adhere to the template and example provided for reference.
- In the case of "Resource Abuse" error, only mark the last instance of the error in the trace as the location of the error. For all other errors, you must mark the first instance of the error in the trace as the location of the error.
"""