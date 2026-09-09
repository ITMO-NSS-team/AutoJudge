// Derived from example.md criteria, without its trace-specific verdict.
export const exampleTaxonomy = `# MAS evaluation

## Search integrity
- Factuality: distinguish real source data from synthetic or placeholder data.
- Constraint satisfaction: check the user's location, time, quantity and source constraints.
- Task completion: determine whether the requested answer was delivered, rather than only a demonstration or function test.

## Execution strategy
- Error recovery: assess whether recovery addresses the original task after an environment or tool error.
- Logic soundness: identify unjustified substitutions of the task or its data.
- Tool usage: assess whether tools and sources appropriate to the task were used and their results checked.

## Specialist scoring
- ideal: the requested task is completed using supported data and appropriate tools.
- fair: a meaningful attempt is made but minor constraints are missed or limitations prevent full completion.
- poor: the original task is bypassed, unsupported data replaces required evidence, or no substantive answer is delivered.
Specialists should provide a score, justification, exact agent names and step IDs as evidence.

## Final attribution
Identify the primary failure decision and responsible agent based on trace evidence and specialist findings.
Do not assume that an environment error is itself an agent error. Do not copy a verdict from an example.
Return exactly agent, step and reason according to the output schema. step is a positive integer encoded as a string.
This template follows a failure-attribution example; use it for traces with an identifiable failure. Adapt the schema for successful or inconclusive cases.`;

export const exampleOutputSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['agent', 'step', 'reason'],
  properties: {
    agent: { type: 'string', minLength: 1, description: 'Exact responsible agent name from the evaluated trace.' },
    step: { type: 'string', pattern: '^[1-9][0-9]*$', description: 'Trace step ID encoded as a positive integer string.' },
    reason: { type: 'string', minLength: 1, description: 'Evidence-based explanation of the primary failure decision.' },
  },
};
