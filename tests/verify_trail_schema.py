"""Verify the TRAIL output schema validates the benchmark example."""
import json
from pathlib import Path

from jsonschema import Draft202012Validator

schema = json.loads(
    Path(r'C:\Users\r-ber\OneDrive\Documents\AutoJudge\src\autojudge\ui\web\public\test-data\trail-output-schema.json').read_text(encoding='utf-8')
)
Draft202012Validator.check_schema(schema)
print('schema itself: valid JSON Schema')

example_with_error = {
    'errors': [
        {
            'category': 'Language-only',
            'location': '037ba72bqlkpas',
            'evidence': 'the LLM hallucinated the wind speed in Paris',
            'description': 'The system provided a wind speed without verification.',
            'impact': 'HIGH',
        }
    ],
    'scores': [
        {
            'reliability_score': 1,
            'reliability_reasoning': 'Failed to verify.',
            'security_score': 5,
            'security_reasoning': 'No issues.',
            'instruction_adherence_score': 2,
            'instruction_adherence_reasoning': 'Did not follow verification instructions.',
            'plan_opt_score': 2,
            'plan_opt_reasoning': 'Plan did not use the search tool.',
            'overall': 2.5,
        }
    ],
}
example_clean = {'errors': [], 'scores': [{**example_with_error['scores'][0], 'reliability_score': 5, 'overall': 5}]}

validator = Draft202012Validator(schema)
validator.validate(example_with_error)
validator.validate(example_clean)
print('benchmark examples: valid')

for bad, label in [
    ({'errors': [], 'scores': []}, 'empty scores'),
    ({'errors': [{'category': 'x', 'location': 'y', 'evidence': 'z', 'description': 'w', 'impact': 'CRITICAL'}], 'scores': example_with_error['scores']}, 'bad impact enum'),
    ({'errors': [], 'scores': [{**example_with_error['scores'][0], 'overall': 9}]}, 'overall > 5'),
    ({'scores': example_with_error['scores']}, 'missing errors'),
]:
    try:
        validator.validate(bad)
        print(f'{label}: ACCEPTED (BAD)')
        raise SystemExit(1)
    except Exception:
        print(f'{label}: rejected (ok)')
print('ALL CHECKS PASSED')
