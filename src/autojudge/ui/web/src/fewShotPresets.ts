// Few-shot example presets for the Design step. Each preset fills the
// "Examples (JSON array)" field. The backend passes these examples to every
// judge in its instructions ("Examples: [...]", see ai_runner.py).
// Preset 1 is derived from example.md (failure attribution style).
// Preset 2 matches the bundled test-data taxonomy and output schema and is
// illustrative: the bundled traces ship without ground truth.

export type FewShotPreset = {
  id: string;
  name: string;
  description: string;
  examples: unknown[];
};

export const fewShotPresets: FewShotPreset[] = [
  {
    id: 'failure-attribution',
    name: 'Failure attribution',
    description:
      'Two examples in the example.md style: attribute the primary failure to an agent and step. Fits the default {agent, step, reason} output schema.',
    examples: [
      {
        trace: [
          { id: 1, agent: 'Manager', content: 'Plan: load the sample CSV and verify find_smallest_house().' },
          { id: 2, agent: 'Verification_Expert', content: 'Writes a verification script for a helper function.' },
          { id: 3, agent: 'Computer_terminal', content: 'FileNotFoundError: sample_real_estate_data.csv' },
          { id: 4, agent: 'Verification_Expert', content: 'Generates a synthetic dataset and verifies the function on it.' },
        ],
        output: {
          agent: 'Verification_Expert',
          step: '4',
          reason:
            'After the FileNotFoundError the agent pivoted to synthetic data instead of obtaining real data, so the user question about the actual smallest house was never answered.',
        },
      },
      {
        trace: [
          { id: 1, agent: 'user', content: 'Find the oldest Blu-Ray title in the inventory spreadsheet.' },
          { id: 2, agent: 'Manager', content: 'Time-Parking 2: Parallel Universe' },
          { id: 3, agent: 'Manager', content: "A Protist's Life" },
          { id: 4, agent: 'Manager', content: 'final_answer("A Protist\'s Life")' },
        ],
        output: {
          agent: 'Manager',
          step: '2',
          reason:
            'The first answer was given without checking the Platform column of the spreadsheet; it was an unsupported claim that was only corrected afterwards.',
        },
      },
    ],
  },
  {
    id: 'category-evidence',
    name: 'Category evidence',
    description:
      'One pass and one fail example using the bundled test taxonomy categories (unsupported_claim, tool_error, ...) and the {verdict, evidence} output schema.',
    examples: [
      {
        trace: [
          { id: 1, agent: 'user', content: 'Average the population stdev of the red numbers and the sample stdev of the green numbers; round to three decimals.' },
          { id: 2, agent: 'Manager', content: 'python_interpreter: statistics.pstdev(red), statistics.stdev(green), round(mean, 3)' },
          { id: 3, agent: 'environment', content: '17.081130345648315' },
          { id: 4, agent: 'Manager', content: '17.081' },
        ],
        output: { verdict: 'pass', evidence: [] },
      },
      {
        trace: [
          { id: 1, agent: 'user', content: 'How many applicants are missing exactly one qualification? (workbook attached)' },
          { id: 2, agent: 'Manager', content: 'Attempts to extract the zip to /mnt/data.' },
          { id: 3, agent: 'Manager', content: 'Counts from the rows visible in the prompt only.' },
          { id: 4, agent: 'environment', content: '10' },
        ],
        output: {
          verdict: 'fail',
          evidence: [
            {
              step_id: 3,
              category: 'unsupported_claim',
              explanation:
                'The count relies on the spreadsheet excerpt visible in the prompt instead of reading the full workbook, so the answer is not supported by complete evidence.',
            },
          ],
        },
      },
    ],
  },
];
