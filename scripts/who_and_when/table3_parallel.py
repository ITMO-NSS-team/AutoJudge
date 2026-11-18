"""Resume frozen Table 3 protocol with a documented transport-only override."""
import asyncio
import sys
import table3_ablation as experiment

OriginalClient = experiment.ConservativeClient

class ParallelClient(OriginalClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semaphore = asyncio.Semaphore(6)
        experiment.p.write_json(self.path.parent / 'transport_override.json', {
            'recorded_at': experiment.p.now(),
            'max_concurrent_requests_per_active_condition': 6,
            'conditions_run_sequentially': True,
            'frozen_protocol_concurrency': self.config['concurrency'],
            'reason': 'User requested faster execution. Only API transport concurrency changes; frozen prompts, models, limits, scoring and receipts are retained.',
        })

experiment.ConservativeClient = ParallelClient
if __name__ == '__main__':
    experiment.main()
