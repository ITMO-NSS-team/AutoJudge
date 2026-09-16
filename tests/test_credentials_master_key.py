"""Cross-platform tests for encrypted credential storage."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1] / 'src' / 'autojudge' / 'ui' / 'backend'
sys.path.insert(0, str(BACKEND))
import credentials


class MasterKeyCredentialTests(unittest.TestCase):
    def test_master_key_round_trip_and_wrong_key_rejection(self):
        secret = 'test-provider-key-value-000000'
        with patch.object(credentials.sys, 'platform', 'linux'), patch.dict(
            os.environ, {'AUTOJUDGE_CREDENTIALS_KEY': 'test-master-secret-123'}, clear=False
        ):
            self.assertEqual(credentials.storage_info()['name'], 'Encrypted container storage')
            payload = credentials.store('OPENROUTER_API_KEY', secret)
            self.assertEqual(payload['storage'], 'master-key')
            self.assertNotIn(secret, json.dumps(payload))
            self.assertEqual(credentials.load('OPENROUTER_API_KEY', payload), secret)

            with patch.dict(
                os.environ, {'AUTOJUDGE_CREDENTIALS_KEY': 'wrong-key'}, clear=False
            ):
                with self.assertRaises(RuntimeError):
                    credentials.load('OPENROUTER_API_KEY', payload)

    def test_missing_master_key_and_keyring_refuses_storage(self):
        with patch.object(credentials.sys, 'platform', 'linux'), patch.dict(
            os.environ, {}, clear=True
        ), patch.object(
            credentials, 'storage_info',
            return_value={'name': 'Environment only', 'available': False, 'persistent': False},
        ):
            with self.assertRaises(RuntimeError):
                credentials.store('OPENROUTER_API_KEY', 'test-provider-key-value-000000')


if __name__ == '__main__':
    unittest.main()
