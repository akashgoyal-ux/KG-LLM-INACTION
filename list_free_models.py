"""Query OpenRouter /models to list free models."""
import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('ChaptersFinancial/.env', override=True)

import requests, urllib3, json
urllib3.disable_warnings()

api_key = os.getenv('OPENROUTER_API_KEY')
resp = requests.get(
    'https://openrouter.ai/api/v1/models',
    headers={'Authorization': f'Bearer {api_key}'},
    verify=False, timeout=20
)
resp.raise_for_status()
models = resp.json().get('data', [])

# Show only free models (pricing.prompt == "0")
free = [m for m in models if str(m.get('pricing', {}).get('prompt', '1')) == '0']
print(f"Free models ({len(free)} total):\n")
for m in sorted(free, key=lambda x: x['id'])[:30]:
    ctx = m.get('context_length', 0)
    print(f"  {m['id']:<55s}  ctx={ctx}")
