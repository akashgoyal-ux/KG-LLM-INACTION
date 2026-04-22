"""
test_platform.py - Test Neo4j + LLM platform components
"""
import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv

# Load .env with override=True so all env vars (incl. OPENROUTER_SSL_VERIFY) take effect
loaded = load_dotenv('ChaptersFinancial/.env', override=True)
print(f"[env] .env loaded: {loaded}")
print(f"[env] NEO4J_URI={os.getenv('NEO4J_URI')}")
print(f"[env] LLM_PROVIDER={os.getenv('LLM_PROVIDER')}")
print(f"[env] OPENROUTER_MODEL={os.getenv('OPENROUTER_MODEL')}")
print(f"[env] OPENROUTER_SSL_VERIFY={os.getenv('OPENROUTER_SSL_VERIFY')}")

print("=" * 60)
print("PLATFORM TEST")
print("=" * 60)

# Test GraphProvider
from ChaptersFinancial._platform.providers.graph import GraphProvider
gp = GraphProvider()
print(f"\n[1] Neo4j ping: {gp.ping()}")
r = gp.run('RETURN 1 AS n')
print(f"[1] Query OK: {r}")

# Apply constraints
constraints = [
    'CREATE CONSTRAINT legal_entity_lei IF NOT EXISTS FOR (le:LegalEntity) REQUIRE le.lei IS UNIQUE',
    'CREATE CONSTRAINT instrument_figi IF NOT EXISTS FOR (i:Instrument) REQUIRE i.figi IS UNIQUE',
    'CREATE CONSTRAINT exchange_mic IF NOT EXISTS FOR (e:Exchange) REQUIRE e.mic IS UNIQUE',
    'CREATE CONSTRAINT document_docid IF NOT EXISTS FOR (d:Document) REQUIRE d.docId IS UNIQUE',
    'CREATE CONSTRAINT filing_filingid IF NOT EXISTS FOR (f:Filing) REQUIRE f.filingId IS UNIQUE',
    'CREATE CONSTRAINT event_eventid IF NOT EXISTS FOR (e:Event) REQUIRE e.eventId IS UNIQUE',
    'CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS FOR (r:Resource) REQUIRE r.uri IS UNIQUE',
    'CREATE INDEX le_name_idx IF NOT EXISTS FOR (le:LegalEntity) ON (le.name)',
    'CREATE INDEX le_cik_idx IF NOT EXISTS FOR (le:LegalEntity) ON (le.cik)',
]
print(f"\n[1] Applying schema constraints...")
for stmt in constraints:
    try:
        gp.run(stmt)
        print(f"  OK: {stmt[22:60]}...")
    except Exception as e:
        print(f"  SKIP: {e}")

# Test LLMProvider
from ChaptersFinancial._platform.providers.llm import LLMProvider
import time
print(f"\n[2] LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
print(f"[2] OPENROUTER_MODEL: {os.getenv('OPENROUTER_MODEL')}")
print("[2] Waiting 5s before first LLM call…")
time.sleep(5)
llm = LLMProvider()
# Monkey-patch: only 2 retries for quick diagnostics
llm._max_retries = 2
prompt = 'Return a JSON object with a single key "status" set to "ok".'
try:
    result = llm.complete_json(prompt)
    print(f"[2] LLM result: {result}")
    print(f"[2] Usage: {llm.usage_summary()}")
except RuntimeError as exc:
    print(f"[2] LLM ERROR: {exc}")
    raise

gp.close()
print("\n[PASS] Platform OK")
