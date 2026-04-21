-- ch03_fin schema constraints (supplements _platform/schema/constraints.cypher)
-- Run after n10s init — adds the Resource constraint used by Neosemantics.

CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS
  FOR (r:Resource) REQUIRE r.uri IS UNIQUE;
