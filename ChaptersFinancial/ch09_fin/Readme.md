# ch09_fin — Financial NED with Ontology Linking

## Objective
Link textual mentions to canonical financial entities/instruments with
calibrated confidence and ontology classes using FIBO/CFI.

## Pipeline
1. **Candidate generation**: alias dictionary (GLEIF + ticker tables) + dense retrieval
2. **Contextual ranking**: BM25 + cross-encoder reranker
3. **Ontology linking**: FIBO/CFI class assignment via Neosemantics

## Definition of Done
- accuracy@1 ≥ 0.85 on dev set
- Calibration ECE ≤ 0.05
