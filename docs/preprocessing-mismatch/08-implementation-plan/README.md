# Implementation Plan

## Overview

This section provides a concrete roadmap for implementing Option B (Packet Embeddings).

---

## Documents in This Section

| Document | Description |
|----------|-------------|
| [files-to-modify.md](./files-to-modify.md) | Existing files that need changes |
| [new-files-needed.md](./new-files-needed.md) | New modules to create |
| [migration-checklist.md](./migration-checklist.md) | Step-by-step execution plan |

---

## Implementation Phases

### Phase 1: Data Extraction (Week 1-2)

| Task | Effort | Priority |
|------|--------|----------|
| Parse JSON files | High | Critical |
| Extract packet directions | Medium | Critical |
| Match JSON to CSV | High | Critical |
| Save to Parquet | Low | High |

### Phase 2: Model Modifications (Week 3-4)

| Task | Effort | Priority |
|------|--------|----------|
| Create packet embedding layer | Medium | Critical |
| Create direction encoder | Medium | Critical |
| Modify LSTM model | Medium | High |
| Modify Transformer model | Medium | High |
| Update training pipeline | Low | High |

### Phase 3: Attack Modifications (Week 5)

| Task | Effort | Priority |
|------|--------|----------|
| Design packet direction attack | Medium | High |
| Implement constrained perturbation | Medium | High |
| Update adversarial training | Low | High |

### Phase 4: Testing & Validation (Week 6)

| Task | Effort | Priority |
|------|--------|----------|
| Unit tests | Medium | High |
| Integration tests | Medium | High |
| Performance benchmarks | Medium | Medium |
| Documentation | Low | Medium |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| JSON parsing errors | Medium | High | Incremental processing, error logging |
| CSV-Join mismatch | High | Critical | Multiple match criteria, fuzzy matching |
| Memory overflow | Medium | Medium | Streaming, chunked processing |
| Model regression | Low | Medium | Keep old models as baseline |

---

## Success Criteria

| Metric | Baseline | Target |
|--------|----------|--------|
| Clean accuracy | 90% | ≥90% |
| Feature attack accuracy | 46% | ≥50% |
| Sequence attack accuracy | 70% | ≤65% |
| Training time | 2-3 hours | ≤4 hours |

---

## Rollback Plan

If implementation fails:
1. Keep original CSV-only pipeline as fallback
2. New code in separate modules (not modifying existing)
3. Configuration flag to switch between pipelines
4. All changes in feature branch

---

Return to [Main Index](../README.md)
