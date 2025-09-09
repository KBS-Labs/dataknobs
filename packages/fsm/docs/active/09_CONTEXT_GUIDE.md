# FSM Project Context Guide

## Purpose
This guide provides a roadmap for navigating the FSM project documentation and understanding the current state, next steps, and how to use each document effectively.

## Document Overview

### Foundation Documents (01-05)
These established the initial design and implementation framework:

- **01-04**: Initial design documents (details in 05)
- **05_UPDATED_IMPLEMENTATION_PLAN.md**: Original comprehensive plan with 10 phases
  - *When to reference*: For detailed phase requirements and original design intentions
  - *Status*: Phases 1-6 complete, Phase 7 in progress, Phases 8-10 pending

### Current State Documents (06-08)
These capture the current reality and learnings:

- **06_LEARNINGS.md**: Implementation insights from Phase 7
  - *When to reference*: Before making design decisions or fixing bugs
  - *Key sections*: Common Pitfalls, Architectural Decisions, Testing Strategy

- **07_IMPLEMENTATION_STATUS.md**: Current project snapshot
  - *When to reference*: Daily - this is your primary working document
  - *Key sections*: Current Issues to Resolve, Test Results, Next Steps

- **08_ARCHITECTURE_DECISIONS.md**: Formal ADRs for key design choices
  - *When to reference*: When implementing new features or questioning design
  - *Key ADRs*: ADR-001 (arc formats), ADR-002 (async-first), ADR-005 (testing strategy)

## Current Status Summary

### Where We Are
- **Overall Progress**: ~85% complete
- **Test Status**: 24/24 passing (100%) for Simple & Advanced API tests
- **Phase 7**: 95% complete - CLI done, APIs implemented, all API tests passing

### Recently Completed
1. **Simple API Test Fixes** - All 12 tests now passing
2. **Advanced API Test Fixes** - All 12 tests now passing  
3. **Arc Name Filtering** - Added support for named arc filtering in both sync and async engines
4. **Execution Context Issues** - Fixed dual state tracking (string name + StateInstance)
5. **Function Registration** - Fixed inline function compilation and registration
6. **Config Handling** - Enhanced to support legacy pre_test format

### Current Focus
- **API Refactoring** - Before proceeding with Phase 8, we need to address code duplication and API-specific workarounds identified during test fixes (see 10_API_TO_GENERAL_REFACTOR.md)

## Next Steps Workflow

### Step 1: API to General Level Refactor (Priority 1) 
**Goal**: Eliminate code duplication and API-specific workarounds before proceeding
**Reference**: 10_API_TO_GENERAL_REFACTOR.md

- [ ] **Phase 1: Core Infrastructure**
  - [ ] Create ContextFactory class
  - [ ] Create ResultFormatter utility class  
  - [ ] Enhance ExecutionContext with get_complete_path() method
- [ ] **Phase 2: ResourceManager Enhancement** 
  - [ ] Add factory methods for provider creation
  - [ ] Remove resource creation logic from APIs
- [ ] **Phase 3: Engine Lifecycle Standardization**
  - [ ] Standardize engine management patterns across APIs
- [ ] **Phase 4: FSM Core Enhancements**
  - [ ] Improve state resolution in FSM core
- [ ] **Phase 5: API Refactoring**
  - [ ] Refactor Simple API to use general infrastructure
  - [ ] Refactor Advanced API to use general infrastructure
- [ ] **Phase 6: Testing and Validation**
  - [ ] Verify all tests still pass after refactoring
  - [ ] Ensure no performance regressions

### Step 2: Complete Phase 7 (Priority 2)
**Reference**: 07_IMPLEMENTATION_STATUS.md - "Phase 7" section

- [x] Fix all API test failures (COMPLETED)
- [ ] Add missing integration tests
- [ ] Complete API documentation
- [ ] Create usage examples
- [ ] Update 07_IMPLEMENTATION_STATUS.md to show Phase 7 complete

### Step 3: Revisit 05_UPDATED_IMPLEMENTATION_PLAN
**After API refactoring and Phase 7 completion, review original plan**

- [ ] Review Phase 7 requirements in detail
- [ ] Identify any missing features from original plan
- [ ] Create issues for gaps
- [ ] Update plan for Phases 8-10 based on learnings

### Step 4: Begin Phase 8 - Monitoring and Observability
**Reference**: 05_UPDATED_IMPLEMENTATION_PLAN.md - "Phase 8" section

- [ ] Review phase requirements
- [ ] Apply learnings from 06_LEARNINGS.md
- [ ] Follow patterns from 08_ARCHITECTURE_DECISIONS.md
- [ ] Create new tests using real implementations (ADR-005)

## Working Guidelines

### Daily Workflow
1. **Start**: Check 07_IMPLEMENTATION_STATUS.md for current priorities
2. **Before coding**: Review relevant section in 06_LEARNINGS.md
3. **Design decisions**: Consult 08_ARCHITECTURE_DECISIONS.md
4. **Implementation**: Follow patterns established in ADRs
5. **Testing**: Use real implementations (ADR-005)
6. **Update**: Mark progress in this document and 07_IMPLEMENTATION_STATUS.md

### When Stuck
1. **Check learnings**: 06_LEARNINGS.md - "Common Pitfalls"
2. **Review decisions**: 08_ARCHITECTURE_DECISIONS.md for design rationale
3. **Original intent**: 05_UPDATED_IMPLEMENTATION_PLAN.md for requirements
4. **Current state**: 07_IMPLEMENTATION_STATUS.md for known issues

### Adding New Features
1. **Check ADRs**: Ensure alignment with architectural decisions
2. **Apply learnings**: Avoid pitfalls documented in 06_LEARNINGS.md
3. **Update status**: Add to 07_IMPLEMENTATION_STATUS.md
4. **Document decisions**: Create new ADR if significant

## Progress Tracking

### Test Progress
```
Current API Tests: 24/24 passing (100%) ✅
- Simple API: 12/12 passing  
- Advanced API: 12/12 passing

Recent Fixes:
[x] Transition Execution: Fixed execution context state tracking
[x] Arc Name Filtering: Added support for named arc filtering
[x] Function Registration: Fixed inline function compilation  
[x] Config Handling: Enhanced legacy format support
[x] All Test Categories: Now passing
```

### Phase Progress
```
[x] Phase 1: Core FSM Engine (100%)
[x] Phase 2: Execution Strategies (100%)
[x] Phase 3: Configuration Management (100%)
[x] Phase 4: Resource Management (100%)
[x] Phase 5: Functions and Connectors (100%)
[x] Phase 6: Streaming and Events (100%)
[~] Phase 7: API and Integration (95% - API tests complete, refactoring needed)
[ ] Phase 8: Monitoring and Observability (0%)
[ ] Phase 9: Advanced Features (0%)
[ ] Phase 10: Production Readiness (0%)
```

## Key Files for Debugging

### Test Files
- `tests/test_api_simple_real.py` - Main integration tests
- `tests/test_cli_real.py` - CLI tests

### Core Implementation
- `src/dataknobs_fsm/api/simple.py` - SimpleFSM (main API)
- `src/dataknobs_fsm/execution/engine.py` - Execution logic
- `src/dataknobs_fsm/execution/batch.py` - Batch processing
- `src/dataknobs_fsm/execution/stream.py` - Stream processing
- `src/dataknobs_fsm/core/arc.py` - Arc execution

## Success Criteria

### Phase 7 Completion
- [x] All API tests passing (24/24)
- [ ] API refactoring complete (10_API_TO_GENERAL_REFACTOR.md)
- [ ] API documentation complete
- [ ] Usage examples created
- [ ] 07_IMPLEMENTATION_STATUS updated to 100% for Phase 7

### Project Completion
- All 10 phases complete
- Comprehensive test coverage (>90%)
- Full documentation
- Production-ready features
- Performance benchmarks met

## Notes for Next Session

When resuming work:
1. Start here (09_CONTEXT_GUIDE.md)
2. Check progress checkboxes above
3. Review 07_IMPLEMENTATION_STATUS.md for current state
4. Continue with next unchecked item in "Next Steps Workflow"

Remember: The documents work together:
- **This guide** (09): Navigation and workflow
- **Status** (07): Current state and issues
- **Learnings** (06): Insights to apply
- **ADRs** (08): Design decisions to follow
- **Original Plan** (05): Requirements to fulfill