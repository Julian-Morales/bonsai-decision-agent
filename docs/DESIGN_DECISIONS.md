# Design Decisions

This document summarizes the main architectural choices behind the Bonsai Decision Agent.

## 1. One decision agent, not a multi-agent system

**Problem:** Multiple cooperating agents would add routing and synchronization complexity without improving the underlying deterministic decisions.

**Decision:** Use one `DecisionAgentV0` with explicit planning and a registry of deterministic problem domains.

**Why:** The difficult part is controlling execution, not creating more language-model personas.

**Trade-off:** The single planner remains an important dependency for interpreting user intent.

## 2. Explicit plans instead of unrestricted tool calling

**Problem:** Direct LLM-to-tool dispatch makes it harder to validate intent, domain transitions, and argument provenance before execution.

**Decision:** Require an explicit plan that is validated before application-tool dispatch.

**Why:** Plans provide a stable control point for safety checks, approvals, and traceability.

**Trade-off:** Planning adds an extra model step and some latency.

## 3. Deterministic code owns business execution

**Problem:** LLM-generated calculations and constraints are not reliable enough to be authoritative optimization logic.

**Decision:** Keep optimization, economics, comparison, and validation in deterministic Python code.

**Why:** Results become reproducible and auditable, while the LLM remains useful for intent interpretation and explanation.

**Trade-off:** New business capabilities must be implemented explicitly rather than improvised by the model.

## 4. Independent validation

**Problem:** A solver producing a candidate should not be the only component declaring that candidate acceptable.

**Decision:** Validate registered solutions through separate deterministic validators.

**Why:** This creates a second check on feasibility and encoded business rules.

**Trade-off:** Each domain requires validator logic in addition to optimizer logic.

## 5. Fail closed on unsupported required capabilities

**Problem:** A planner can propose a familiar-looking parameter that does not actually encode the user's requested constraint.

**Decision:** Block required unsupported capabilities before requested optimizer dispatch.

**Why:** The system should say “unsupported” rather than produce a result for a different problem.

**Trade-off:** Some requests that sound reasonable cannot proceed until the deterministic model is extended.

## 6. Human approval for protected non-default actions

**Problem:** A valid plan can still change business assumptions or solver behavior in ways the user should review.

**Decision:** Pause selected actions for explicit approval and bind approval to the prepared action/state.

**Why:** This keeps the user in control without requiring approval for ordinary read-only interactions.

**Trade-off:** Approval introduces an intentional interruption in otherwise automated workflows.

## 7. Controlled recovery instead of blind retries

**Problem:** Retrying after an uncertain dispatch can duplicate side effects or obscure what actually happened.

**Decision:** Distinguish failures before dispatch from uncertain outcomes after dispatch, and do not blindly re-run uncertain actions.

**Why:** Recovery behavior remains auditable and safer.

**Trade-off:** Some failures require manual review rather than automatic continuation.

## 8. Multi-domain registry instead of one universal optimization schema

**Problem:** Bonsai Packaging and Production Capacity Planning have different data, parameters, constraints, optimizers, and validators.

**Decision:** Share the orchestration/control plane while keeping separate domain adapters and deterministic implementations.

**Why:** This demonstrates architectural reuse without forcing unrelated domains into the same mathematical abstraction.

**Trade-off:** Adding a domain still requires deliberate adapter and validation work.

## 9. Product UI separated from the core

**Problem:** Product presentation evolves faster than optimization and safety semantics.

**Decision:** Keep the local web layer as an adapter/presentation surface over the accepted core behavior.

**Why:** UI improvements do not require weakening or rewriting deterministic control logic.

**Trade-off:** Structured presentation mappings must be maintained as core result types evolve.

## 10. Local connection profiles and server-side secrets

**Problem:** A portfolio/demo application needs convenient provider switching without embedding API keys in frontend code.

**Decision:** Store profile metadata locally and keep keys in a server-side `data/secrets.json` file that normal frontend reads do not expose.

**Why:** It is simple, portable, and appropriate for trusted local use.

**Trade-off:** Secrets are plaintext on disk, so this is not a production secret-management design.

## 11. No unnecessary RAG, database, or orchestration framework

**Problem:** Extra infrastructure can make an agent project look more complex without improving the decision workflow.

**Decision:** Use direct deterministic state stores, a small local HTTP layer, and the existing provider client rather than adding RAG, a database, LangGraph, or a multi-agent framework without a concrete need.

**Why:** The architecture remains easier to audit and explain.

**Trade-off:** Session history is not durable and there is no production-scale persistence layer.
