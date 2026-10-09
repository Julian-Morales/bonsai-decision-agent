# Architecture

The Bonsai Decision Agent separates probabilistic language reasoning from deterministic business execution.

## High-level structure

```mermaid
flowchart TB
    subgraph UI[Product layer]
      W[Three-panel web UI]
      API[Local HTTP API / UI adapter]
      CP[Connection profiles]
    end

    subgraph CONTROL[Agent control plane]
      A[DecisionAgentV0]
      P[Planner]
      PV[PlanValidator]
      CG[RequirementCapabilityGuard]
      AP[ApprovalPolicy]
      EX[Controlled executor]
    end

    subgraph DOMAIN[Domain layer]
      PR[ProblemRegistry]
      B[Bonsai Packaging adapter]
      C[Production Capacity adapter]
    end

    subgraph COMPUTE[Deterministic compute]
      BO[Bonsai optimizer]
      BV[Bonsai independent validator]
      PO[Production optimizer]
      PV2[Production independent validator]
    end

    subgraph STATE[Session state and provenance]
      SC[Scenarios]
      RH[Run history]
      SS[Solution store]
      AL[Approvals / execution ledger]
    end

    W --> API --> A
    CP --> API
    A --> P --> PV --> CG --> AP --> EX
    EX --> PR
    PR --> B --> BO --> BV
    PR --> C --> PO --> PV2
    EX --> SC
    EX --> RH
    EX --> SS
    AP --> AL
```

## Controlled execution lifecycle

```mermaid
flowchart LR
    Q[User request] --> PLAN[Explicit plan]
    PLAN --> VALIDATE[Validate structure, tools, bindings, domain transitions]
    VALIDATE --> GUARD[Check required capabilities]
    GUARD -->|Unsupported| BLOCK[Block before requested dispatch]
    GUARD -->|Supported| POLICY[Approval policy]
    POLICY -->|Approval required| WAIT[Await explicit approval]
    WAIT -->|Rejected| REJECT[Stop]
    WAIT -->|Approved| EXEC[Resume approved action]
    POLICY -->|No approval| EXEC
    EXEC --> TOOL[Domain tool]
    TOOL --> OPT[Deterministic optimizer]
    OPT --> CHECK[Independent validator]
    CHECK --> RESULT[Registered result + provenance]
```

## Responsibilities

### DecisionAgentV0

Coordinates the interaction. It owns neither the optimization mathematics nor final authority over executable constraints.

### Planner

Uses the configured language model to translate a natural-language request into explicit steps and argument bindings.

### PlanValidator

Rejects invalid plans before execution: unsupported tool transitions, malformed bindings, or unsafe cross-domain behavior do not silently proceed.

### RequirementCapabilityGuard

Checks whether a required user constraint is actually enforceable by the active deterministic problem. If not, the request fails closed before optimizer dispatch.

### ApprovalPolicy

Non-default protected actions can pause before execution. Approval is bound to the prepared action/state so a later resume does not silently change what the user approved.

### Controlled executor

Dispatches only validated, permitted steps and records execution state. Recovery distinguishes pre-dispatch failures from uncertain post-dispatch outcomes.

### ProblemRegistry

Provides one orchestration surface over separate deterministic domains. Bonsai Packaging and Production Capacity Planning share control-plane behavior without sharing one giant optimization schema.

### Optimizers and independent validators

Each domain contains deterministic computation and a separate validator. The language model never computes authoritative optimization results.

## UI boundary

The web application is a presentation and interaction layer around the accepted agent/core behavior. It provides:

- problem and scenario navigation;
- natural-language conversation;
- approval and capability cards;
- structured Decision View results;
- recent-run inspection;
- connection profiles and model discovery;
- optional Developer details for plans, traces, approvals, and provenance.

The UI does not own optimization rules.

## State and persistence

Scenarios, registered solutions, and completed runs are session-local. Connection-profile metadata and local UI metadata can persist under `data/`, but this is not a durable production database.

API keys are stored in `data/secrets.json` as plaintext on the local filesystem. The application is designed for trusted local use and should not be exposed as a public multi-user service without a separate security/deployment layer.

## Technical inspection in the UI

The business-facing interface keeps execution internals collapsed by default, while **Developer details** can expose the interaction-specific plan, tool trace, execution record, approvals, and ledger when a technical reviewer wants to inspect them.

![Developer details for a validated Production Capacity run](images/developer-details.png)
