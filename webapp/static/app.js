const state = {
  workspace: {problems: [], scenarios: [], runs: []},
  connections: {profiles: [], active_profile_id: null, allow_key_exposure: false},
  selectedProfileId: null,
  selectedRunKey: null,
  selectedScenarioKey: null,
  currentView: null,
  showArchivedRuns: false,
  busy: false,
  switchBusy: false,
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {'Content-Type': 'application/json', ...(options.headers || {})},
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload?.error?.message || payload?.message || `Request failed (${response.status})`;
    const error = new Error(message);
    error.payload = payload;
    error.status = response.status;
    throw error;
  }
  return payload;
}

function escapeValue(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toLocaleString(undefined, {maximumFractionDigits: 4});
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

// Centralized display-only formatting; computation never uses formatted strings.
function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return value.toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits});
}
function formatCurrency(value) {
  if (value === null || value === undefined || typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return `USD ${Math.abs(value) >= 1_000_000 ? `${formatNumber(value / 1_000_000, 2)}M` : formatNumber(value, 2)}`;
}
function formatPercent(value) {
  return typeof value === 'number' && Number.isFinite(value) ? `${formatNumber(value, 2)}%` : '—';
}
function formatMetric(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value !== 'number' || !Number.isFinite(value)) return escapeValue(value);
  if (Number.isInteger(value) && Math.abs(value) < 1_000_000) return value.toLocaleString('en-US');
  return formatNumber(value, 2);
}

function activeProblem() {
  return state.workspace.problems.find((p) => p.active) || null;
}

// Presentation names/colors are allowlisted labels, not capability metadata.
function problemDisplayName(problem) {
  if (!problem) return 'No active problem';
  return ({bonsai_packaging: 'Bonsai Packaging', production_capacity_planning: 'Production Capacity'})[problem.problem_id] || problem.display_name;
}

function runMeta(run) {
  return (state.workspace.run_ui_metadata || []).find((m) => m.problem_id === run.problem_id && m.run_id === run.run_id) || {};
}

function runMetadataPath(run, action) {
  return `/api/run-ui/${encodeURIComponent(run.problem_id)}/${encodeURIComponent(run.run_id)}/${action}`;
}

async function updateRunMetadata(run, action) {
  let body = {};
  if (action === 'rename') {
    const response = window.prompt(`Label for ${run.run_id} (run ID stays unchanged)`, runMeta(run).label || run.run_id);
    if (response === null) return;
    body = {label: response};
  }
  try {
    await api(runMetadataPath(run, action), {method: 'POST', body: JSON.stringify(body)});
    // UI metadata is deliberately separate from immutable run snapshots.
    await loadWorkspace();
  } catch (error) { appendSystemError(error.message); }
}

function renderWorkspace() {
  const problems = state.workspace.problems || [];
  const scenarios = state.workspace.scenarios || [];
  const runs = state.workspace.runs || [];
  $('problemCount').textContent = problems.length;
  $('scenarioCount').textContent = scenarios.length;
  const visibleRuns = runs.filter((r) => !runMeta(r).archived);
  const archivedRuns = runs.filter((r) => runMeta(r).archived);
  $('runCount').textContent = visibleRuns.length;

  const problemList = $('problemList');
  problemList.replaceChildren();
  problems.forEach((problem) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `nav-item domain-nav${problem.active ? ' active' : ''}${problem.problem_id === 'production_capacity_planning' ? ' domain-production' : ' domain-bonsai'}`;
    const title = document.createElement('div'); title.className = 'nav-title';
    title.textContent = problemDisplayName(problem);
    const meta = document.createElement('div'); meta.className = 'nav-meta';
    meta.textContent = problem.active ? 'Active problem' : 'Switch to this problem';
    title.title = title.textContent;
    button.append(title, meta);
    button.addEventListener('click', () => selectProblem(problem.problem_id));
    problemList.append(button);
  });
  if (!problems.length) problemList.innerHTML = '<div class="nav-empty">No registered problems.</div>';

  const scenarioList = $('scenarioList'); scenarioList.replaceChildren();
  scenarios.slice().reverse().slice(0, 10).forEach((scenario) => {
    const key = `${scenario.problem_id}:${scenario.handle}`;
    const item = document.createElement('button'); item.type = 'button';
    item.className = `nav-item scenario-item${state.selectedScenarioKey === key ? ' selected' : ''}`;
    item.setAttribute('aria-pressed', state.selectedScenarioKey === key ? 'true' : 'false');
    const title = document.createElement('div'); title.className = 'nav-title'; title.textContent = scenario.name;
    const meta = document.createElement('div'); meta.className = 'nav-meta';
    const problem = problems.find((p) => p.problem_id === scenario.problem_id);
    meta.textContent = `${problemDisplayName(problem)} · ${scenario.optimized ? 'optimized' : 'not run'}`;
    title.title = scenario.name;
    item.append(title, meta);
    item.addEventListener('click', () => selectScenario(scenario));
    scenarioList.append(item);
  });
  if (!scenarios.length) scenarioList.innerHTML = '<div class="nav-empty">No scenarios yet. Ask the agent to create one for the current problem.</div>';

  const runList = $('runList'); runList.replaceChildren();
  const listed = state.showArchivedRuns ? archivedRuns : visibleRuns;
  // All hidden runs must remain reachable for reversible restoration.
  const displayed = state.showArchivedRuns ? listed.slice().reverse() : listed.slice().reverse().slice(0, 12);
  displayed.forEach((run) => {
    const key = `${run.problem_id}:${run.run_id}`;
    const meta = runMeta(run);
    const wrapper = document.createElement('div');
    wrapper.className = `run-row${state.selectedRunKey === key ? ' selected' : ''}`;
    const item = document.createElement('button'); item.type = 'button';
    item.className = `nav-item run-item${state.selectedRunKey === key ? ' selected' : ''}`;
    item.disabled = run.status !== 'completed';
    item.setAttribute('aria-pressed', state.selectedRunKey === key ? 'true' : 'false');
    item.title = run.status === 'completed' ? `View historical result ${run.run_id}` : `Run ${run.run_id} did not complete`;
    const title = document.createElement('div'); title.className = 'nav-title';
    title.textContent = meta.label || run.run_id; title.title = title.textContent;
    const details = document.createElement('div'); details.className = 'nav-meta';
    const problem = problems.find((p) => p.problem_id === run.problem_id);
    details.textContent = `${problemDisplayName(problem)} · ${run.run_kind} · ${run.status}`;
    item.append(title, details);
    if (run.status === 'completed') item.addEventListener('click', () => selectHistoricalRun(run));
    wrapper.append(item);

    const menuButton = document.createElement('button');
    menuButton.type = 'button'; menuButton.className = 'run-menu-button';
    menuButton.textContent = '⋯'; menuButton.setAttribute('aria-label', `Actions for ${run.run_id}`);
    menuButton.setAttribute('aria-expanded', 'false'); menuButton.setAttribute('aria-haspopup', 'true');
    const menu = document.createElement('div'); menu.className = 'run-overflow-menu hidden'; menu.setAttribute('role', 'menu');
    for (const [label, action] of [['Rename', 'rename'], [meta.archived ? 'Restore to Recent Runs' : 'Remove from Recent Runs', meta.archived ? 'restore' : 'archive']]) {
      const control = document.createElement('button'); control.type = 'button'; control.setAttribute('role', 'menuitem');
      control.textContent = label; control.addEventListener('click', (event) => {
        event.stopPropagation(); menu.classList.add('hidden'); updateRunMetadata(run, action); menuButton.setAttribute('aria-expanded','false');
      });
      menu.append(control);
    }
    menuButton.addEventListener('click', (event) => {
      event.stopPropagation();
      const opening = menu.classList.contains('hidden');
      menu.classList.toggle('hidden', !opening);
      menuButton.setAttribute('aria-expanded', opening ? 'true' : 'false');
      if (opening) menu.querySelector('button')?.focus();
    });
    menuButton.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') { menu.classList.add('hidden'); menuButton.setAttribute('aria-expanded', 'false'); menuButton.focus(); }
    });
    menu.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        event.stopPropagation(); menu.classList.add('hidden');
        menuButton.setAttribute('aria-expanded', 'false'); menuButton.focus();
      }
    });
    wrapper.append(menuButton, menu); runList.append(wrapper);
  });
  if (!listed.length) {
    const empty = document.createElement('div'); empty.className = 'nav-empty';
    empty.textContent = state.showArchivedRuns ? 'No hidden runs.' : 'No completed runs yet. Ask a decision question to get started.';
    runList.append(empty);
  }
  const toggle = $('toggleArchivedRuns');
  toggle.classList.toggle('hidden', !archivedRuns.length && !state.showArchivedRuns);
  toggle.textContent = state.showArchivedRuns ? `Back to Recent Runs (${visibleRuns.length})` : `Show hidden runs (${archivedRuns.length})`;
  toggle.setAttribute('aria-expanded', state.showArchivedRuns ? 'true' : 'false');

  const active = activeProblem();
  const name = problemDisplayName(active);
  $('activeProblemPill').textContent = name;
  $('activeDomainHeader').textContent = `Current decision context · ${name}`;
  $('composerDomainContext').textContent = name;
  const panel = document.querySelector('.conversation-panel');
  panel.classList.toggle('domain-production', active?.problem_id === 'production_capacity_planning');
  panel.classList.toggle('domain-bonsai', active?.problem_id === 'bonsai_packaging');
  const select = $('headerProblemSelect');
  const prior = select.value;
  select.replaceChildren();
  problems.forEach((problem) => {
    const option = document.createElement('option');
    option.value = problem.problem_id; option.textContent = problemDisplayName(problem);
    select.append(option);
  });
  select.value = problems.some((p) => p.problem_id === prior) ? prior : (active?.problem_id || '');
  $('switchProblemButton').disabled = state.switchBusy || !select.value || select.value === active?.problem_id;
  const parameterList = $('parameterList'); parameterList.replaceChildren();
  renderParameterGroups(parameterList, active?.problem_id, active?.configurable_parameters || []);
}

async function selectProblem(problemId) {
  if (state.switchBusy) return;
  state.switchBusy = true;
  const switchButton = $('switchProblemButton');
  switchButton.disabled = true;
  switchButton.classList.add('control-busy');
  switchButton.textContent = 'Switching…';
  try {
    await api(`/api/problems/${encodeURIComponent(problemId)}/select`, {method: 'POST', body: '{}'});
    await loadWorkspace();
    state.selectedRunKey = null;
    state.selectedScenarioKey = null;
    renderWorkspace();
    renderDecision(null); // explicit domain switch, not a plan or automatic retry
  } catch (error) {
    appendSystemError(error.message);
  } finally {
    state.switchBusy = false;
    switchButton.classList.remove('control-busy');
    switchButton.textContent = 'Switch';
    switchButton.disabled = !switchButton || !$('headerProblemSelect').value || $('headerProblemSelect').value === activeProblem()?.problem_id;
  }
}

async function selectHistoricalRun(run) {
  if (!run || run.status !== 'completed') return;
  const runKey = `${run.problem_id}:${run.run_id}`;
  if (state.selectedRunKey === runKey) return;
  try {
    const view = await api(`/api/problems/${encodeURIComponent(run.problem_id)}/runs/${encodeURIComponent(run.run_id)}`);
    state.selectedRunKey = runKey;
    state.selectedScenarioKey = null;
    renderWorkspace();
    renderDecision(view);
  } catch (error) {
    appendSystemError(error.message);
  }
}

async function selectScenario(scenario) {
  if (!scenario) return;
  const key = `${scenario.problem_id}:${scenario.handle}`;
  if (state.selectedScenarioKey === key) return;
  try {
    const view = await api(`/api/problems/${encodeURIComponent(scenario.problem_id)}/scenarios/${encodeURIComponent(scenario.handle)}`);
    state.selectedScenarioKey = key;
    state.selectedRunKey = null;
    renderWorkspace();
    renderDecision(view);  // read-only history navigation, never chat or problem selection
  } catch (error) { appendSystemError(error.message); }
}

function appendMessage(role, text, developer = null, presentation = null) {
  const wrapper = document.createElement('div');
  wrapper.className = `message-row ${role}`;
  const bubble = document.createElement('div'); bubble.className = 'message-bubble';
  const label = document.createElement('div'); label.className = 'message-label';
  label.textContent = role === 'user' ? 'You' : 'Decision Agent';
  const primary = document.createElement('div'); primary.className = 'message-content';
  const hasSummary = role === 'assistant' && presentation?.show_full_response && presentation?.summary;
  if (hasSummary) {
    const compact = document.createElement('p'); compact.className = 'message-summary';
    compact.textContent = presentation.summary; primary.append(compact);
    const details = document.createElement('details'); details.className = 'agent-response-details';
    const summary = document.createElement('summary'); summary.textContent = 'Full agent response';
    details.append(summary, window.renderSafeMarkdown(String(text || ''))); primary.append(details);
  } else if (role === 'assistant') {
    primary.append(window.renderSafeMarkdown(String(text || '')));
  } else { primary.textContent = text; }
  if (role === 'assistant' && presentation?.synthesis_warning) {
    const warning = document.createElement('p'); warning.className = 'synthesis-warning';
    warning.textContent = presentation.synthesis_warning; primary.append(warning);
  }
  bubble.append(label, primary);
  if (developer && role === 'assistant') {
    const details = $('developerDetailsTemplate').content.firstElementChild.cloneNode(true);
    const rawJson = JSON.stringify(developer, null, 2);
    details.querySelector('pre').textContent = rawJson;
    const copy = document.createElement('button'); copy.type = 'button';
    copy.className = 'copy-json-button secondary-button compact'; copy.textContent = 'Copy JSON';
    copy.setAttribute('aria-label', 'Copy this interaction JSON');
    copy.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(rawJson); copy.textContent = 'Copied'; }
      catch (_) { copy.textContent = 'Copy unavailable'; }
    });
    details.append(copy);
    bubble.append(details);
  }
  wrapper.append(bubble);
  $('messages').append(wrapper);
  $('conversationScroll').scrollTop = $('conversationScroll').scrollHeight;
}

function appendSystemError(message) {
  appendMessage('assistant', `Application error: ${message}`);
}

function workflowClass(type) {
  if (['blocked_unsupported_requirement', 'approval_pending', 'invalid_domain_transition'].includes(type)) return 'warning';
  if (['approval_rejected', 'error', 'outcome_unknown'].includes(type)) return 'danger';
  if (['validation_result'].includes(type)) return 'info';
  return '';
}

function addLine(parent, label, value) {
  if (value === null || value === undefined || value === '' || (Array.isArray(value) && !value.length)) return;
  const line = document.createElement('div');
  const strong = document.createElement('strong');
  strong.textContent = `${label}: `;
  line.append(strong, document.createTextNode(escapeValue(value)));
  parent.append(line);
}

function renderWorkflowCards(cards) {
  if (!cards?.length) return;
  const stack = document.createElement('div');
  stack.className = 'workflow-stack';
  cards.filter((card) => !['optimization_success', 'approval_approved', 'approval_rejected'].includes(card.type) && !(card.type === 'validation_result' && card.valid !== false)).forEach((card) => {
    const element = document.createElement('article');
    if (card.type === 'approval_pending') {
      element.dataset.approvalId = card.approval_id;
      element.dataset.approvalStatus = 'pending';
    }
    element.className = `workflow-card ${workflowClass(card.type)}`;
    const title = document.createElement('div');
    title.className = 'workflow-title';
    title.textContent = card.title || card.type;
    const body = document.createElement('div');
    body.className = 'workflow-body';
    if (card.type === 'approval_pending') {
      addLine(body, 'Action', friendlyLabel(card.tool));
      addLine(body, 'Problem', nameForProblemId(card.problem_id));
      addLine(body, 'Reason approval is required', (card.reason_codes || []).map(friendlyLabel).join(', '));
      if (card.summary?.parameter_overrides && Object.keys(card.summary.parameter_overrides).length) {
        appendParameterChanges(body, card.problem_id, card.summary.parameter_overrides);
      }
      const actions = document.createElement('div');
      actions.className = 'workflow-actions';
      const approve = document.createElement('button');
      approve.className = 'primary-button compact';
      approve.textContent = 'Approve';
      approve.addEventListener('click', () => decideApproval(card.approval_id, 'approve'));
      const reject = document.createElement('button');
      reject.className = 'danger-button compact';
      reject.textContent = 'Reject';
      reject.addEventListener('click', () => decideApproval(card.approval_id, 'reject'));
      actions.append(reject, approve);
      body.append(actions);
    } else if (card.type === 'invalid_domain_transition') {
      const problems = state.workspace.problems || [];
      const requested = problems.find((p) => p.problem_id === card.requested_problem_id);
      const current = problems.find((p) => p.problem_id === card.active_problem_id);
      addLine(body, 'Requested problem', requested ? problemDisplayName(requested) : 'Not deterministically identified');
      addLine(body, 'Currently active', problemDisplayName(current));
      addLine(body, 'Why not executed', card.why_not_executed);
      addLine(body, 'Execution', 'No execution occurred');
      const line = document.createElement('div'); line.className = 'workflow-actions';
      if (requested && !requested.active) {
        const switchButton = document.createElement('button');
        switchButton.type = 'button'; switchButton.className = 'secondary-button compact';
        switchButton.textContent = `Switch to ${problemDisplayName(requested)}`;
        switchButton.addEventListener('click', async () => {
          await selectProblem(requested.problem_id); // explicit user action; no retry
          switchButton.disabled = true;
        });
        line.append(switchButton);
      }
      body.append(line);
    } else if (card.type === 'blocked_unsupported_requirement') {
      addLine(body, 'Requested capability', friendlyLabel(card.requested_capability));
      addLine(body, 'Requested value / bound', card.requested_requirement);
      addLine(body, 'Why unsupported', card.why_unsupported);
      const noExec = document.createElement('div');
      noExec.style.marginTop = '6px';
      const strong = document.createElement('strong');
      strong.textContent = 'No optimization was executed.';
      noExec.append(strong);
      body.append(noExec);
    } else if (card.type === 'approval_rejected') {
      addLine(body, 'Action', card.tool);
      body.append(document.createTextNode('The protected action was not executed.'));
    } else if (card.type === 'optimization_success') {
      addLine(body, 'Run', card.run_id);
      addLine(body, 'Solution', card.solution_handle);
      addLine(body, 'Independent validation', card.validated === true ? 'Passed' : card.validated === false ? 'Failed' : 'Not reported');
    } else if (card.type === 'validation_result') {
      addLine(body, 'Handle', card.handle);
      addLine(body, 'Valid', card.valid);
      addLine(body, 'Errors', card.error_count);
    } else {
      addLine(body, 'Status', friendlyLabel(card.status));
      addLine(body, 'Issue', friendlyLabel(card.error_code));
      addLine(body, 'Detail', card.safe_message || card.detail);
    }
    element.append(title, body);
    stack.append(element);
  });
  $('messages').append(stack);
}

// UX-3: business copy only. Canonical IDs are unchanged in developer payloads.
const BUSINESS_LABELS = Object.freeze({
  max_box_types: 'Maximum number of box types',
  unsupported_required_capability: 'Unsupported by current optimization model',
  non_default_business_context: 'Non-default business assumption',
  pallet_cost: 'Pallet cost', demand_multiplier: 'Demand multiplier',
  overtime_cost_multiplier: 'Overtime cost multiplier', thickness: 'Box thickness',
  milp_time: 'Solver time limit', mip_gap: 'Optimality gap',
  extra_candidates: 'Extra candidates', lns_time: 'LNS time', lns_rounds: 'LNS rounds',
  target_count: 'Target count', target_time: 'Target time',
  run_optimization: 'Run optimization', create_scenario: 'Create scenario',
  validate_solution: 'Validate solution', select_problem: 'Switch problem',
  failed_before_dispatch: 'Not dispatched', outcome_unknown: 'Outcome unknown',
  invalid_domain_transition: 'Different problem selected',
  authentication_failed: 'Authentication failed',
  endpoint_failure: 'Endpoint unreachable',
});
function friendlyLabel(value) {
  const text = String(value ?? '—');
  if (BUSINESS_LABELS[text]) return BUSINESS_LABELS[text];
  return text.replaceAll('_', ' ').replace(/\b\w/g, ch => ch.toUpperCase());
}
function parameterMetadata(problemId, name) {
  const problem = state.workspace.problems.find(p => p.problem_id === problemId);
  return (problem?.configurable_parameters || []).find(param => param.name === name) || null;
}
function parameterDisplayName(problemId, name) {
  const param = parameterMetadata(problemId, name);
  return param?.display_name || BUSINESS_LABELS[name] || friendlyLabel(name);
}
// UI-only precision: keep small nonzero solver tolerances and entered fractions visible.
function formatAdaptiveNumber(value, minDigits = 2) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  if (value !== 0 && Math.abs(value) < 0.01) {
    return value.toLocaleString('en-US', {maximumSignificantDigits: 14, useGrouping: false});
  }
  return (value === 0 ? 0 : value).toLocaleString('en-US', {minimumFractionDigits: minDigits, maximumFractionDigits: 12});
}
function formatParameterValue(problemId, name, value) {
  if (value === undefined || value === null) return '—';
  if (typeof value !== 'number' || !Number.isFinite(value)) return escapeValue(value);
  const param = parameterMetadata(problemId, name);
  if (name === 'pallet_cost' && problemId === 'bonsai_packaging') return `USD ${formatAdaptiveNumber(value)}`;
  if (name === 'thickness' && problemId === 'bonsai_packaging') return `${formatAdaptiveNumber(value)} mm`;
  if (['demand_multiplier', 'overtime_cost_multiplier'].includes(name)) return `${formatAdaptiveNumber(value)}×`;
  if (['milp_time', 'lns_time', 'target_time'].includes(name)) return `${formatAdaptiveNumber(value)} s`;
  if (param?.value_type === 'int' || param?.value_type === 'integer') return formatMetric(value);
  return formatAdaptiveNumber(value);
}
function parameterConstraints(problemId, param) {
  if (Array.isArray(param.allowed_values) && param.allowed_values.length)
    return `Options ${param.allowed_values.map(value => formatParameterValue(problemId, param.name, value)).join(', ')}`;
  const min = param.minimum, max = param.maximum;
  if (min !== null && min !== undefined && min > 0 && max == null && min <= 1e-9) return 'Must be positive';
  if (min !== null && min !== undefined && max !== null && max !== undefined)
    return `Range ${formatParameterValue(problemId, param.name, min)} – ${formatParameterValue(problemId, param.name, max)}`;
  if (min !== null && min !== undefined) return `Minimum ${formatParameterValue(problemId, param.name, min)}`;
  if (max !== null && max !== undefined) return `Maximum ${formatParameterValue(problemId, param.name, max)}`;
  return '';
}
function parameterDescription(problemId, param) {
  const bits = [`Default ${formatParameterValue(problemId, param.name, param.default)}`];
  const constraint = parameterConstraints(problemId, param);
  if (constraint) bits.push(constraint);
  return bits.join(' · ');
}
function parameterTable(problemId, parameters) {
  const table = document.createElement('table'); table.className = 'kv-table parameter-overview';
  const head = document.createElement('tr');
  for (const label of ['Parameter', 'Default and allowed values']) {
    const th = document.createElement('th'); th.textContent = label; head.append(th);
  }
  table.append(head);
  parameters.forEach(param => {
    const row = document.createElement('tr');
    const name = document.createElement('td'); name.textContent = parameterDisplayName(problemId, param.name); name.title = param.name;
    const detail = document.createElement('td'); detail.textContent = parameterDescription(problemId, param);
    row.append(name, detail); table.append(row);
  });
  return table;
}
function renderParameterGroups(container, problemId, parameters) {
  if (!parameters.length) {
    const empty = document.createElement('p'); empty.textContent = 'No configurable parameters advertised by the executable problem contract.';
    container.append(empty); return;
  }
  const business = parameters.filter(param => param.category === 'business');
  const solver = parameters.filter(param => param.category === 'solver');
  const others = parameters.filter(param => param.category !== 'business' && param.category !== 'solver');
  if (business.length) {
    const heading = document.createElement('h4'); heading.className = 'parameter-group-heading'; heading.textContent = 'Business parameters';
    container.append(heading, parameterTable(problemId, business));
  }
  if (others.length) container.append(parameterTable(problemId, others));
  if (solver.length) {
    const details = document.createElement('details'); details.className = 'advanced-parameters';
    const summary = document.createElement('summary'); summary.textContent = `Advanced solver settings (${solver.length})`;
    details.append(summary, parameterTable(problemId, solver)); container.append(details);
  }
}
function appendParameterChanges(container, problemId, overrides, heading = 'Requested changes') {
  if (!overrides || typeof overrides !== 'object' || Array.isArray(overrides)) return false;
  const entries = Object.entries(overrides);
  if (!entries.length) return false;
  const section = document.createElement('section'); section.className = 'parameter-changes';
  const caption = document.createElement('div'); caption.className = 'parameter-changes-heading'; caption.textContent = heading;
  section.append(caption);
  entries.forEach(([name, requested]) => {
    const param = parameterMetadata(problemId, name);
    const row = document.createElement('div'); row.className = 'parameter-change-row';
    const label = document.createElement('span'); label.className = 'parameter-change-name';
    label.textContent = parameterDisplayName(problemId, name);
    const value = document.createElement('span'); value.className = 'parameter-change-values';
    const priorAvailable = Boolean(param && param.default !== null && param.default !== undefined);
    if (priorAvailable) {
      const prior = document.createElement('span'); prior.className = 'parameter-prior';
      prior.textContent = formatParameterValue(problemId, name, param.default);
      prior.title = 'Executable default value, not necessarily the current state';
      value.append(prior);
      const arrow = document.createElement('span'); arrow.className = 'parameter-arrow'; arrow.textContent = '→'; value.append(arrow);
    }
    const next = document.createElement('strong'); next.textContent = formatParameterValue(problemId, name, requested);
    value.append(next); row.append(label, value); section.append(row);
  });
  container.append(section);
  return true;
}
function nameForProblemId(id) {
  return problemDisplayName(state.workspace.problems.find(p => p.problem_id === id)) || id;
}
function syncApprovalCards(cards) {
  const resolved = (cards || []).filter(c => ['approval_approved', 'approval_rejected'].includes(c.type));
  for (const card of resolved) {
    for (const node of document.querySelectorAll('.workflow-card[data-approval-id]')) {
      if (node.dataset.approvalId !== String(card.approval_id) || node.dataset.approvalStatus !== 'pending') continue;
      node.dataset.approvalStatus = card.type === 'approval_approved' ? 'approved' : 'rejected';
      node.classList.toggle('danger', card.type === 'approval_rejected');
      node.classList.remove('warning');
      node.querySelector('.workflow-title').textContent = card.type === 'approval_approved' ? 'Approved ✓' : 'Rejected';
      const actions = node.querySelector('.workflow-actions');
      if (actions) { actions.querySelectorAll('button').forEach(button => button.disabled = true); actions.remove(); }
    }
  }
}

async function decideApproval(approvalId, action) {
  if (!approvalId || state.busy) return;
  setBusy(true);
  try {
    const payload = await api(`/api/approvals/${encodeURIComponent(approvalId)}/${action}`, {method: 'POST', body: '{}'});
    syncApprovalCards(payload.workflow_cards || []);
    appendMessage('assistant', payload.message?.text || `${action} completed.`, payload.developer_details, payload.presentation);
    renderWorkflowCards(payload.workflow_cards || []);
    applyAgentPayload(payload);
  } catch (error) {
    appendSystemError(error.message);
  } finally {
    setBusy(false);
  }
}

function setBusy(value) {
  const busy = Boolean(value);
  state.busy = busy;
  const button = $('sendButton');
  button.disabled = busy;
  button.classList.toggle('is-busy', busy);
  button.setAttribute('aria-busy', busy ? 'true' : 'false');
  $('sendButtonLabel').textContent = busy ? 'Sending…' : 'Send';
  $('messageInput').disabled = busy;
  document.querySelectorAll('.workflow-card[data-approval-status="pending"] button, .approval-panel-actions button').forEach(control => {
    control.disabled = busy;
  });
}

function kpi(label, value, sub = '') {
  const card = document.createElement('div');
  card.className = 'kpi';
  const l = document.createElement('div'); l.className = 'kpi-label'; l.textContent = label;
  const v = document.createElement('div'); v.className = 'kpi-value'; v.textContent = formatMetric(value);
  card.append(l, v);
  if (sub) { const s = document.createElement('div'); s.className = 'kpi-sub'; s.textContent = sub; card.append(s); }
  return card;
}

function objectTable(value, title) {
  const card = document.createElement('section');
  card.className = 'decision-card';
  if (title) { const h = document.createElement('h3'); h.textContent = title; card.append(h); }
  const table = document.createElement('table');
  table.className = 'kv-table';
  const entries = Object.entries(value || {}).filter(([, v]) => v !== null && v !== undefined && typeof v !== 'object');
  entries.slice(0, 14).forEach(([key, val]) => {
    const row = document.createElement('tr');
    const a = document.createElement('td'); a.textContent = friendlyLabel(key);
    const b = document.createElement('td'); b.textContent = escapeValue(val);
    row.append(a, b); table.append(row);
  });
  card.append(table);
  return card;
}

function addSimpleComparison(container, baseline, optimized, baselineLabel = 'Baseline', optimizedLabel = 'Optimized') {
  if (typeof baseline !== 'number' || typeof optimized !== 'number') return;
  const b = baseline, o = optimized;
  if (!Number.isFinite(b) || !Number.isFinite(o)) return;
  const max = Math.max(Math.abs(b), Math.abs(o), 1);
  const card = document.createElement('section'); card.className = 'decision-card';
  const h = document.createElement('h3'); h.textContent = `${baselineLabel} vs ${optimizedLabel}`; card.append(h);
  [[baselineLabel, b], [optimizedLabel, o]].forEach(([label, value]) => {
    const wrap = document.createElement('div'); wrap.className = 'bar-compare';
    const head = document.createElement('div'); head.className = 'bar-head';
    const left = document.createElement('span'); left.textContent = label;
    const right = document.createElement('span'); right.textContent = formatMetric(value);
    head.append(left, right);
    const track = document.createElement('div'); track.className = 'bar-track';
    const fill = document.createElement('div'); fill.className = 'bar-fill'; fill.style.width = `${Math.max(2, Math.abs(value) / max * 100)}%`;
    track.append(fill); wrap.append(head, track); card.append(wrap);
  });
  container.append(card);
}

function businessValue(value) {
  if (value === null || value === undefined) return '—';
  if (Array.isArray(value)) return value.map(escapeValue).join(' · ');
  if (typeof value === 'object') return Object.entries(value).map(([key, item]) =>
    `${friendlyLabel(key)}: ${escapeValue(item)}`).join(' · ');
  return escapeValue(value);
}
function decisionSection(title, lines) {
  const section = document.createElement('section'); section.className = 'decision-card';
  const heading = document.createElement('h3'); heading.textContent = title; section.append(heading);
  const table = document.createElement('table'); table.className = 'kv-table';
  for (const [label, value] of lines) {
    if (value === null || value === undefined || value === '') continue;
    const row = document.createElement('tr');
    const a = document.createElement('td'); a.textContent = friendlyLabel(label);
    const b = document.createElement('td'); b.textContent = businessValue(value);
    row.append(a,b); table.append(row);
  }
  if (table.children.length) section.append(table);
  return section;
}
function addDecisionKpis(container, values) {
  const grid = document.createElement('div'); grid.className = 'kpi-grid';
  for (const [label, value, formatter, note] of values) {
    if (value === null || value === undefined || typeof value !== 'number' || !Number.isFinite(value)) continue;
    const card = document.createElement('div'); card.className = 'kpi';
    const l = document.createElement('div'); l.className = 'kpi-label'; l.textContent = label;
    const v = document.createElement('div'); v.className = 'kpi-value'; v.textContent = (formatter || formatMetric)(value);
    card.append(l,v);
    if (note) { const n = document.createElement('div'); n.className = 'kpi-sub'; n.textContent = note; card.append(n); }
    grid.append(card);
  }
  if (grid.children.length) container.append(grid);
}
function decisionStatus(text, className = '') {
  const strip = document.createElement('div'); strip.className = `validation-strip ${className}`;
  strip.textContent = text; return strip;
}
function renderProblemOverview(content) {
  const problem = activeProblem();
  $('decisionTitle').textContent = problemDisplayName(problem);
  $('decisionBadge').textContent = 'Overview';
  if (!problem) {
    content.append(decisionSection('Problem overview', [['Status', 'No active registered problem']]));
    return;
  }
  const spec = problem;
  const desc = document.createElement('p'); desc.className = 'decision-intro';
  desc.textContent = spec.description || spec.objective_summary || 'Configured optimization problem';
  content.append(desc);
  content.append(decisionSection('Problem', [
    ['Display name', problemDisplayName(spec)],
    ['Purpose', spec.objective_summary || spec.description],
  ]));
  const card = document.createElement('section'); card.className = 'decision-card';
  const heading = document.createElement('h3'); heading.textContent = 'Supported configurable parameters'; card.append(heading);
  renderParameterGroups(card, problem.problem_id, spec.configurable_parameters || []);
  content.append(card);
}
function renderApprovalReview(content, data) {
  const problem = state.workspace.problems.find(p => p.problem_id === data.problem_id);
  content.append(decisionStatus('Awaiting explicit approval', 'review'));
  content.append(decisionSection('Requested action', [
    ['Problem', problemDisplayName(problem)], ['Action', friendlyLabel(data.action)],
    ['Reason', (data.reason_codes || []).map(friendlyLabel).join(', ')],
    ['Status', friendlyLabel(data.status)],
  ]));
  const changes = data.requested_changes || {};
  if (appendParameterChanges(content, data.problem_id, changes, 'Requested parameter changes')) {
    const note = document.createElement('p'); note.className = 'parameter-note';
    note.textContent = 'Prior values, where shown, are executable defaults.';
    content.append(note);
  }
  else if (data.action_summary) content.append(decisionSection('Action details', Object.entries(data.action_summary).filter(([key]) => !['parameter_overrides','approval_id'].includes(key))));
  if (data.approval_id) {
    const actions = document.createElement('div'); actions.className = 'workflow-actions approval-panel-actions';
    for (const [label, action, klass] of [['Reject', 'reject', 'danger-button'], ['Approve', 'approve', 'primary-button']]) {
      const button = document.createElement('button'); button.className = `${klass} compact`; button.type = 'button'; button.textContent = label;
      button.disabled = state.busy; button.addEventListener('click', () => decideApproval(data.approval_id, action)); actions.append(button);
    }
    content.append(actions);
  }
}
function renderDomainResult(content, data, historical = false) {
  const validated = data.validated_final || {};
  const baseline = data.baseline || {};
  const pid = data.problem_id || data.problem?.problem_id || activeProblem()?.problem_id;
  const production = pid === 'production_capacity_planning';
  const status = validated.valid === true && validated.independently_validated !== false;
  const comparable = data.reference_context_compatible !== false;
  $('decisionTitle').textContent = production ? 'Production Capacity Result' : 'Bonsai Packaging Result';
  $('decisionBadge').className = `status-badge ${historical ? 'neutral' : status ? 'good' : 'warn'}`;
  $('decisionBadge').textContent = historical ? 'Historical run' : 'Latest result';
  if (historical) {
    const meta = state.workspace.run_ui_metadata?.find(m => m.problem_id === pid && m.run_id === data.run_id);
    content.append(decisionSection('Historical result', [
      ['Label', meta?.label || data.run_id], ['Canonical run ID', data.run_id],
      ['Problem', nameForProblemId(pid)], ['Status', data.status],
      ['Run kind', friendlyLabel(data.run_kind)], ['Solution handle', data.solution_handle],
    ]));
  }
  if (validated.valid !== undefined) content.append(decisionStatus(
    validated.valid === true ? 'Independent validation passed' : 'Independent validation did not pass',
    validated.valid === true ? '' : 'bad'));
  if (production) {
    addDecisionKpis(content, [
      ['Validated cost', validated.total_cost, formatCurrency],
      ['Baseline cost', comparable ? baseline.total_cost : null, formatCurrency],
      ['Production units', validated.total_units, formatMetric],
      ['Overtime hours', validated.overtime_hours, (v) => formatNumber(v, 2)],
      ['Production cost', validated.production_cost, formatCurrency],
      ['Overtime cost', validated.overtime_cost, formatCurrency],
    ]);
    if (comparable) addSimpleComparison(content, baseline.total_cost, validated.total_cost, 'Baseline cost', 'Validated cost');
  } else {
    addDecisionKpis(content, [
      ['Validated total cost', validated.total, formatCurrency],
      ['Baseline cost', comparable ? baseline.total : null, formatCurrency],
      ['Savings', comparable ? validated.saving : null, formatCurrency],
      ['Savings %', comparable ? validated.saving_pct : null, formatPercent],
      ['Box types', validated.n_types, formatMetric],
    ]);
    if (comparable) addSimpleComparison(content, baseline.total, validated.total, 'Baseline cost', 'Optimized cost');
    addDecisionKpis(content, [
      ['Packaging cost', validated.packaging, formatCurrency],
      ['Freight cost', validated.freight, formatCurrency],
    ]);
  }
  content.append(decisionSection('Independent validation', [
    ['Result', validated.valid === true ? 'Passed' : validated.valid === false ? 'Failed' : null],
    ['Errors', validated.error_count],
  ]));
  if (data.reference_context_compatible === false) content.append(decisionStatus('Baseline comparison unavailable. This run uses different business assumptions, so a like-for-like savings comparison is not shown.', 'review'));
  if (data.effective_business_context && Object.keys(data.effective_business_context).length)
    content.append(objectTable(data.effective_business_context, 'Effective assumptions'));
  if (historical && data.parameter_overrides && Object.keys(data.parameter_overrides).length)
    appendParameterChanges(content, pid, data.parameter_overrides, 'Run parameters');
}
function renderDecision(view) {
  const content = $('decisionContent'); content.replaceChildren();
  state.currentView = view || null;
  const badge = $('decisionBadge'); badge.className = 'status-badge neutral'; badge.textContent = 'Ready';
  $('decisionTitle').textContent = view?.title || 'Decision context';
  const data = view?.data || {};
  const pid = data.problem_id || data.problem?.problem_id || null;
  const production = pid ? pid === 'production_capacity_planning' : activeProblem()?.problem_id === 'production_capacity_planning';
  $('decisionContent').classList.toggle('domain-production', production);
  $('decisionContent').classList.toggle('domain-bonsai', !production);
  if (!view || view.type === 'empty') { renderProblemOverview(content); return; }
  if (view.synthesis_warning) content.append(decisionStatus(view.synthesis_warning, 'review'));
  if (view.type === 'response_generation_issue') {
    badge.className = 'status-badge warn'; badge.textContent = 'Response issue';
    content.append(decisionStatus('No tools were executed and no decision state changed.', 'review'));
    content.append(decisionSection('Response generation issue', [
      ['Problem', nameForProblemId(data.problem_id)], ['Status', data.message],
    ]));
    return;
  }
  if (view.type === 'approval_resolved') {
    const approved = data.status === 'approved';
    badge.className = approved ? 'status-badge good' : 'status-badge bad';
    badge.textContent = approved ? 'Approved' : 'Rejected';
    content.append(decisionStatus(approved ? 'Approval recorded.' : 'Request rejected; protected action not executed.', approved ? '' : 'bad'));
    content.append(decisionSection('Approval decision', [
      ['Problem', nameForProblemId(data.problem_id)], ['Action', friendlyLabel(data.action)],
      ['Status', data.status], ['Reason', (data.reason_codes || []).map(friendlyLabel).join(', ')],
    ]));
    return;
  }
  if (view.type === 'approval_review') {
    badge.className = 'status-badge warn'; badge.textContent = 'Action required';
    renderApprovalReview(content, data); return;
  }
  if (view.type === 'domain_transition') {
    badge.className = 'status-badge warn'; badge.textContent = 'Not executed';
    content.append(decisionStatus('No execution occurred', 'review'));
    content.append(decisionSection('Different problem selected', [
      ['Currently active', nameForProblemId(data.active_problem_id)],
      ['Requested', data.requested_problem_id ? nameForProblemId(data.requested_problem_id) : 'Not deterministically identified'],
      ['Reason', data.why_not_executed],
    ]));
    return; // Switching remains solely the explicit conversation card action.
  }
  if (view.type === 'unsupported_requirement') {
    badge.className = 'status-badge warn'; badge.textContent = 'Not executed';
    content.append(decisionStatus('No optimization was executed.', 'review'));
    content.append(decisionSection('Unsupported requirement', [
      ['Problem', nameForProblemId(data.problem_id)],
      ['Capability', friendlyLabel(data.capability_id)], ['Requested value / bound', data.extracted_requirement],
      ['Reason', friendlyLabel(data.reason_code)], ['Explanation', data.safe_explanation],
    ])); return;
  }
  if (view.type === 'execution_failure') {
    badge.className = 'status-badge bad'; badge.textContent = data.status === 'outcome_unknown' || data.dispatch_state === 'outcome_unknown' ? 'Outcome unknown' : 'Execution issue';
    const dispatch = data.dispatch_state;
    const description = dispatch === 'failed_before_dispatch' || data.dispatch_attempted === false ?
      'The failing action was not dispatched.' : dispatch === 'outcome_unknown' || data.status === 'outcome_unknown' ?
      'Dispatch may have occurred; outcome cannot be confirmed. Do not retry blindly.' : 'Review dispatch details before taking action.';
    content.append(decisionStatus(description, 'bad'));
    content.append(decisionSection('Execution issue', [
      ['Status', friendlyLabel(data.plan_status || data.status)], ['Error category', friendlyLabel(data.error_code)],
      ['Problem', data.problem_id ? nameForProblemId(data.problem_id) : null],
      ['Action', friendlyLabel(data.action)], ['Step', data.step_id], ['Dispatch state', friendlyLabel(dispatch)],
      ['Optimization started', data.optimization_started === true ? 'Yes' : data.optimization_started === false ? 'No' : 'Unknown'],
      ['Message', data.safe_message],
    ])); return;
  }
  if (['optimization', 'scenario', 'historical_run_result'].includes(view.type) && data.validated_final) {
    renderDomainResult(content, data, view.type === 'historical_run_result'); return;
  }
  if (view.type === 'scenario_detail') {
    badge.textContent = 'Scenario';
    content.append(decisionSection('Scenario details', [
      ['Name', data.name], ['Problem', nameForProblemId(data.problem_id)],
      ['Scenario handle', data.handle], ['Solution handle', data.solution_handle],
    ]));
    if (data.parameter_overrides && Object.keys(data.parameter_overrides).length)
      appendParameterChanges(content, data.problem_id, data.parameter_overrides, 'Configured overrides');
    else content.append(decisionSection('Configured overrides', [['Status', 'No overrides configured']]));
    if (data.associated_runs?.length) {
      const rows = data.associated_runs.map(run => [run.run_id, `${run.status}${run.solution_handle ? ` · ${run.solution_handle}` : ''}`]);
      content.append(decisionSection('Associated runs (stored references)', rows));
    }
    return;
  }
  if (view.type === 'validation') {
    badge.className = data.valid ? 'status-badge good' : 'status-badge bad'; badge.textContent = data.valid ? 'Valid' : 'Invalid';
    content.append(decisionStatus(data.valid ? 'Independent validation passed' : 'Independent validation did not pass', data.valid ? '' : 'bad'));
    addDecisionKpis(content, [['Validated cost', data.total ?? data.total_cost, formatCurrency], ['Error count', data.error_count, formatMetric]]);
    content.append(objectTable(data, 'Validation metrics')); return;
  }
  if (view.type === 'comparison') {
    badge.className = data.comparable ? 'status-badge good' : 'status-badge warn'; badge.textContent = data.comparable ? 'Comparable' : 'Not comparable';
    if (data.solution_a || data.solution_b) {
      const columns = document.createElement('div'); columns.className = 'comparison-grid';
      if (data.solution_a) columns.append(objectTable(data.solution_a, data.solution_a_handle || 'Solution A'));
      if (data.solution_b) columns.append(objectTable(data.solution_b, data.solution_b_handle || 'Solution B'));
      content.append(columns);
    }
    const delta = data.delta_b_minus_a || {};
    const numeric = Object.entries(delta).filter(([,value]) => typeof value === 'number' && Number.isFinite(value));
    addDecisionKpis(content, numeric.slice(0,6).map(([name, val]) => [`Δ ${friendlyLabel(name)}`, val, formatMetric, 'B − A']));
    if (Object.keys(delta).length) content.append(objectTable(delta, 'Deterministic change (B − A)'));
    return;
  }
  if (view.type === 'baseline') {
    badge.textContent = 'Baseline';
    addDecisionKpis(content, Object.entries(data).filter(([, v]) => typeof v === 'number').slice(0,4).map(([k,v]) => [friendlyLabel(k),v,formatMetric]));
    content.append(objectTable(data, 'Baseline')); return;
  }
  if (view.type === 'historical_run') {
    badge.textContent = 'Run history';
    const runs = Array.isArray(data.runs) ? data.runs : [];
    runs.slice().reverse().slice(0,8).forEach(run => content.append(objectTable(run, run.run_id || 'Run')));
    if (!runs.length) content.append(objectTable(data, 'Run history'));
    return;
  }
  content.append(objectTable(data, view.title || 'Decision data'));
}

function applyAgentPayload(payload) {
  if (payload.workspace) {
    state.workspace = {...payload.workspace, run_ui_metadata: state.workspace.run_ui_metadata || []};
    renderWorkspace();
  }
  const view = payload.decision_view;
  // Informational chat is not a new decision. Explicitly selected historical
  // results/scenarios survive until a new real workflow or result needs focus.
  if (!view || view.type === 'empty') {
    const current = state.currentView;
    const transient = current?.type === 'response_generation_issue' ||
      (current?.type === 'execution_failure' && current?.data?.error_code === 'agent_request_failed');
    if (!current || (transient && payload.presentation?.kind === 'informational')) renderDecision(null);
    return;
  }
  state.selectedRunKey = null;
  state.selectedScenarioKey = null;
  renderWorkspace();
  renderDecision(view);
}

async function sendMessage(message) {
  if (state.busy) return;
  appendMessage('user', message);
  setBusy(true);
  try {
    const payload = await api('/api/chat', {method: 'POST', body: JSON.stringify({message})});
    appendMessage('assistant', payload.message?.text || '', payload.developer_details, payload.presentation);
    renderWorkflowCards(payload.workflow_cards || []);
    applyAgentPayload(payload);
  } catch (error) {
    appendSystemError(error.message);
  } finally {
    setBusy(false);
  }
}

async function loadWorkspace() {
  const payload = await api('/api/workspace');
  state.workspace = payload;
  renderWorkspace();
  if (!state.currentView) renderDecision(null);
}

function connectionById(id) {
  return state.connections.profiles.find((p) => p.id === id) || null;
}

function updateConnectionChip() {
  // Active connection is persisted configuration/session selection, NOT the last
  // diagnostic result. A failed Test Connection must not change this indicator.
  const active = connectionById(state.connections.active_profile_id);
  const dot = $('connectionDot');
  dot.className = 'status-dot';
  if (!active) {
    $('connectionLabel').textContent = 'No active profile';
    return;
  }
  const ready = Boolean(active.api_key_configured && typeof active.model === 'string' && active.model.trim());
  $('connectionLabel').textContent = `${active.display_name} · ${ready ? 'Connected' : 'Active · Not verified'}`;
  dot.classList.add(ready ? 'connected' : 'configured');
}

function renderProfileList() {
  const list = $('profileList'); list.replaceChildren();
  state.connections.profiles.forEach((profile) => {
    const button = document.createElement('button'); button.type = 'button';
    button.className = `profile-item${profile.id === state.selectedProfileId ? ' selected' : ''}`;
    const title = document.createElement('div'); title.className = 'profile-item-title'; title.textContent = profile.display_name;
    const meta = document.createElement('div'); meta.className = 'profile-item-meta';
    const active = profile.id === state.connections.active_profile_id ? ' · active' : '';
    meta.textContent = `${profile.provider} · ${profile.connection_state}${active}`;
    button.append(title, meta); button.addEventListener('click', () => selectProfile(profile.id)); list.append(button);
  });
  if (!state.connections.profiles.length) list.innerHTML = '<div class="nav-empty">Create a connection profile to start.</div>';
}

function fillProfileForm(profile) {
  const active = Boolean(profile && profile.id === state.connections.active_profile_id);
  $('profileActiveStatus').textContent = !profile ? 'Not selected' : active ? 'Active profile' : 'Saved profile';
  $('profileActiveStatus').className = `profile-state${active ? ' active' : ''}`;
  $('profileName').value = profile?.display_name || '';
  $('profileProvider').value = profile?.provider || 'nvidia';
  $('profileBaseUrl').value = profile?.base_url || '';
  $('profileModel').value = profile?.model || '';
  $('apiKeyInput').value = '';
  $('keyState').textContent = profile?.api_key_configured ? 'Key configured — enter a new key to replace it' : 'No key configured';

  const selector = $('discoveredModelSelect');
  selector.replaceChildren();
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = 'Fetch models to populate this list';
  selector.append(placeholder);
  selector.disabled = true;

  const diagnostic = profile?.latest_diagnostic;
  if (diagnostic) {
    showConnectionResult(diagnostic);
  } else {
    $('testResult').textContent = profile?.api_key_configured ? 'Not tested' : 'API key required';
    $('testResult').className = 'test-result';
  }

  const noProfile = !profile;
  ['duplicateProfile','renameProfile','deleteProfile','saveKey','saveModel'].forEach((id) => $(id).disabled = noProfile);
  $('deleteKey').disabled = noProfile || !profile?.api_key_configured;
  $('fetchModels').disabled = noProfile || !profile?.api_key_configured;
  $('testConnection').disabled = noProfile || !profile?.api_key_configured;
  $('activateProfile').disabled = noProfile || !profile?.model;
}

function selectProfile(id) {
  state.selectedProfileId = id;
  renderProfileList();
  fillProfileForm(connectionById(id));
}

async function loadConnections(selectId = null) {
  state.connections = await api('/api/connections');
  updateConnectionChip();
  if (selectId && connectionById(selectId)) state.selectedProfileId = selectId;
  if (!state.selectedProfileId || !connectionById(state.selectedProfileId)) state.selectedProfileId = state.connections.active_profile_id || state.connections.profiles[0]?.id || null;
  renderProfileList();
  fillProfileForm(connectionById(state.selectedProfileId));
}

function profilePayload() {
  return {
    display_name: $('profileName').value.trim(),
    provider: $('profileProvider').value,
    base_url: $('profileBaseUrl').value.trim() || null,
    model: $('profileModel').value.trim(),
  };
}

async function saveCurrentProfile() {
  const current = connectionById(state.selectedProfileId);
  try {
    let profileId;
    if (current) {
      await api(`/api/connections/${encodeURIComponent(current.id)}`, {method:'PUT', body:JSON.stringify(profilePayload())});
      profileId = current.id;
    } else {
      const created = await api('/api/connections', {method:'POST', body:JSON.stringify(profilePayload())});
      profileId = created.profile.id;
    }
    await loadConnections(profileId);
    $('testResult').textContent = 'Profile saved.'; $('testResult').className = 'test-result good';
    return profileId;
  } catch (error) {
    $('testResult').textContent = error.message; $('testResult').className = 'test-result bad';
    return null;
  }
}

async function duplicateCurrent() {
  const current = connectionById(state.selectedProfileId); if (!current) return;
  try {
    const result = await api(`/api/connections/${encodeURIComponent(current.id)}/duplicate`, {method:'POST', body:'{}'});
    await loadConnections(result.profile.id);
  } catch (error) { $('testResult').textContent = error.message; $('testResult').className = 'test-result bad'; }
}

async function renameCurrent() {
  const current = connectionById(state.selectedProfileId); if (!current) return;
  const name = window.prompt('New profile name', current.display_name);
  if (!name?.trim()) return;
  try {
    await api(`/api/connections/${encodeURIComponent(current.id)}`, {method:'PUT', body:JSON.stringify({display_name:name.trim()})});
    await loadConnections(current.id);
  } catch (error) { $('testResult').textContent = error.message; $('testResult').className = 'test-result bad'; }
}

async function deleteCurrent() {
  const current = connectionById(state.selectedProfileId); if (!current) return;
  if (!window.confirm(`Delete connection profile “${current.display_name}”?`)) return;
  try { await api(`/api/connections/${encodeURIComponent(current.id)}`, {method:'DELETE'}); state.selectedProfileId = null; await loadConnections(); }
  catch (error) { $('testResult').textContent = error.message; $('testResult').className = 'test-result bad'; }
}

async function saveKey() {
  const current = connectionById(state.selectedProfileId); if (!current) return;
  const key = $('apiKeyInput').value;
  if (!key.trim()) { $('testResult').textContent = 'Enter a key first.'; $('testResult').className = 'test-result bad'; return; }
  try {
    await api(`/api/connections/${encodeURIComponent(current.id)}/key`, {method:'POST', body:JSON.stringify({api_key:key})});
    $('apiKeyInput').value = '';
    await loadConnections(current.id);
    $('testResult').textContent = 'Key saved.'; $('testResult').className = 'test-result good';
  } catch (error) { $('testResult').textContent = error.message; $('testResult').className = 'test-result bad'; }
}

async function deleteKey() {
  const current = connectionById(state.selectedProfileId); if (!current) return;
  try { await api(`/api/connections/${encodeURIComponent(current.id)}/key`, {method:'DELETE'}); await loadConnections(current.id); $('testResult').textContent = 'Key deleted.'; }
  catch (error) { $('testResult').textContent = error.message; $('testResult').className = 'test-result bad'; }
}

async function saveModel() {
  const current = connectionById(state.selectedProfileId); if (!current) return;
  try {
    await api(`/api/connections/${encodeURIComponent(current.id)}`, {
      method:'PUT',
      body:JSON.stringify({model:$('profileModel').value.trim()}),
    });
    await loadConnections(current.id);
    $('testResult').textContent = $('profileModel').value.trim() ? 'Model saved.' : 'Model selection cleared.';
    $('testResult').className = 'test-result good';
  } catch (error) {
    $('testResult').textContent = error.message; $('testResult').className = 'test-result bad';
  }
}

function showConnectionResult(result) {
  // Present diagnostic evidence without mutating active profile/session state.
  const code = result?.code;
  const modelTested = result?.inference_verified === true;
  let text;
  let appearance = 'bad';
  if (modelTested) {
    text = 'Inference verified — selected model responded successfully.';
    appearance = 'good';
  } else if (code === 'model_test_successful') {
    text = 'Model metadata verified; inference not tested.';
    appearance = 'good';
  } else if (code === 'connected_authenticated') {
    text = 'Endpoint reachable · Authentication verified · Model discovery verified. Inference not tested.';
    appearance = 'good';
  } else if (code === 'model_discovery_unavailable') {
    text = 'Endpoint reachable · Model discovery unavailable · Authentication not verified. Try a manual Model ID.';
    appearance = 'warn';
  } else if (code === 'authentication_failed') {
    text = 'Authentication failed. Check the saved API key and permissions.';
  } else if (code === 'model_unavailable') {
    text = 'Selected model unavailable. Check its identifier or account access.';
  } else if (code === 'inference_test_failed') {
    text = 'Inference test failed. This does not necessarily mean the endpoint is unreachable.';
  } else if (code === 'endpoint_failure') {
    text = result?.connected === true ? 'Endpoint returned an error.' : 'Endpoint unreachable.';
  } else {
    text = result?.message || 'Connection diagnostic failed.';
    appearance = result?.ok ? 'good' : 'bad';
  }
  $('testResult').textContent = text;
  $('testResult').className = `test-result ${appearance}`;
}

const connectionPending = new Set();
async function withConnectionBusy(buttonId, label, operation) {
  if (connectionPending.has(buttonId)) return;
  connectionPending.add(buttonId);
  const button = $(buttonId), text = button.textContent;
  button.disabled = true; button.classList.add('control-busy'); button.textContent = label;
  try { return await operation(); }
  finally {
    connectionPending.delete(buttonId);
    button.textContent = text; button.classList.remove('control-busy');
    // Respect configuration preconditions after the operation, too.
    const profile = connectionById(state.selectedProfileId);
    button.disabled = !profile || (buttonId !== 'activateProfile' && !profile.api_key_configured) || (buttonId === 'activateProfile' && !profile.model);
  }
}
async function activateCurrent() {
  const current = connectionById(state.selectedProfileId); if (!current) return;
  return withConnectionBusy('activateProfile', 'Activating…', async () => {
    try { await api(`/api/connections/${encodeURIComponent(current.id)}/activate`, {method:'POST', body:'{}'}); await loadConnections(current.id); }
    catch (error) { $('testResult').textContent = error.message; $('testResult').className = 'test-result bad'; }
  });
}

async function testCurrentConnection() {
  const current = connectionById(state.selectedProfileId); if (!current) return;
  return withConnectionBusy('testConnection', 'Testing…', () => doTestCurrentConnection(current));
}
async function doTestCurrentConnection(current) {
  $('testResult').textContent = 'Testing…'; $('testResult').className = 'test-result';
  try {
    const result = await api(`/api/connections/${encodeURIComponent(current.id)}/test`, {method:'POST', body:'{}'});
    await loadConnections(current.id);
    showConnectionResult(result);
  } catch (error) {
    const result = error.payload || {code:'provider_error', message:error.message, ok:false};
    await loadConnections(current.id);
    showConnectionResult(result);
  }
}

async function fetchModels() {
  const current = connectionById(state.selectedProfileId); if (!current) return;
  return withConnectionBusy('fetchModels', 'Fetching…', () => doFetchModels(current));
}
async function doFetchModels(current) {
  $('testResult').textContent = 'Fetching models…'; $('testResult').className = 'test-result';
  try {
    const result = await api(`/api/connections/${encodeURIComponent(current.id)}/models`);
    await loadConnections(current.id);
    const selector = $('discoveredModelSelect');
    selector.replaceChildren();
    const placeholder = document.createElement('option');
    placeholder.value = '';
    const models = result.models || [];
    placeholder.textContent = models.length ? 'Select a discovered model' : 'No discovered models';
    selector.append(placeholder);
    models.forEach((id) => { const option = document.createElement('option'); option.value = id; option.textContent = id; selector.append(option); });
    selector.disabled = models.length === 0;
    if (current.model && models.includes(current.model)) selector.value = current.model;
    if (result.code === 'model_discovery_unavailable') {
      showConnectionResult(result);
    } else if (!models.length) {
      $('testResult').textContent = 'Connected/authenticated, but no models were returned. Enter a Model ID manually.';
      $('testResult').className = 'test-result warn';
    } else {
      $('testResult').textContent = `${models.length} models available — authentication and discovery verified; inference not tested. Select a model below.`;
      $('testResult').className = 'test-result good';
    }
  } catch (error) {
    const result = error.payload || {code:'provider_error', message:error.message, ok:false};
    await loadConnections(current.id);
    showConnectionResult(result);
  }
}

async function chooseDiscoveredModel() {
  const selected = $('discoveredModelSelect').value;
  if (!selected) return;
  $('profileModel').value = selected;
  await saveModel();
}

let guideReturnFocus = null;
function openGuide() {
  guideReturnFocus = document.activeElement;
  const backdrop = $('guideBackdrop');
  backdrop.classList.remove('hidden'); backdrop.setAttribute('aria-hidden', 'false');
  $('closeGuide').focus();
}
function closeGuide() {
  const backdrop = $('guideBackdrop');
  backdrop.classList.add('hidden'); backdrop.setAttribute('aria-hidden', 'true');
  if (guideReturnFocus?.isConnected) guideReturnFocus.focus();
  guideReturnFocus = null;
}
function openSettings() { $('settingsBackdrop').classList.remove('hidden'); $('settingsBackdrop').setAttribute('aria-hidden','false'); loadConnections(); }
function closeSettings() { $('settingsBackdrop').classList.add('hidden'); $('settingsBackdrop').setAttribute('aria-hidden','true'); }

$('composer').addEventListener('submit', (event) => {
  event.preventDefault();
  const message = $('messageInput').value.trim();
  if (!message || state.busy) return;
  $('messageInput').value = '';
  sendMessage(message);
});
$('messageInput').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    if (state.busy) return;
    $('composer').requestSubmit();
  }
});
$('toggleArchivedRuns').addEventListener('click', () => {
  state.showArchivedRuns = !state.showArchivedRuns;
  renderWorkspace();
});
$('headerProblemSelect').addEventListener('change', () => {
  $('switchProblemButton').disabled = state.switchBusy || !$('headerProblemSelect').value || $('headerProblemSelect').value === activeProblem()?.problem_id;
});
$('switchProblemButton').addEventListener('click', () => {
  const next = $('headerProblemSelect').value;
  if (next && next !== activeProblem()?.problem_id) selectProblem(next);
});
$('guideButton').addEventListener('click', openGuide);
$('closeGuide').addEventListener('click', closeGuide);
$('guideBackdrop').addEventListener('click', event => { if (event.target === $('guideBackdrop')) closeGuide(); });
document.addEventListener('keydown', event => {
  if ($('guideBackdrop').classList.contains('hidden')) return;
  if (event.key === 'Escape') { event.preventDefault(); closeGuide(); return; }
  if (event.key === 'Tab') {
    const controls = Array.from($('guideBackdrop').querySelectorAll('button:not(:disabled),a[href]'));
    const first = controls[0], last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
  }
});
$('settingsButton').addEventListener('click', openSettings);
$('connectionChip').addEventListener('click', openSettings);
$('closeSettings').addEventListener('click', closeSettings);
$('settingsBackdrop').addEventListener('click', (event) => { if (event.target === $('settingsBackdrop')) closeSettings(); });
$('newProfile').addEventListener('click', () => { state.selectedProfileId = null; renderProfileList(); fillProfileForm(null); $('profileName').focus(); });
$('saveProfile').addEventListener('click', saveCurrentProfile);
$('duplicateProfile').addEventListener('click', duplicateCurrent);
$('renameProfile').addEventListener('click', renameCurrent);
$('deleteProfile').addEventListener('click', deleteCurrent);
$('saveKey').addEventListener('click', saveKey);
$('saveModel').addEventListener('click', saveModel);
$('discoveredModelSelect').addEventListener('change', chooseDiscoveredModel);
$('deleteKey').addEventListener('click', deleteKey);
$('activateProfile').addEventListener('click', activateCurrent);
$('testConnection').addEventListener('click', testCurrentConnection);
$('fetchModels').addEventListener('click', fetchModels);

Promise.all([loadWorkspace(), loadConnections()]).catch((error) => appendSystemError(error.message));
