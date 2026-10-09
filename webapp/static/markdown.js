/* UX-2: tiny offline Markdown subset, constructed exclusively via DOM APIs.
 * No innerHTML, arbitrary HTML, URL interpolation, or third-party network code.
 * Unsupported Markdown remains visible text rather than being interpreted as HTML.
 */
(function () {
  'use strict';
  const safeHref = (url) => {
    try {
      const parsed = new URL(url, 'https://invalid.local/');
      if (!/^(https?:|mailto:)$/.test(parsed.protocol)) return null;
      // Relative links are not meaningful for the local agent; deny implicit origin links.
      if (!/^(https?:\/\/|mailto:)/i.test(url)) return null;
      return url;
    } catch { return null; }
  };
  const el = (tag, className) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    return node;
  };
  function inline(parent, text) {
    const source = String(text);
    let pos = 0;
    // Strict allowlist of inline syntax; content always written as text nodes.
    const token = /(`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_|\[[^\]\n]+\]\([^()\n]*\))/g;
    for (const match of source.matchAll(token)) {
      const index = match.index;
      if (index > pos) parent.append(document.createTextNode(source.slice(pos, index)));
      const raw = match[0];
      if (raw.startsWith('`')) {
        const node = el('code'); node.textContent = raw.slice(1, -1); parent.append(node);
      } else if (raw.startsWith('**') || raw.startsWith('__')) {
        const node = el('strong'); inline(node, raw.slice(2, -2)); parent.append(node);
      } else if (raw.startsWith('*') || raw.startsWith('_')) {
        const node = el('em'); inline(node, raw.slice(1, -1)); parent.append(node);
      } else {
        const parts = /^\[([^\]]+)\]\(([^()]*)\)$/.exec(raw);
        const href = parts && safeHref(parts[2].trim());
        if (href) {
          const node = el('a'); node.href = href; node.target = '_blank';
          node.rel = 'noopener noreferrer'; node.textContent = parts[1]; parent.append(node);
        } else parent.append(document.createTextNode(parts ? parts[1] : raw));
      }
      pos = index + raw.length;
    }
    if (pos < source.length) parent.append(document.createTextNode(source.slice(pos)));
  }
  const cells = (line) => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(x => x.trim());
  const tableRule = (line) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
  function renderSafeMarkdown(markdown) {
    const root = el('div', 'markdown-body');
    const lines = String(markdown || '').replace(/\r\n?/g, '\n').split('\n');
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }
      const fence = /^\s*(`{3,}|~{3,})([\w-]*)\s*$/.exec(line);
      if (fence) {
        i++;
        const code = [];
        const end = new RegExp('^\\s*' + fence[1][0] + '{' + fence[1].length + ',}\\s*$');
        while (i < lines.length && !end.test(lines[i])) code.push(lines[i++]);
        if (i < lines.length) i++;
        const pre = el('pre'); const c = el('code'); c.textContent = code.join('\n');
        pre.append(c); root.append(pre); continue;
      }
      const heading = /^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
      if (heading) {
        const h = el('h' + heading[1].length); inline(h, heading[2]); root.append(h); i++; continue;
      }
      if (i + 1 < lines.length && line.includes('|') && tableRule(lines[i + 1])) {
        const cols = cells(line); const table = el('table'); const thead = el('thead'); const tr = el('tr');
        cols.forEach(cell => { const th = el('th'); inline(th, cell); tr.append(th); });
        thead.append(tr); table.append(thead); i += 2; const tbody = el('tbody');
        while (i < lines.length && lines[i].trim() && lines[i].includes('|')) {
          const row = el('tr'); cells(lines[i++]).forEach(cell => { const td = el('td'); inline(td, cell); row.append(td); }); tbody.append(row);
        }
        table.append(tbody); root.append(table); continue;
      }
      if (/^\s{0,3}>\s?/.test(line)) {
        const quote = [];
        while (i < lines.length && /^\s{0,3}>\s?/.test(lines[i])) quote.push(lines[i++].replace(/^\s{0,3}>\s?/, ''));
        const block = el('blockquote'); const p = el('p'); inline(p, quote.join('\n')); block.append(p); root.append(block); continue;
      }
      const list = /^\s{0,3}([-+*]|\d+[.)])\s+(.+)$/.exec(line);
      if (list) {
        const ordered = /^\d/.test(list[1]); const listNode = el(ordered ? 'ol' : 'ul');
        while (i < lines.length) {
          const m = /^\s{0,3}([-+*]|\d+[.)])\s+(.+)$/.exec(lines[i]);
          if (!m || /^\d/.test(m[1]) !== ordered) break;
          const li = el('li'); inline(li, m[2]); listNode.append(li); i++;
        }
        root.append(listNode); continue;
      }
      const paragraph = [line]; i++;
      while (i < lines.length && lines[i].trim() && !/^\s{0,3}(#{1,6}\s|>|[-+*]\s|\d+[.)]\s|`{3}|~{3})/.test(lines[i]) &&
          !(i + 1 < lines.length && lines[i].includes('|') && tableRule(lines[i + 1]))) paragraph.push(lines[i++]);
      const p = el('p'); inline(p, paragraph.join('\n')); root.append(p);
    }
    return root;
  }
  window.renderSafeMarkdown = renderSafeMarkdown;
}());
