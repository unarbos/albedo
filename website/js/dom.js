export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k === "title") node.title = v;
    else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else {
      node.setAttribute(k, v);
    }
  }
  append(node, children);
  return node;
}

function append(node, children) {
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
}

export function mount(node, ...children) {
  clear(node);
  append(node, children);
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

export function link(href, text, props = {}) {
  return el("a", { href, target: "_blank", rel: "noopener", ...props }, text);
}
