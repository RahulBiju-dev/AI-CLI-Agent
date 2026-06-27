/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * Selene — AI Agent Interface
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * A complete vanilla JS application for a local AI agent chat interface.
 * Handles SSE streaming, starfield canvas, slash commands, sidebar management,
 * settings drawer, session management, token counting, message rendering with
 * markdown + syntax highlighting, thinking panels, and tool invocation cards.
 *
 * Libraries (loaded via CDN in HTML):
 *   - marked     — Markdown rendering
 *   - hljs       — Syntax highlighting
 *   - DOMPurify  — HTML sanitization
 *
 * @file app.js
 * @version 2.0.0
 */

/* ═══════════════════════════════════════════════════════════════════════════════
 * §1  CONFIGURATION & STATE
 * ═══════════════════════════════════════════════════════════════════════════════ */

/** @type {Object} Application-wide configuration constants */
const CONFIG = {
  SSE_ENDPOINT: '/api/chat',
  SAVE_ENDPOINT: '/api/sessions',
  SETTINGS_ENDPOINT: '/api/settings',
  MAX_TOKENS: 8192,
  STORAGE_KEYS: {
    SETTINGS: 'selene_settings',
    CONVERSATIONS: 'selene_conversations',
    SIDEBAR: 'selene_sidebar_collapsed',
    ACTIVE_CONVERSATION: 'selene_active_conversation'
  },
  DEFAULT_SETTINGS: {
    temperature: 0.7,
    topP: 1.0,
    topK: 40,
    numCtx: 8192,
    systemPrompt: 'You are Selene, a helpful and precise AI assistant.',
    showThinking: true,
    streamTokens: true,
    verboseMode: false,
    saveHistory: true
  }
};

/**
 * Reactive application state.
 * All UI reads from and writes to this single state object.
 * @type {Object}
 */
const state = {
  /** @type {Array<{id:string, title:string, messages:Array, tokenCount:number, timestamp:number}>} */
  conversations: [],
  /** @type {string|null} */
  activeConversationId: null,
  /** @type {number} Current token usage in the active conversation */
  currentTokens: 0,
  /** @type {number} Maximum token budget */
  maxTokens: CONFIG.MAX_TOKENS,
  /** @type {Object} Active settings (merged from defaults + localStorage) */
  settings: { ...CONFIG.DEFAULT_SETTINGS },
  /** @type {boolean} Whether an SSE stream is currently in progress */
  isStreaming: false,
  /** @type {AbortController|null} Controller for cancelling active fetch streams */
  abortController: null,
  /** @type {boolean} Whether the sidebar is collapsed */
  sidebarCollapsed: false
};


/* ═══════════════════════════════════════════════════════════════════════════════
 * §2  DOM REFERENCES
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Cached DOM element references.
 * Populated once at initialization to avoid repeated querySelector calls.
 * @type {Object}
 */
let dom = {};

/**
 * Cache all DOM references used throughout the application.
 * Must be called after DOMContentLoaded.
 */
function cacheDOMReferences() {
  dom = {
    // Layout
    sidebar: document.getElementById('sidebar'),
    sidebarToggle: document.getElementById('rail-toggle'),
    sidebarContent: document.querySelector('.sidebar-content'),
    newChatBtn: document.querySelector('.new-chat-btn'),
    sidebarSearch: document.querySelector('.search-input'),
    conversationList: document.getElementById('session-list'),
    chatViewport: document.querySelector('.chat-viewport'),

    // Canvas
    starfieldCanvas: document.getElementById('starfield'),

    // Messages
    messagesContainer: document.getElementById('messages-container'),
    messageInput: document.getElementById('message-input'),
    sendBtn: document.getElementById('send-btn'),

    // Slash menu
    slashMenu: document.getElementById('slash-menu'),

    // Settings
    settingsDrawer: document.getElementById('settings-drawer'),
    settingsScrim: document.getElementById('scrim'),
    settingsToggle: document.getElementById('settings-toggle'),
    closeDrawerBtn: document.getElementById('close-drawer'),

    // Token counter
    tokenCountText: document.querySelector('.token-text'),
    tokenRingProgress: document.querySelector('.ring-progress'),

    // Settings controls — sliders
    settingTemperature: document.getElementById('temp-slider'),
    settingTopP: document.getElementById('topp-slider'),
    settingNumCtx: document.getElementById('ctx-slider'),

    // Settings controls — readouts
    readoutTemperature: document.getElementById('temp-val'),
    readoutTopP: document.getElementById('topp-val'),
    readoutNumCtx: document.getElementById('ctx-val'),

    // Settings controls — system prompt
    settingSystemPrompt: document.querySelector('.system-prompt-input'),

    // Settings controls — toggles
    toggleThinking: null,  // Resolved by index in behaviors section
    toggleStream: null,
    toggleVerbose: null,

    // Session buttons
    sessionButtons: document.querySelectorAll('.btn-group .secondary-btn'),

    // Templates
    tplUserMsg: document.getElementById('tpl-user-msg'),
    tplAssistantMsg: document.getElementById('tpl-assistant-msg'),
    tplToolCard: document.getElementById('tpl-tool-card')
  };

  // Resolve toggle checkboxes by their position in the Behaviors section
  const toggleLabels = document.querySelectorAll('.toggle-setting');
  if (toggleLabels.length >= 1) dom.toggleThinking = toggleLabels[0].querySelector('input[type="checkbox"]');
  if (toggleLabels.length >= 2) dom.toggleStream = toggleLabels[1].querySelector('input[type="checkbox"]');
  if (toggleLabels.length >= 3) dom.toggleVerbose = toggleLabels[2].querySelector('input[type="checkbox"]');
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §3  UTILITY FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Escape HTML entities to prevent XSS in user-generated content.
 * @param {string} str — Raw string
 * @returns {string} Escaped string safe for innerHTML insertion
 */
function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

/**
 * Format a number with comma separators.
 * @param {number} n — Number to format
 * @returns {string} Formatted string (e.g. 4872 → '4,872')
 */
function formatNumber(n) {
  return Number(n).toLocaleString('en-US');
}

/**
 * Convert a timestamp to a human-readable relative time string.
 * @param {number} timestamp — Unix timestamp in milliseconds
 * @returns {string} Relative time (e.g. '2m ago', '1h ago', 'Just now')
 */
function relativeTime(timestamp) {
  const now = Date.now();
  const diff = now - timestamp;
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (seconds < 10) return 'Just now';
  if (seconds < 60) return `${seconds}s ago`;
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 30) return `${days}d ago`;
  return new Date(timestamp).toLocaleDateString();
}

/**
 * Generate a unique ID string.
 * @returns {string} UUID-like identifier
 */
function generateId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * Create a debounced version of a function.
 * @param {Function} fn — Function to debounce
 * @param {number} ms — Delay in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(fn, ms) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}

/**
 * Create a throttled version of a function.
 * Uses trailing edge + leading edge for responsiveness.
 * @param {Function} fn — Function to throttle
 * @param {number} ms — Minimum interval in milliseconds
 * @returns {Function} Throttled function
 */
function throttle(fn, ms) {
  let lastCall = 0;
  let timer = null;
  return function (...args) {
    const now = Date.now();
    const remaining = ms - (now - lastCall);
    clearTimeout(timer);
    if (remaining <= 0) {
      lastCall = now;
      fn.apply(this, args);
    } else {
      timer = setTimeout(() => {
        lastCall = Date.now();
        fn.apply(this, args);
      }, remaining);
    }
  };
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §4  STARFIELD CANVAS  (Canvas 2D)
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Initialize the ambient starfield background effect.
 *
 * Creates 80-120 stars with subtle slow drift behind the message scroll area.
 * Uses requestAnimationFrame with delta-time for framerate-independent motion.
 * Stars wrap around viewport edges. Canvas resizes on window resize (debounced).
 * Capped at 120 particles to never cause jank.
 */
function initStarfield() {
  const canvas = dom.starfieldCanvas;
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let width = 0;
  let height = 0;

  /** @type {Array<{x:number,y:number,size:number,opacity:number,speedX:number,speedY:number}>} */
  let stars = [];

  const STAR_COUNT = Math.floor(Math.random() * 41) + 80; // 80–120

  /**
   * Create a single star with randomized properties.
   * @returns {Object} Star descriptor
   */
  function createStar() {
    return {
      x: Math.random() * width,
      y: Math.random() * height,
      size: Math.random() * 1.5 + 0.5,       // 0.5–2px
      opacity: Math.random() * 0.10 + 0.15,    // 0.15–0.25
      speedX: (Math.random() - 0.5) * 0.15,    // -0.075 to 0.075
      speedY: (Math.random() - 0.5) * 0.15
    };
  }

  /**
   * Resize canvas to match parent container, accounting for devicePixelRatio.
   */
  function resizeCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }

  /**
   * Populate the stars array.
   */
  function populateStars() {
    stars = [];
    for (let i = 0; i < STAR_COUNT; i++) {
      stars.push(createStar());
    }
  }

  // Debounced resize handler
  const debouncedResize = debounce(() => {
    resizeCanvas();
    // Reposition stars that might be out of bounds after resize
    stars.forEach(star => {
      if (star.x > width) star.x = Math.random() * width;
      if (star.y > height) star.y = Math.random() * height;
    });
  }, 150);

  window.addEventListener('resize', debouncedResize);

  // Initial setup
  resizeCanvas();
  populateStars();

  let lastTimestamp = 0;

  /**
   * Main render loop — clears canvas and redraws all stars.
   * @param {DOMHighResTimeStamp} timestamp — Current animation frame time
   */
  function renderLoop(timestamp) {
    if (!lastTimestamp) {
      lastTimestamp = timestamp;
    }
    const dt = timestamp - lastTimestamp;
    lastTimestamp = timestamp;

    // Normalize to ~16ms frame (60fps baseline)
    const dtFactor = dt / 16;

    ctx.clearRect(0, 0, width, height);

    for (let i = 0; i < stars.length; i++) {
      const star = stars[i];

      // Update position with delta-time normalization
      star.x += star.speedX * dtFactor;
      star.y += star.speedY * dtFactor;

      // Wrap around edges
      if (star.x < -2) star.x = width + 1;
      if (star.x > width + 2) star.x = -1;
      if (star.y < -2) star.y = height + 1;
      if (star.y > height + 2) star.y = -1;

      // Draw star as a filled circle
      ctx.fillStyle = `rgba(232, 230, 227, ${star.opacity})`;
      ctx.beginPath();
      ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
      ctx.fill();
    }

    requestAnimationFrame(renderLoop);
  }

  requestAnimationFrame(renderLoop);
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §5  TOKEN COUNTER
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Update the token counter display and SVG progress ring.
 *
 * @param {number} current — Current token usage
 * @param {number} [max] — Maximum token budget (defaults to state.maxTokens)
 */
function updateTokenCounter(current, max) {
  max = max || state.maxTokens;
  state.currentTokens = current;
  state.maxTokens = max;

  // Update text label: '4,872 / 8,192'
  if (dom.tokenCountText) {
    dom.tokenCountText.textContent = `${formatNumber(current)} / ${formatNumber(max)}`;
  }

  // Update SVG progress ring via stroke-dashoffset
  if (dom.tokenRingProgress) {
    const radius = 10; // r attribute on the SVG circle
    const circumference = 2 * Math.PI * radius; // ≈ 62.83
    const percentage = Math.min(current / max, 1);
    const offset = circumference - (percentage * circumference);
    dom.tokenRingProgress.style.strokeDashoffset = offset;
  }
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §6  SIDEBAR MANAGEMENT
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Toggle sidebar collapse/expand state.
 * Stores preference in localStorage.
 */
function toggleSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  dom.sidebar.classList.toggle('collapsed', state.sidebarCollapsed);
  dom.sidebar.setAttribute('aria-expanded', !state.sidebarCollapsed);

  try {
    localStorage.setItem(CONFIG.STORAGE_KEYS.SIDEBAR, JSON.stringify(state.sidebarCollapsed));
  } catch (e) { /* storage full or unavailable */ }
}

/**
 * Restore sidebar collapsed state from localStorage.
 */
function restoreSidebarState() {
  try {
    const stored = localStorage.getItem(CONFIG.STORAGE_KEYS.SIDEBAR);
    if (stored !== null) {
      state.sidebarCollapsed = JSON.parse(stored);
      dom.sidebar.classList.toggle('collapsed', state.sidebarCollapsed);
    }
  } catch (e) { /* ignore parse errors */ }
}

/**
 * Create a new conversation, clearing the chat viewport.
 */
function createNewConversation() {
  const conversation = {
    id: generateId(),
    title: 'New Chat',
    messages: [],
    tokenCount: 0,
    timestamp: Date.now()
  };

  state.conversations.unshift(conversation);
  state.activeConversationId = conversation.id;
  state.currentTokens = 0;

  // Clear viewport
  clearChatViewport();

  // Re-render sidebar list
  renderConversationList();
  updateTokenCounter(0);

  // Auto-save
  saveConversationsToStorage();

  // Focus input
  dom.messageInput.focus();
}

/**
 * Clear the chat viewport and show the welcome message.
 */
function clearChatViewport() {
  dom.messagesContainer.innerHTML = `
    <div class="message-group">
      <div class="message assistant-msg welcome-msg">
        <div class="msg-content">Hi, I'm Selene. Type <kbd>/</kbd> for commands or start chatting.</div>
      </div>
    </div>`;
}

/**
 * Render the conversation list in the sidebar from state.
 * @param {string} [filterText=''] — Optional search filter
 */
function renderConversationList(filterText = '') {
  if (!dom.conversationList) return;

  const filtered = state.conversations.filter(c =>
    c.title.toLowerCase().includes(filterText.toLowerCase())
  );

  if (filtered.length === 0) {
    dom.conversationList.innerHTML = filterText
      ? '<div class="session-meta" style="padding:12px;text-align:center;">No matches</div>'
      : '';
    return;
  }

  dom.conversationList.innerHTML = filtered.map(conv => `
    <div class="session-card ${conv.id === state.activeConversationId ? 'active' : ''}"
         data-conversation-id="${conv.id}"
         role="button"
         tabindex="0"
         aria-label="Load conversation: ${escapeHtml(conv.title)}">
      <div class="session-title">${escapeHtml(conv.title)}</div>
      <div class="session-meta">${relativeTime(conv.timestamp)} · ${formatNumber(conv.tokenCount)} tokens</div>
    </div>
  `).join('');

  // Attach click handlers via delegation + apply tilt
  dom.conversationList.querySelectorAll('.session-card').forEach(card => {
    card.addEventListener('click', () => {
      const id = card.dataset.conversationId;
      loadConversation(id);
    });
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        const id = card.dataset.conversationId;
        loadConversation(id);
      }
    });
    applyTilt(card);
  });
}

/**
 * Load a conversation into the chat viewport by its ID.
 * @param {string} id — Conversation ID
 */
function loadConversation(id) {
  const conv = state.conversations.find(c => c.id === id);
  if (!conv) return;

  state.activeConversationId = id;
  state.currentTokens = conv.tokenCount || 0;

  // Clear viewport
  dom.messagesContainer.innerHTML = '<div class="message-group"></div>';

  // Remove welcome message
  const welcome = dom.messagesContainer.querySelector('.welcome-msg');
  if (welcome) welcome.remove();

  // Replay messages
  const group = dom.messagesContainer.querySelector('.message-group') || dom.messagesContainer;
  conv.messages.forEach(msg => {
    if (msg.role === 'user') {
      const node = dom.tplUserMsg.content.cloneNode(true).querySelector('.message');
      node.querySelector('.msg-content').textContent = msg.content;
      group.appendChild(node);
    } else if (msg.role === 'assistant') {
      const node = dom.tplAssistantMsg.content.cloneNode(true).querySelector('.message');

      // Thinking content
      if (msg.thinking) {
        const panel = node.querySelector('.thinking-panel');
        panel.style.display = 'block';
        node.querySelector('.thinking-content').textContent = msg.thinking;
        const toggle = node.querySelector('.thinking-toggle');
        toggle.addEventListener('click', () => {
          panel.classList.toggle('collapsed');
          panel.classList.toggle('expanded');
        });
      }

      // Render markdown content
      const contentEl = node.querySelector('.msg-content');
      contentEl.innerHTML = DOMPurify.sanitize(marked.parse(msg.content || ''));

      // Apply syntax highlighting to code blocks
      contentEl.querySelectorAll('pre code').forEach(block => {
        hljs.highlightElement(block);
      });

      // Setup copy buttons
      setupCopyButtons(contentEl);

      group.appendChild(node);

      // Tool cards
      if (msg.tools && msg.tools.length) {
        msg.tools.forEach(tool => {
          const toolNode = createToolCard(tool.name, tool.id);
          if (tool.status === 'completed') {
            finalizeToolCard(toolNode, 'completed', tool.output);
          } else if (tool.status === 'error') {
            finalizeToolCard(toolNode, 'error', tool.output);
          }
          group.appendChild(toolNode);
        });
      }
    }
  });

  dom.messagesContainer.scrollTop = dom.messagesContainer.scrollHeight;

  // Update token counter
  updateTokenCounter(state.currentTokens);

  // Update sidebar highlighting
  renderConversationList(dom.sidebarSearch ? dom.sidebarSearch.value : '');

  // Persist active conversation
  try {
    localStorage.setItem(CONFIG.STORAGE_KEYS.ACTIVE_CONVERSATION, id);
  } catch (e) { /* ignore */ }
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §7  MESSAGE RENDERING
 * ═══════════════════════════════════════════════════════════════════════════════ */

/** @type {HTMLElement|null} Currently active assistant message element during streaming */
let activeAssistantMsgEl = null;

/** @type {string} Accumulator for streaming markdown content */
let activeMarkdownBuffer = '';

/**
 * Append a user message to the chat viewport.
 * Escapes HTML, right-aligned bubble, plain text only.
 * @param {string} text — User's message text
 */
function appendUserMessage(text) {
  // Remove welcome message on first chat
  const welcome = dom.messagesContainer.querySelector('.welcome-msg');
  if (welcome) welcome.closest('.message-group')?.remove() || welcome.remove();

  // Ensure a message group container exists
  let group = dom.messagesContainer.querySelector('.message-group');
  if (!group) {
    group = document.createElement('div');
    group.className = 'message-group';
    dom.messagesContainer.appendChild(group);
  }

  const node = dom.tplUserMsg.content.cloneNode(true).querySelector('.message');
  node.querySelector('.msg-content').textContent = text; // textContent auto-escapes
  group.appendChild(node);
  dom.messagesContainer.scrollTop = dom.messagesContainer.scrollHeight;

  // Store in active conversation
  const conv = getActiveConversation();
  if (conv) {
    conv.messages.push({ role: 'user', content: text });
    conv.timestamp = Date.now();
    // Auto-generate title from first user message
    if (conv.title === 'New Chat' && conv.messages.length === 1) {
      conv.title = text.substring(0, 50) + (text.length > 50 ? '…' : '');
      renderConversationList(dom.sidebarSearch ? dom.sidebarSearch.value : '');
    }
    saveConversationsToStorage();
  }
}

/**
 * Create a new assistant message bubble and prepare it for streaming tokens.
 * @returns {HTMLElement} The assistant message DOM element
 */
function createAssistantMessage() {
  let group = dom.messagesContainer.querySelector('.message-group');
  if (!group) {
    group = document.createElement('div');
    group.className = 'message-group';
    dom.messagesContainer.appendChild(group);
  }

  const node = dom.tplAssistantMsg.content.cloneNode(true).querySelector('.message');

  // Setup thinking toggle
  const toggle = node.querySelector('.thinking-toggle');
  const panel = node.querySelector('.thinking-panel');
  if (toggle && panel) {
    toggle.setAttribute('aria-expanded', 'false');
    toggle.addEventListener('click', () => {
      const isExpanded = panel.classList.contains('expanded');
      panel.classList.toggle('collapsed', isExpanded);
      panel.classList.toggle('expanded', !isExpanded);
      toggle.setAttribute('aria-expanded', !isExpanded);
    });
  }

  group.appendChild(node);
  dom.messagesContainer.scrollTop = dom.messagesContainer.scrollHeight;

  activeAssistantMsgEl = node;
  activeMarkdownBuffer = '';

  return node;
}

/**
 * Append thinking content to the active assistant message's thinking panel.
 * @param {string} text — Thinking text chunk
 */
function appendThinkingContent(text) {
  if (!activeAssistantMsgEl) return;

  const panel = activeAssistantMsgEl.querySelector('.thinking-panel');
  const content = activeAssistantMsgEl.querySelector('.thinking-content');
  if (!panel || !content) return;

  // Show the panel
  panel.style.display = 'block';

  // Append text with line breaks
  content.textContent += text;
}

/**
 * Finalize the active assistant message after streaming completes.
 * Re-renders the full markdown, applies highlighting, stores in conversation.
 * @param {string} [finalContent] — Optional final content override
 */
function finalizeAssistantMessage(finalContent) {
  if (!activeAssistantMsgEl) return;

  const content = finalContent || activeMarkdownBuffer;
  const contentDiv = activeAssistantMsgEl.querySelector('.msg-content');

  if (contentDiv) {
    // Final parse of complete markdown
    contentDiv.innerHTML = DOMPurify.sanitize(marked.parse(content));

    // Apply syntax highlighting
    contentDiv.querySelectorAll('pre code').forEach(block => {
      hljs.highlightElement(block);
    });

    // Setup copy buttons
    setupCopyButtons(contentDiv);
  }

  // Store in active conversation
  const conv = getActiveConversation();
  if (conv) {
    const thinkingContent = activeAssistantMsgEl.querySelector('.thinking-content');
    conv.messages.push({
      role: 'assistant',
      content: content,
      thinking: thinkingContent ? thinkingContent.textContent : undefined
    });
    conv.timestamp = Date.now();
    saveConversationsToStorage();
  }

  activeAssistantMsgEl = null;
  activeMarkdownBuffer = '';

  // Announce to screen readers
  announceToScreenReader('Response complete.');
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §8  SSE STREAMING & RENDER QUEUE (RAF-batched)
 * ═══════════════════════════════════════════════════════════════════════════════ */

/** @type {string} Buffer for tokens waiting to be flushed to DOM */
let tokenBuffer = '';

/** @type {number|null} requestAnimationFrame ID for token flushing */
let rafId = null;

/**
 * Queue a token for batched DOM update via requestAnimationFrame.
 * Collects tokens in a buffer and flushes them on the next animation frame.
 * @param {string} text — Token text to append
 */
function queueToken(text) {
  tokenBuffer += text;
  if (!rafId) {
    rafId = requestAnimationFrame(flushTokens);
  }
}

/**
 * Flush all buffered tokens to the active assistant message DOM element.
 * Parses the full accumulated markdown buffer and replaces innerHTML.
 * Adds `.token-new` class for materialize animation.
 */
function flushTokens() {
  rafId = null;

  if (!tokenBuffer || !activeAssistantMsgEl) return;

  activeMarkdownBuffer += tokenBuffer;
  tokenBuffer = '';

  const contentDiv = activeAssistantMsgEl.querySelector('.msg-content');
  if (!contentDiv) return;

  // Re-render the full markdown (incremental rendering causes too many edge cases)
  contentDiv.innerHTML = DOMPurify.sanitize(marked.parse(activeMarkdownBuffer));

  // Apply syntax highlighting to any code blocks
  contentDiv.querySelectorAll('pre code').forEach(block => {
    hljs.highlightElement(block);
  });

  // Setup copy buttons
  setupCopyButtons(contentDiv);

  // Auto-scroll if near bottom
  const container = dom.messagesContainer;
  if (container.scrollHeight - container.scrollTop <= container.clientHeight + 150) {
    container.scrollTop = container.scrollHeight;
  }
}

/**
 * Stream a chat request to the SSE endpoint using fetch + ReadableStream.
 *
 * Since EventSource only supports GET, we use fetch with POST and manually
 * parse the SSE text/event-stream format.
 *
 * @param {Array<{role:string, content:string}>} messages — Message history to send
 */
async function streamChat(messages) {
  if (state.isStreaming) return;

  state.isStreaming = true;
  state.abortController = new AbortController();
  setInputEnabled(false);

  // Create assistant message skeleton
  createAssistantMessage();

  let thinkingBuffer = '';

  try {
    const response = await fetch(CONFIG.SSE_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: messages[messages.length - 1]?.content || '',
        messages: messages,
        temperature: state.settings.temperature,
        top_p: state.settings.topP,
        top_k: state.settings.topK,
        num_ctx: state.settings.numCtx,
        system: state.settings.systemPrompt,
        stream: state.settings.streamTokens
      }),
      signal: state.abortController.signal
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    if (!response.body) {
      throw new Error('ReadableStream not supported in this browser');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let sseBuffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        // Flush any remaining tokens
        if (tokenBuffer) {
          flushTokens();
        }
        finalizeAssistantMessage();
        break;
      }

      sseBuffer += decoder.decode(value, { stream: true });
      const lines = sseBuffer.split('\n');
      sseBuffer = lines.pop(); // Keep incomplete line in buffer

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;

        const jsonStr = line.substring(6).trim();
        if (!jsonStr || jsonStr === '[DONE]') {
          continue;
        }

        try {
          const data = JSON.parse(jsonStr);
          handleSSEEvent(data, thinkingBuffer);

          // Update thinkingBuffer reference if it was a thinking event
          if (data.type === 'thinking_chunk' || data.type === 'thinking') {
            thinkingBuffer += (data.text || data.content || '');
          }
        } catch (parseErr) {
          if (state.settings.verboseMode) {
            console.warn('[SSE] Failed to parse:', jsonStr, parseErr);
          }
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      // User cancelled — finalize what we have
      if (tokenBuffer) flushTokens();
      finalizeAssistantMessage();
    } else {
      console.error('[SSE] Stream error:', err);
      queueToken(`\n\n**Error communicating with backend:** ${err.message}`);
      if (tokenBuffer) flushTokens();
      finalizeAssistantMessage();
    }
  } finally {
    state.isStreaming = false;
    state.abortController = null;
    setInputEnabled(true);
    dom.messageInput.focus();
  }
}

/**
 * Handle a parsed SSE event based on its type.
 *
 * Supported event types:
 *   - content_chunk / token: append text to active assistant message
 *   - thinking_start: show thinking panel
 *   - thinking_chunk / thinking: route to thinking panel
 *   - tool_start: create tool card in running state
 *   - tool_end: update tool card to completed/error
 *   - token_usage: update token counter
 *   - done: finalize message
 *   - error: show error message
 *
 * @param {Object} data — Parsed SSE event data
 * @param {string} thinkingBuffer — Current accumulated thinking text
 */
function handleSSEEvent(data) {
  switch (data.type) {
    case 'content_chunk':
    case 'token':
      queueToken(data.text || data.content || '');
      break;

    case 'thinking_start':
      if (activeAssistantMsgEl && state.settings.showThinking) {
        const panel = activeAssistantMsgEl.querySelector('.thinking-panel');
        if (panel) panel.style.display = 'block';
      }
      break;

    case 'thinking_chunk':
    case 'thinking':
      if (state.settings.showThinking) {
        appendThinkingContent(data.text || data.content || '');
      }
      break;

    case 'tool_start': {
      if (!activeAssistantMsgEl) break;
      const group = dom.messagesContainer.querySelector('.message-group');
      if (group) {
        const toolEl = createToolCard(data.name || data.tool, data.id || data.tool_id);
        if (data.input) {
          const inputEl = toolEl.querySelector('.tool-input');
          if (inputEl) inputEl.innerHTML = `<pre>${escapeHtml(JSON.stringify(data.input, null, 2))}</pre>`;
        }
        group.appendChild(toolEl);
      }
      break;
    }

    case 'tool_end': {
      const toolId = data.id || data.tool_id;
      const toolEl = dom.messagesContainer.querySelector(`[data-tool-id="${toolId}"]`);
      if (toolEl) {
        const status = data.error ? 'error' : 'completed';
        finalizeToolCard(toolEl, status, data.output || data.result || data.error);
      }
      break;
    }

    case 'token_usage':
      updateTokenCounter(data.total || data.used || 0, data.budget || data.max || state.maxTokens);
      // Update conversation token count
      const conv = getActiveConversation();
      if (conv) {
        conv.tokenCount = data.total || data.used || 0;
        saveConversationsToStorage();
      }
      break;

    case 'done':
      // Flush remaining tokens and finalize
      if (tokenBuffer) flushTokens();
      finalizeAssistantMessage();
      break;

    case 'error':
      queueToken(`\n\n**Error:** ${data.message || data.text || 'Unknown error'}`);
      break;

    default:
      if (state.settings.verboseMode) {
        console.log('[SSE] Unhandled event type:', data.type, data);
      }
  }
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §9  TOOL INVOCATION CARDS
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Create a tool invocation card in the "running" state.
 *
 * @param {string} toolName — Name of the tool being invoked
 * @param {string} toolId — Unique identifier for this tool invocation
 * @returns {HTMLElement} The tool card DOM element
 */
function createToolCard(toolName, toolId) {
  if (dom.tplToolCard) {
    const node = dom.tplToolCard.content.cloneNode(true).querySelector('.tool-card');
    node.setAttribute('data-tool-id', toolId || generateId());
    node.querySelector('.tool-name').textContent = toolName || 'tool';

    const statusIcon = node.querySelector('.tool-status-icon');
    if (statusIcon) statusIcon.textContent = '⚡';

    // Click header to expand/collapse
    const header = node.querySelector('.tool-header');
    const details = node.querySelector('.tool-details');
    if (header && details) {
      details.style.display = 'none';
      header.style.cursor = 'pointer';
      header.setAttribute('role', 'button');
      header.setAttribute('aria-expanded', 'false');
      header.addEventListener('click', () => {
        const isHidden = details.style.display === 'none';
        details.style.display = isHidden ? 'block' : 'none';
        header.setAttribute('aria-expanded', isHidden);
      });
    }

    return node;
  }

  // Fallback: build manually if no template
  const card = document.createElement('div');
  card.className = 'tool-card running';
  card.setAttribute('data-tool-id', toolId || generateId());
  card.innerHTML = `
    <div class="tool-header" role="button" aria-expanded="false" style="cursor:pointer;">
      <span class="tool-icon">⚡</span>
      <span class="tool-name">${escapeHtml(toolName || 'tool')}</span>
      <span class="tool-status">Running...</span>
      <span class="tool-chevron">▸</span>
    </div>
    <div class="tool-body" style="display:none;">
      <div class="tool-input"><pre></pre></div>
      <div class="tool-output"><pre></pre></div>
    </div>`;

  const header = card.querySelector('.tool-header');
  const body = card.querySelector('.tool-body');
  header.addEventListener('click', () => {
    const isHidden = body.style.display === 'none';
    body.style.display = isHidden ? 'block' : 'none';
    header.setAttribute('aria-expanded', isHidden);
    card.querySelector('.tool-chevron').textContent = isHidden ? '▾' : '▸';
  });

  return card;
}

/**
 * Update a tool card to its final state (completed or error).
 *
 * @param {HTMLElement} toolEl — The tool card DOM element
 * @param {string} status — 'completed' or 'error'
 * @param {string} [output] — Tool output/result text
 */
function finalizeToolCard(toolEl, status, output) {
  toolEl.classList.remove('running');
  toolEl.classList.add(status);

  const statusIcon = toolEl.querySelector('.tool-status-icon') || toolEl.querySelector('.tool-icon');
  if (statusIcon) {
    statusIcon.textContent = status === 'completed' ? '✓' : '✗';
  }

  const statusText = toolEl.querySelector('.tool-status');
  if (statusText) {
    statusText.textContent = status === 'completed' ? 'Completed' : 'Error';
  }

  if (output) {
    const outputEl = toolEl.querySelector('.tool-output');
    if (outputEl) {
      const outputStr = typeof output === 'string' ? output : JSON.stringify(output, null, 2);
      outputEl.innerHTML = `<pre>${escapeHtml(outputStr)}</pre>`;
    }
  }
}

/**
 * Toggle expand/collapse of a tool card body (used for onclick in inline HTML).
 * @param {HTMLElement} headerEl — The tool-header element that was clicked
 */
function toggleToolExpand(headerEl) {
  const card = headerEl.closest('.tool-card');
  if (!card) return;
  const body = card.querySelector('.tool-body') || card.querySelector('.tool-details');
  if (!body) return;

  const isHidden = body.style.display === 'none' || !body.style.display;
  body.style.display = isHidden ? 'block' : 'none';

  const chevron = card.querySelector('.tool-chevron');
  if (chevron) chevron.textContent = isHidden ? '▾' : '▸';
}

// Expose for inline onclick handlers
window.toggleToolExpand = toggleToolExpand;


/* ═══════════════════════════════════════════════════════════════════════════════
 * §10  MESSAGE INPUT SYSTEM
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Set up auto-resizing textarea behavior.
 * The textarea grows with content up to its CSS max-height.
 */
function setupAutoResizeTextarea() {
  dom.messageInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
  });
}

/**
 * Enable or disable the message input and send button.
 * @param {boolean} enabled — Whether input should be enabled
 */
function setInputEnabled(enabled) {
  dom.messageInput.disabled = !enabled;
  dom.sendBtn.disabled = !enabled;
  dom.sendBtn.setAttribute('aria-disabled', !enabled);

  if (enabled) {
    dom.messageInput.removeAttribute('readonly');
    dom.messageInput.placeholder = 'Type / for commands...';
  } else {
    dom.messageInput.setAttribute('readonly', 'true');
    dom.messageInput.placeholder = 'Selene is thinking...';
  }
}

/**
 * Handle message submission from the input field.
 * Creates user message, sends to SSE, shows assistant skeleton.
 */
function handleSubmit() {
  const text = dom.messageInput.value.trim();
  if (!text || state.isStreaming) return;

  // Ensure we have an active conversation
  if (!state.activeConversationId) {
    createNewConversation();
    // Remove welcome message (createNewConversation shows one, but we'll append messages)
    const welcome = dom.messagesContainer.querySelector('.welcome-msg');
    if (welcome) welcome.closest('.message-group')?.remove() || welcome.remove();
  }

  // Clear input
  dom.messageInput.value = '';
  dom.messageInput.style.height = 'auto';

  // Append user message to viewport
  appendUserMessage(text);

  // Prepare messages for API
  const conv = getActiveConversation();
  const apiMessages = conv
    ? conv.messages.map(m => ({ role: m.role, content: m.content }))
    : [{ role: 'user', content: text }];

  // Stream the response
  streamChat(apiMessages);
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §11  SLASH COMMAND MENU
 * ═══════════════════════════════════════════════════════════════════════════════ */

/** @type {Array<{cmd:string, desc:string, immediate:boolean}>} Available slash commands */
const SLASH_COMMANDS = [
  { cmd: '/clear',  desc: 'Clear current conversation',    immediate: true },
  { cmd: '/save',   desc: 'Save session to history',       immediate: false },
  { cmd: '/load',   desc: 'Load a saved session',          immediate: false },
  { cmd: '/system', desc: 'Edit the system prompt',        immediate: true },
  { cmd: '/help',   desc: 'Show all commands',             immediate: true }
];

/** @type {number} Currently highlighted command index in the slash menu */
let slashSelectedIndex = 0;

/** @type {Array} Filtered commands currently displayed */
let filteredCommands = [...SLASH_COMMANDS];

/**
 * Show the slash command menu above the input field.
 * @param {string} [filter=''] — Filter text typed after /
 */
function showSlashMenu(filter = '') {
  filteredCommands = SLASH_COMMANDS.filter(c =>
    c.cmd.toLowerCase().includes(('/' + filter).toLowerCase())
  );

  if (filteredCommands.length === 0) {
    hideSlashMenu();
    return;
  }

  slashSelectedIndex = Math.min(slashSelectedIndex, filteredCommands.length - 1);

  dom.slashMenu.innerHTML = filteredCommands.map((c, i) => `
    <div class="cmd-item ${i === slashSelectedIndex ? 'selected' : ''}"
         data-idx="${i}"
         role="option"
         aria-selected="${i === slashSelectedIndex}">
      <span class="cmd-label">${escapeHtml(c.cmd)}</span>
      <span class="cmd-desc">${escapeHtml(c.desc)}</span>
    </div>
  `).join('');

  dom.slashMenu.classList.add('visible');
  dom.slashMenu.setAttribute('role', 'listbox');

  // Hover handlers
  dom.slashMenu.querySelectorAll('.cmd-item').forEach(item => {
    item.addEventListener('mouseenter', () => {
      slashSelectedIndex = parseInt(item.dataset.idx);
      highlightSlashItem();
    });
    item.addEventListener('click', () => {
      executeSlashCommand(filteredCommands[parseInt(item.dataset.idx)]);
    });
  });
}

/**
 * Update visual highlight on the selected slash command without full re-render.
 */
function highlightSlashItem() {
  dom.slashMenu.querySelectorAll('.cmd-item').forEach((item, i) => {
    item.classList.toggle('selected', i === slashSelectedIndex);
    item.setAttribute('aria-selected', i === slashSelectedIndex);
  });
}

/**
 * Hide the slash command menu.
 */
function hideSlashMenu() {
  dom.slashMenu.classList.remove('visible');
  slashSelectedIndex = 0;
}

/**
 * Execute a selected slash command.
 * @param {Object} cmdObj — Command object { cmd, desc, immediate }
 */
function executeSlashCommand(cmdObj) {
  hideSlashMenu();

  switch (cmdObj.cmd) {
    case '/clear':
      clearActiveConversation();
      dom.messageInput.value = '';
      break;

    case '/save':
      dom.messageInput.value = '/save ';
      dom.messageInput.focus();
      return; // Don't clear — user needs to type name

    case '/load':
      dom.messageInput.value = '/load ';
      dom.messageInput.focus();
      return;

    case '/system':
      openSettingsDrawer();
      dom.messageInput.value = '';
      // Focus the system prompt textarea after drawer animation
      setTimeout(() => {
        if (dom.settingSystemPrompt) dom.settingSystemPrompt.focus();
      }, 450);
      break;

    case '/help':
      dom.messageInput.value = '';
      showHelpMessage();
      break;

    default:
      dom.messageInput.value = cmdObj.cmd + ' ';
      dom.messageInput.focus();
      return;
  }

  dom.messageInput.focus();
}

/**
 * Handle a typed slash command (with arguments) on Enter.
 * @param {string} fullText — Full input text starting with /
 * @returns {boolean} True if a command was handled
 */
function handleTypedSlashCommand(fullText) {
  const parts = fullText.trim().split(/\s+/);
  const cmd = parts[0].toLowerCase();
  const arg = parts.slice(1).join(' ');

  switch (cmd) {
    case '/clear':
      clearActiveConversation();
      return true;

    case '/save':
      if (arg) {
        saveSession(arg);
      } else {
        saveSession();
      }
      return true;

    case '/load':
      if (arg) {
        // Find conversation by title
        const found = state.conversations.find(c =>
          c.title.toLowerCase().includes(arg.toLowerCase())
        );
        if (found) {
          loadConversation(found.id);
        } else {
          showSystemMessage(`No session found matching "${arg}"`);
        }
      }
      return true;

    case '/system':
      openSettingsDrawer();
      setTimeout(() => {
        if (dom.settingSystemPrompt) dom.settingSystemPrompt.focus();
      }, 450);
      return true;

    case '/help':
      showHelpMessage();
      return true;

    default:
      return false;
  }
}

/**
 * Clear the active conversation messages and reset viewport.
 */
function clearActiveConversation() {
  const conv = getActiveConversation();
  if (conv) {
    conv.messages = [];
    conv.tokenCount = 0;
    saveConversationsToStorage();
  }

  clearChatViewport();
  updateTokenCounter(0);
}

/**
 * Show a help message listing all available slash commands.
 */
function showHelpMessage() {
  const helpText = SLASH_COMMANDS.map(c =>
    `**${c.cmd}** — ${c.desc}`
  ).join('\n');

  showSystemMessage('## Available Commands\n\n' + helpText);
}

/**
 * Show a system message in the chat viewport (non-interactive).
 * @param {string} markdownText — Markdown-formatted message text
 */
function showSystemMessage(markdownText) {
  let group = dom.messagesContainer.querySelector('.message-group');
  if (!group) {
    group = document.createElement('div');
    group.className = 'message-group';
    dom.messagesContainer.appendChild(group);
  }

  const node = dom.tplAssistantMsg.content.cloneNode(true).querySelector('.message');
  node.querySelector('.msg-content').innerHTML = DOMPurify.sanitize(marked.parse(markdownText));
  group.appendChild(node);
  dom.messagesContainer.scrollTop = dom.messagesContainer.scrollHeight;
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §12  SETTINGS DRAWER
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Open the settings drawer with scrim overlay.
 */
function openSettingsDrawer() {
  dom.settingsDrawer.classList.add('open');
  dom.settingsScrim.classList.add('active');
  dom.settingsDrawer.setAttribute('aria-hidden', 'false');

  // Trap focus inside drawer
  dom.closeDrawerBtn.focus();
}

/**
 * Close the settings drawer and save settings.
 */
function closeSettingsDrawer() {
  dom.settingsDrawer.classList.remove('open');
  dom.settingsScrim.classList.remove('active');
  dom.settingsDrawer.setAttribute('aria-hidden', 'true');

  // Collect and persist current settings
  collectSettingsFromDOM();
  saveSettingsToStorage();
}

/**
 * Read all settings control values from the DOM into state.settings.
 */
function collectSettingsFromDOM() {
  if (dom.settingTemperature) {
    state.settings.temperature = parseFloat(dom.settingTemperature.value);
  }
  if (dom.settingTopP) {
    state.settings.topP = parseFloat(dom.settingTopP.value);
  }
  if (dom.settingNumCtx) {
    state.settings.numCtx = parseInt(dom.settingNumCtx.value, 10);
    state.maxTokens = state.settings.numCtx;
  }
  if (dom.settingSystemPrompt) {
    state.settings.systemPrompt = dom.settingSystemPrompt.value;
  }
  if (dom.toggleThinking) {
    state.settings.showThinking = dom.toggleThinking.checked;
  }
  if (dom.toggleStream) {
    state.settings.streamTokens = dom.toggleStream.checked;
  }
  if (dom.toggleVerbose) {
    state.settings.verboseMode = dom.toggleVerbose.checked;
  }
}

/**
 * Apply state.settings values to the DOM controls.
 */
function applySettingsToDOM() {
  if (dom.settingTemperature) {
    dom.settingTemperature.value = state.settings.temperature;
    if (dom.readoutTemperature) dom.readoutTemperature.textContent = state.settings.temperature;
  }
  if (dom.settingTopP) {
    dom.settingTopP.value = state.settings.topP;
    if (dom.readoutTopP) dom.readoutTopP.textContent = state.settings.topP;
  }
  if (dom.settingNumCtx) {
    dom.settingNumCtx.value = state.settings.numCtx;
    if (dom.readoutNumCtx) dom.readoutNumCtx.textContent = state.settings.numCtx;
  }
  if (dom.settingSystemPrompt) {
    dom.settingSystemPrompt.value = state.settings.systemPrompt;
  }
  if (dom.toggleThinking) {
    dom.toggleThinking.checked = state.settings.showThinking;
  }
  if (dom.toggleStream) {
    dom.toggleStream.checked = state.settings.streamTokens;
  }
  if (dom.toggleVerbose) {
    dom.toggleVerbose.checked = state.settings.verboseMode;
  }
}

/**
 * Set up live readout updates on range slider input events.
 */
function setupSliderReadouts() {
  const sliderMap = [
    { slider: dom.settingTemperature, readout: dom.readoutTemperature },
    { slider: dom.settingTopP, readout: dom.readoutTopP },
    { slider: dom.settingNumCtx, readout: dom.readoutNumCtx }
  ];

  sliderMap.forEach(({ slider, readout }) => {
    if (slider && readout) {
      slider.addEventListener('input', () => {
        readout.textContent = slider.value;
      });
    }
  });
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §13  SESSION MANAGEMENT
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Get the currently active conversation object.
 * @returns {Object|undefined} Active conversation or undefined
 */
function getActiveConversation() {
  return state.conversations.find(c => c.id === state.activeConversationId);
}

/**
 * Save a session (active conversation) with an optional name.
 * @param {string} [name] — Optional session name (defaults to conversation title)
 */
function saveSession(name) {
  const conv = getActiveConversation();
  if (!conv) {
    showSystemMessage('No active conversation to save.');
    return;
  }

  if (name) {
    conv.title = name;
  }

  conv.timestamp = Date.now();
  saveConversationsToStorage();
  renderConversationList(dom.sidebarSearch ? dom.sidebarSearch.value : '');

  showSystemMessage(`Session saved as **"${escapeHtml(conv.title)}"**`);
}

/**
 * Export the active conversation as a downloadable JSON file.
 */
function exportConversation() {
  const conv = getActiveConversation();
  if (!conv) {
    showSystemMessage('No active conversation to export.');
    return;
  }

  const exportData = {
    title: conv.title,
    id: conv.id,
    timestamp: conv.timestamp,
    tokenCount: conv.tokenCount,
    messages: conv.messages,
    settings: { ...state.settings },
    exportedAt: new Date().toISOString()
  };

  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `selene-${conv.title.replace(/[^a-z0-9]/gi, '-').toLowerCase()}-${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  showSystemMessage('Conversation exported as JSON.');
}

/**
 * Save all conversations to localStorage.
 */
function saveConversationsToStorage() {
  try {
    localStorage.setItem(
      CONFIG.STORAGE_KEYS.CONVERSATIONS,
      JSON.stringify(state.conversations)
    );
    localStorage.setItem(
      CONFIG.STORAGE_KEYS.ACTIVE_CONVERSATION,
      state.activeConversationId || ''
    );
  } catch (e) {
    console.warn('[Selene] Failed to save conversations to localStorage:', e);
  }
}

/**
 * Load all conversations from localStorage into state.
 */
function loadConversationsFromStorage() {
  try {
    const stored = localStorage.getItem(CONFIG.STORAGE_KEYS.CONVERSATIONS);
    if (stored) {
      state.conversations = JSON.parse(stored);
    }

    const activeId = localStorage.getItem(CONFIG.STORAGE_KEYS.ACTIVE_CONVERSATION);
    if (activeId && state.conversations.some(c => c.id === activeId)) {
      state.activeConversationId = activeId;
    }
  } catch (e) {
    console.warn('[Selene] Failed to load conversations from localStorage:', e);
    state.conversations = [];
  }
}

/**
 * Save current settings to localStorage.
 */
function saveSettingsToStorage() {
  try {
    localStorage.setItem(CONFIG.STORAGE_KEYS.SETTINGS, JSON.stringify(state.settings));
  } catch (e) {
    console.warn('[Selene] Failed to save settings:', e);
  }
}

/**
 * Load settings from localStorage, merging with defaults.
 */
function loadSettingsFromStorage() {
  try {
    const stored = localStorage.getItem(CONFIG.STORAGE_KEYS.SETTINGS);
    if (stored) {
      const parsed = JSON.parse(stored);
      state.settings = { ...CONFIG.DEFAULT_SETTINGS, ...parsed };
    }
  } catch (e) {
    console.warn('[Selene] Failed to load settings:', e);
    state.settings = { ...CONFIG.DEFAULT_SETTINGS };
  }
}

/**
 * Load history and settings from the backend API (if available).
 * Falls back gracefully if the API is unavailable.
 */
async function loadFromBackend() {
  try {
    const res = await fetch(CONFIG.SETTINGS_ENDPOINT);
    if (!res.ok) return;

    const data = await res.json();

    // Populate session list from backend
    if (data.saved_sessions) {
      dom.conversationList.innerHTML = '';
      data.saved_sessions.forEach(sessionName => {
        const cleanName = sessionName.replace('.json', '');
        const card = document.createElement('div');
        card.className = `session-card ${data.active_session_name === sessionName ? 'active' : ''}`;
        card.innerHTML = `<div class="session-title">${escapeHtml(cleanName)}</div>`;
        card.setAttribute('role', 'button');
        card.setAttribute('tabindex', '0');
        card.addEventListener('click', () => {
          executeSlashCommand({ cmd: `/load ${cleanName}`, desc: '', immediate: false });
        });
        dom.conversationList.appendChild(card);
        applyTilt(card);
      });
    }

    // Apply backend settings to sliders
    if (data.settings && data.settings.options) {
      const opts = data.settings.options;
      if (opts.temperature !== undefined) {
        state.settings.temperature = opts.temperature;
      }
      if (opts.top_p !== undefined) {
        state.settings.topP = opts.top_p;
      }
      if (opts.num_ctx !== undefined) {
        state.settings.numCtx = opts.num_ctx;
        state.maxTokens = opts.num_ctx;
      }
      applySettingsToDOM();
    }
    if (data.settings && data.settings.system !== undefined) {
      state.settings.systemPrompt = data.settings.system;
      if (dom.settingSystemPrompt) {
        dom.settingSystemPrompt.value = data.settings.system;
      }
    }

    // Rehydrate message history
    if (data.history && data.history.length > 0) {
      const welcome = dom.messagesContainer.querySelector('.welcome-msg');
      if (welcome) welcome.closest('.message-group')?.remove() || welcome.remove();

      let group = dom.messagesContainer.querySelector('.message-group');
      if (!group) {
        group = document.createElement('div');
        group.className = 'message-group';
        dom.messagesContainer.appendChild(group);
      }

      data.history.forEach(msg => {
        if (msg.role === 'user') {
          const node = dom.tplUserMsg.content.cloneNode(true).querySelector('.message');
          node.querySelector('.msg-content').textContent = msg.content;
          group.appendChild(node);
        } else if (msg.role === 'assistant') {
          const node = dom.tplAssistantMsg.content.cloneNode(true).querySelector('.message');

          if (msg.thinking) {
            const panel = node.querySelector('.thinking-panel');
            panel.style.display = 'block';
            node.querySelector('.thinking-content').textContent = msg.thinking;
            const toggle = node.querySelector('.thinking-toggle');
            toggle.addEventListener('click', function () {
              panel.classList.toggle('collapsed');
              panel.classList.toggle('expanded');
            });
          }

          const contentEl = node.querySelector('.msg-content');
          contentEl.innerHTML = DOMPurify.sanitize(marked.parse(msg.content || ''));
          contentEl.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
          });
          setupCopyButtons(contentEl);

          group.appendChild(node);
        }
      });

      dom.messagesContainer.scrollTop = dom.messagesContainer.scrollHeight;
    }
  } catch (e) {
    if (state.settings.verboseMode) {
      console.warn('[Selene] Could not load from backend:', e);
    }
  }
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §14  COPY CODE FUNCTION
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Set up copy buttons within a rendered content container.
 * Each code block's copy button copies the code text to clipboard.
 *
 * @param {HTMLElement} container — DOM element containing rendered markdown
 */
function setupCopyButtons(container) {
  container.querySelectorAll('.copy-btn').forEach(btn => {
    // Prevent duplicate listeners
    if (btn._copyBound) return;
    btn._copyBound = true;

    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const pre = btn.closest('pre') || btn.closest('.code-header')?.nextElementSibling;
      let codeText = '';

      if (pre) {
        const codeEl = pre.querySelector('code') || pre;
        codeText = codeEl.innerText || codeEl.textContent;
      }

      copyToClipboard(codeText, btn);
    });
  });
}

/**
 * Copy text to clipboard with visual feedback on the trigger button.
 * Uses navigator.clipboard API with fallback to textarea-based copy.
 *
 * @param {string} text — Text to copy
 * @param {HTMLElement} [btn] — Button element to show feedback on
 */
async function copyToClipboard(text, btn) {
  try {
    // Modern clipboard API
    await navigator.clipboard.writeText(text);
    showCopyFeedback(btn, true);
  } catch (err) {
    // Fallback: textarea-based copy for older browsers
    try {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.cssText = 'position:fixed;opacity:0;top:-1000px;left:-1000px;';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      showCopyFeedback(btn, true);
    } catch (fallbackErr) {
      console.error('[Selene] Failed to copy:', fallbackErr);
      showCopyFeedback(btn, false);
    }
  }
}

/**
 * Show visual feedback on a copy button after a copy attempt.
 * @param {HTMLElement} btn — The copy button element
 * @param {boolean} success — Whether the copy succeeded
 */
function showCopyFeedback(btn, success) {
  if (!btn) return;

  const originalText = btn.textContent;
  btn.textContent = success ? '✓ Copied!' : '✗ Failed';
  btn.disabled = true;

  setTimeout(() => {
    btn.textContent = originalText;
    btn.disabled = false;
  }, 2000);
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §15  CARD TILT EFFECT
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Apply a subtle 3D tilt effect to an element on mouse hover.
 *
 * Uses mousemove with RAF + 60ms throttle for smooth performance.
 * Max tilt: ±3 degrees. Resets smoothly on mouseleave.
 *
 * @param {HTMLElement} el — Element to apply tilt effect to
 */
function applyTilt(el) {
  let ticking = false;
  let rect = null;

  el.addEventListener('mouseenter', () => {
    rect = el.getBoundingClientRect();
  });

  const handleMove = throttle((e) => {
    if (!rect) return;

    requestAnimationFrame(() => {
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Map position to -3deg to +3deg
      const rotateY = ((x / rect.width) - 0.5) * 6;
      const rotateX = -((y / rect.height) - 0.5) * 6;

      el.style.transform = `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-2px)`;
    });
  }, 60);

  el.addEventListener('mousemove', handleMove);

  el.addEventListener('mouseleave', () => {
    el.style.transform = '';
    rect = null;
  });
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §16  MARKED.JS CONFIGURATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Configure the marked.js Markdown parser.
 *
 * - GFM mode with line breaks enabled
 * - Custom code block rendering with language label + copy button
 * - Syntax highlighting via highlight.js
 */
function configureMarked() {
  // Custom renderer for code blocks
  const renderer = new marked.Renderer();

  /**
   * Custom code block renderer.
   * Wraps code in a container with a language label header and a copy button.
   */
  renderer.code = function (code, language) {
    // Handle marked v12+ which passes an object
    let codeStr = code;
    let lang = language;
    if (typeof code === 'object' && code !== null) {
      codeStr = code.text || code.raw || '';
      lang = code.lang || language || '';
    }

    const validLang = lang && hljs.getLanguage(lang) ? lang : '';
    const displayLang = validLang || 'text';

    let highlighted;
    if (validLang) {
      highlighted = hljs.highlight(codeStr, { language: validLang }).value;
    } else {
      highlighted = hljs.highlightAuto(codeStr).value;
    }

    return `<pre><div class="code-header"><span class="code-lang">${escapeHtml(displayLang)}</span><button class="copy-btn" aria-label="Copy code">Copy</button></div><code class="hljs language-${escapeHtml(displayLang)}">${highlighted}</code></pre>`;
  };

  marked.setOptions({
    renderer: renderer,
    breaks: true,
    gfm: true
  });
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §17  KEYBOARD SHORTCUTS
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Set up global keyboard shortcut handlers.
 *
 * - Escape: Close settings drawer, dismiss slash menu
 * - Ctrl+Shift+S: Toggle sidebar
 */
function setupKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    // Escape: close drawers/menus
    if (e.key === 'Escape') {
      if (dom.settingsDrawer.classList.contains('open')) {
        closeSettingsDrawer();
        e.preventDefault();
        return;
      }
      if (dom.slashMenu.classList.contains('visible')) {
        hideSlashMenu();
        e.preventDefault();
        return;
      }
    }

    // Ctrl+Shift+S: Toggle sidebar
    if (e.ctrlKey && e.shiftKey && e.key === 'S') {
      e.preventDefault();
      toggleSidebar();
      return;
    }
  });
}

/**
 * Set up input-specific keyboard handlers (Enter, arrows for slash menu, etc).
 */
function setupInputKeyboard() {
  dom.messageInput.addEventListener('keydown', (e) => {
    // Slash menu navigation
    if (dom.slashMenu.classList.contains('visible')) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        slashSelectedIndex = (slashSelectedIndex + 1) % filteredCommands.length;
        highlightSlashItem();
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        slashSelectedIndex = (slashSelectedIndex - 1 + filteredCommands.length) % filteredCommands.length;
        highlightSlashItem();
        return;
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredCommands[slashSelectedIndex]) {
          executeSlashCommand(filteredCommands[slashSelectedIndex]);
        }
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        hideSlashMenu();
        return;
      }
    }

    // Submit on Enter (not Shift+Enter)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const text = dom.messageInput.value.trim();

      // Handle typed slash commands
      if (text.startsWith('/')) {
        if (handleTypedSlashCommand(text)) {
          dom.messageInput.value = '';
          dom.messageInput.style.height = 'auto';
          return;
        }
      }

      handleSubmit();
    }
  });

  // Slash menu trigger on input
  dom.messageInput.addEventListener('input', () => {
    const val = dom.messageInput.value;

    if (val.startsWith('/')) {
      const filter = val.substring(1);
      slashSelectedIndex = 0;
      showSlashMenu(filter);
    } else {
      hideSlashMenu();
    }
  });
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §18  ACCESSIBILITY
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Set ARIA attributes on key interactive elements.
 */
function setupAccessibility() {
  // Sidebar
  if (dom.sidebar) {
    dom.sidebar.setAttribute('role', 'navigation');
    dom.sidebar.setAttribute('aria-label', 'Conversation history');
    dom.sidebar.setAttribute('aria-expanded', !state.sidebarCollapsed);
  }

  // Chat viewport
  const viewport = dom.chatViewport;
  if (viewport) {
    viewport.setAttribute('role', 'main');
    viewport.setAttribute('aria-label', 'Chat messages');
  }

  // Settings drawer
  if (dom.settingsDrawer) {
    dom.settingsDrawer.setAttribute('role', 'complementary');
    dom.settingsDrawer.setAttribute('aria-label', 'Settings panel');
    dom.settingsDrawer.setAttribute('aria-hidden', 'true');
  }

  // Send button
  if (dom.sendBtn) {
    dom.sendBtn.setAttribute('aria-label', 'Send message');
  }

  // Sidebar toggle
  if (dom.sidebarToggle) {
    dom.sidebarToggle.setAttribute('aria-label', 'Toggle sidebar');
  }

  // Slash menu
  if (dom.slashMenu) {
    dom.slashMenu.setAttribute('role', 'listbox');
    dom.slashMenu.setAttribute('aria-label', 'Slash commands');
  }

  // Settings toggle
  if (dom.settingsToggle) {
    dom.settingsToggle.setAttribute('aria-label', 'Open settings');
  }

  // Live region for streaming announcements
  let liveRegion = document.getElementById('selene-live-region');
  if (!liveRegion) {
    liveRegion = document.createElement('div');
    liveRegion.id = 'selene-live-region';
    liveRegion.setAttribute('role', 'status');
    liveRegion.setAttribute('aria-live', 'polite');
    liveRegion.setAttribute('aria-atomic', 'true');
    liveRegion.style.cssText = 'position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);';
    document.body.appendChild(liveRegion);
  }
}

/**
 * Announce a message to screen readers via the live region.
 * @param {string} message — Text to announce
 */
function announceToScreenReader(message) {
  const liveRegion = document.getElementById('selene-live-region');
  if (liveRegion) {
    liveRegion.textContent = '';
    // Force re-announcement by clearing and re-setting
    requestAnimationFrame(() => {
      liveRegion.textContent = message;
    });
  }
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §19  EVENT LISTENERS SETUP
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Bind all event listeners to cached DOM elements.
 * Called once during initialization.
 */
function setupEventListeners() {
  // --- Sidebar ---
  dom.sidebarToggle.addEventListener('click', toggleSidebar);

  dom.newChatBtn.addEventListener('click', createNewConversation);

  // Sidebar search with debounced filtering
  if (dom.sidebarSearch) {
    dom.sidebarSearch.addEventListener('input', debounce((e) => {
      renderConversationList(e.target.value);
    }, 200));
  }

  // --- Settings drawer ---
  dom.settingsToggle.addEventListener('click', openSettingsDrawer);
  dom.closeDrawerBtn.addEventListener('click', closeSettingsDrawer);
  dom.settingsScrim.addEventListener('click', closeSettingsDrawer);

  // --- Send button ---
  dom.sendBtn.addEventListener('click', handleSubmit);

  // --- Input system ---
  setupAutoResizeTextarea();
  setupInputKeyboard();

  // --- Slider readouts ---
  setupSliderReadouts();

  // --- Session buttons ---
  if (dom.sessionButtons.length >= 3) {
    dom.sessionButtons[0].addEventListener('click', () => saveSession());
    dom.sessionButtons[1].addEventListener('click', () => {
      // Open a simple prompt for session name
      const conv = getActiveConversation();
      const name = prompt('Enter session name to load:', '');
      if (name) {
        const found = state.conversations.find(c =>
          c.title.toLowerCase().includes(name.toLowerCase())
        );
        if (found) {
          loadConversation(found.id);
          closeSettingsDrawer();
        } else {
          showSystemMessage(`No session found matching "${name}"`);
        }
      }
    });
    dom.sessionButtons[2].addEventListener('click', () => exportConversation());
  }

  // --- Click outside slash menu to dismiss ---
  document.addEventListener('click', (e) => {
    if (dom.slashMenu.classList.contains('visible') &&
        !dom.slashMenu.contains(e.target) &&
        e.target !== dom.messageInput) {
      hideSlashMenu();
    }
  });
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * §20  INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Main initialization function.
 * Runs after DOMContentLoaded — sets up the entire application.
 */
function init() {
  // 1. Cache all DOM references
  cacheDOMReferences();

  // 2. Configure marked.js Markdown parser
  configureMarked();

  // 3. Load settings from localStorage
  loadSettingsFromStorage();

  // 4. Load conversations from localStorage
  loadConversationsFromStorage();

  // 5. Restore sidebar state
  restoreSidebarState();

  // 6. Apply settings to DOM controls
  applySettingsToDOM();

  // 7. Initialize starfield canvas
  initStarfield();

  // 8. Set up all event listeners
  setupEventListeners();

  // 9. Set up keyboard shortcuts
  setupKeyboardShortcuts();

  // 10. Set up accessibility attributes
  setupAccessibility();

  // 11. Render conversation list in sidebar
  renderConversationList();

  // 12. Load the active conversation if one exists, or show welcome
  if (state.activeConversationId) {
    loadConversation(state.activeConversationId);
  }

  // 13. Initialize token counter display
  updateTokenCounter(state.currentTokens, state.maxTokens);

  // 14. Load history from backend (async, non-blocking)
  loadFromBackend();

  // 15. Focus the message input
  dom.messageInput.focus();

  if (state.settings.verboseMode) {
    console.log('[Selene] Initialized.', {
      conversations: state.conversations.length,
      activeId: state.activeConversationId,
      settings: state.settings
    });
  }
}

// Boot
document.addEventListener('DOMContentLoaded', init);
