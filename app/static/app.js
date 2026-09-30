(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  async function request(url, options = {}) {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try { detail = (await response.json()).detail || detail; } catch (_) { /* no-op */ }
      throw new Error(detail);
    }
    const type = response.headers.get("content-type") || "";
    return type.includes("json") ? response.json() : response.text();
  }

  function toast(message, error = false) {
    let element = $("#toast");
    if (!element) {
      element = document.createElement("div");
      element.id = "toast";
      Object.assign(element.style, {
        position: "fixed", zIndex: 100, left: "50%", bottom: "24px",
        transform: "translate(-50%, 15px)", padding: "10px 16px", borderRadius: "30px",
        color: "white", fontSize: "12px", opacity: 0, transition: ".2s ease",
      });
      document.body.append(element);
    }
    element.textContent = message;
    element.style.background = error ? "#93402e" : "#245c46";
    element.style.opacity = 1;
    element.style.transform = "translate(-50%, 0)";
    clearTimeout(element._timer);
    element._timer = setTimeout(() => {
      element.style.opacity = 0;
      element.style.transform = "translate(-50%, 15px)";
    }, 2800);
  }

  $$('[data-job]').forEach((button) => {
    button.addEventListener("click", async () => {
      const kind = button.dataset.job;
      button.disabled = true;
      try {
        const result = await request(`/api/jobs/${kind}`, { method: "POST" });
        toast(`Job #${result.job_id} queued`);
        pollStatus();
      } catch (error) {
        toast(error.message, true);
      } finally {
        button.disabled = false;
      }
    });
  });

  const retryErrors = $("#retry-errors");
  if (retryErrors) {
    retryErrors.addEventListener("click", async () => {
      retryErrors.disabled = true;
      try {
        const result = await request("/api/retry-errors", { method: "POST" });
        toast(`${result.reset} files reset; job #${result.job_id} queued`);
        pollStatus();
      } catch (error) {
        toast(error.message, true);
        retryErrors.disabled = false;
      }
    });
  }

  $$('[data-cancel-job]').forEach((button) => {
    button.addEventListener("click", async () => {
      if (!confirm("Stop this job after the current file?")) return;
      try {
        await request(`/api/jobs/${button.dataset.cancelJob}/cancel`, { method: "POST" });
        toast("Cancellation requested");
        setTimeout(() => location.reload(), 800);
      } catch (error) { toast(error.message, true); }
    });
  });

  async function pollStatus() {
    try {
      const status = await request("/api/status");
      const banner = $("#job-banner");
      if (!banner) return;
      if (!status.job) {
        banner.classList.add("hidden");
        return;
      }
      banner.classList.remove("hidden");
      $("#job-phase").textContent = status.job.phase.replaceAll("_", " ");
      $("#job-message").textContent = status.job.message || "Working…";
      $("#job-progress").style.width = status.job.percent == null ? "12%" : `${status.job.percent}%`;
    } catch (_) { /* transient network issue; next poll will retry */ }
  }
  pollStatus();
  setInterval(pollStatus, 4000);

  let activeCard = null;
  const reviewCards = $$(".review-card");
  reviewCards.forEach((card) => {
    card.addEventListener("focus", () => selectCard(card));
    card.addEventListener("click", (event) => {
      if (event.target.closest("a")) return;
      selectCard(card);
    });
    $$('[data-decision]', card).forEach((button) => {
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        selectCard(card);
        setDecision(card, button.dataset.decision);
      });
    });
  });

  function selectCard(card) {
    if (activeCard) activeCard.classList.remove("selected-card");
    activeCard = card;
    activeCard.classList.add("selected-card");
  }

  function adjustDecisionCount(decision, amount) {
    $$(`[data-decision-count="${decision}"]`).forEach((counter) => {
      counter.textContent = String(Math.max(0, Number(counter.textContent) + amount));
    });
  }

  async function setDecision(card, decision) {
    const linkId = card.dataset.linkId;
    const previous = card.dataset.currentDecision;
    try {
      await request(`/api/collection-links/${linkId}`, {
        method: "PATCH", body: JSON.stringify({ decision }),
      });
      card.classList.remove("decision-keep", "decision-reject", "decision-pending");
      card.classList.add(`decision-${decision}`);
      card.dataset.currentDecision = decision;
      if (previous && previous !== decision) {
        adjustDecisionCount(previous, -1);
        adjustDecisionCount(decision, 1);
      }
      $$('[data-decision]', card).forEach((button) => {
        button.classList.toggle("selected", button.dataset.decision === decision);
      });
    } catch (error) { toast(error.message, true); }
  }

  document.addEventListener("keydown", (event) => {
    if (!activeCard || event.metaKey || event.ctrlKey || event.altKey) return;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (["ArrowLeft", "ArrowRight"].includes(event.key)) {
      event.preventDefault();
      const current = reviewCards.indexOf(activeCard);
      const offset = event.key === "ArrowRight" ? 1 : -1;
      const next = reviewCards[Math.max(0, Math.min(reviewCards.length - 1, current + offset))];
      if (next) {
        next.focus({ preventScroll: true });
        next.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
      }
      return;
    }
    const decision = { k: "keep", r: "reject", u: "pending" }[event.key.toLowerCase()];
    if (decision) {
      event.preventDefault();
      setDecision(activeCard, decision);
    }
  });

  const renameCollection = $("#rename-collection");
  if (renameCollection) {
    renameCollection.addEventListener("click", async () => {
      const titleElement = $("#collection-title");
      const title = prompt("Collection title", titleElement.textContent.trim());
      if (title == null || !title.trim()) return;
      try {
        const result = await request(`/api/collections/${renameCollection.dataset.collectionId}`, {
          method: "PATCH", body: JSON.stringify({ title: title.trim() }),
        });
        titleElement.textContent = result.title;
        document.title = `${result.title} · Photo Book Curator`;
        toast("Collection renamed");
      } catch (error) { toast(error.message, true); }
    });
  }

  const collectionStatus = $("#collection-status");
  if (collectionStatus) {
    collectionStatus.addEventListener("change", async () => {
      try {
        await request(`/api/collections/${collectionStatus.dataset.collectionId}`, {
          method: "PATCH", body: JSON.stringify({ status: collectionStatus.value }),
        });
        toast("Collection status saved");
      } catch (error) { toast(error.message, true); }
    });
  }

  const rating = $(".rating[data-photo-id]");
  if (rating) {
    $$('[data-rating]', rating).forEach((button) => {
      button.addEventListener("click", async () => {
        const clicked = Number(button.dataset.rating);
        const current = Number(rating.dataset.currentRating || 0);
        const value = clicked === current ? null : clicked;
        try {
          await request(`/api/photos/${rating.dataset.photoId}/rating`, {
            method: "PATCH", body: JSON.stringify({ rating: value }),
          });
          rating.dataset.currentRating = value == null ? "" : String(value);
          $$('[data-rating]', rating).forEach((item) => {
            item.classList.toggle("active", value != null && Number(item.dataset.rating) <= value);
          });
          toast(value == null ? "Rating cleared" : "Rating saved");
        } catch (error) { toast(error.message, true); }
      });
    });
  }
})();
