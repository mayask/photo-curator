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
  $$(".review-card").forEach((card) => {
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

  async function setDecision(card, decision) {
    const linkId = card.dataset.linkId;
    try {
      await request(`/api/collection-links/${linkId}`, {
        method: "PATCH", body: JSON.stringify({ decision }),
      });
      card.classList.remove("decision-keep", "decision-reject", "decision-pending");
      card.classList.add(`decision-${decision}`);
      $$('[data-decision]', card).forEach((button) => {
        button.classList.toggle("selected", button.dataset.decision === decision);
      });
    } catch (error) { toast(error.message, true); }
  }

  document.addEventListener("keydown", (event) => {
    if (!activeCard || event.metaKey || event.ctrlKey || event.altKey) return;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    const decision = { k: "keep", r: "reject", u: "pending" }[event.key.toLowerCase()];
    if (decision) {
      event.preventDefault();
      setDecision(activeCard, decision);
    }
  });

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
        const value = Number(button.dataset.rating);
        try {
          await request(`/api/photos/${rating.dataset.photoId}/rating`, {
            method: "PATCH", body: JSON.stringify({ rating: value }),
          });
          $$('[data-rating]', rating).forEach((item) => {
            item.classList.toggle("active", Number(item.dataset.rating) <= value);
          });
          toast("Rating saved");
        } catch (error) { toast(error.message, true); }
      });
    });
  }
})();
