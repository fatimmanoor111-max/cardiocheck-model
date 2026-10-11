const form = document.getElementById("form");
const btn = document.getElementById("btn");
const result = document.getElementById("result");
const out = result.querySelector(".out");
const placeholder = result.querySelector(".placeholder");
const errorEl = document.getElementById("error");

const MESSAGES = {
  low: "The model estimates a low likelihood of heart disease for these measurements.",
  moderate: "The model estimates a moderate likelihood. Clinical follow-up is advisable.",
  high: "The model estimates a high likelihood of heart disease. Please consult a cardiologist.",
};
const LABELS = { low: "Low risk", moderate: "Moderate risk", high: "High risk" };

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.hidden = true;
  btn.disabled = true;
  btn.textContent = "Analyzing…";
  try {
    const payload = Object.fromEntries(new FormData(form).entries());
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");

    placeholder.hidden = true;
    out.hidden = false;
    result.className = "card result " + data.level;
    document.getElementById("badge").textContent = LABELS[data.level];
    document.getElementById("pct").textContent = data.probability;
    document.getElementById("msg").textContent = MESSAGES[data.level];
    requestAnimationFrame(() => (document.getElementById("bar").style.width = data.probability + "%"));
    if (window.innerWidth < 860) result.scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (err) {
    out.hidden = true;
    placeholder.hidden = true;
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict Risk";
  }
});

document.getElementById("reset").addEventListener("click", () => {
  out.hidden = true;
  errorEl.hidden = true;
  placeholder.hidden = false;
  result.className = "card result idle";
  document.getElementById("bar").style.width = "0";
});
