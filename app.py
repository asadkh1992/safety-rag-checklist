import os
import re
import json
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader

# ============================================================
# Safety RAG Checklist Generator
# Streamlit + FAISS + Sentence Transformers + Groq
# ============================================================

st.set_page_config(
    page_title="Safety RAG | Job Safety Checklist",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(255,193,7,.10), transparent 28%),
            radial-gradient(circle at 85% 15%, rgba(33,150,243,.10), transparent 25%),
            linear-gradient(135deg, #07131f 0%, #0c1d2b 50%, #102637 100%);
        color: #f5f7fa;
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #091823 0%, #0c2230 100%);
    }
    .hero {
        padding: 1.3rem 1.5rem;
        border: 1px solid rgba(255,193,7,.28);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(255,193,7,.12), rgba(33,150,243,.08));
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2.25rem; }
    .hero p { margin: .35rem 0 0; opacity: .85; }
    .source {
        padding: .55rem .75rem;
        border-left: 3px solid #ffc107;
        background: rgba(255,255,255,.04);
        border-radius: 5px;
        margin: .35rem 0;
        font-size: .9rem;
    }
    .risk-high { color: #ff6b6b; font-weight: 700; }
    .risk-med { color: #ffc107; font-weight: 700; }
    .risk-low { color: #66d9a3; font-weight: 700; }
    .small { font-size: .82rem; opacity: .72; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Seed knowledge base ----------
# These are concise, original summaries of commonly used safety principles.
# They are NOT reproductions of copyrighted standards.
SEED_DOCS = [
    {
        "source": "ISO 45001:2018 — Occupational health and safety management systems",
        "topic": "general",
        "text": (
            "Plan work using hazard identification and risk assessment before the task. "
            "Apply the hierarchy of controls: eliminate hazards where feasible, then substitution, "
            "engineering controls, administrative controls, and PPE. Establish operational controls, "
            "competence, communication, emergency preparedness, contractor controls, and monitoring. "
            "Workers should have a mechanism to participate in OH&S matters and report hazards."
        ),
    },
    {
        "source": "OSHA — Job Hazard Analysis / Hazard Communication principles",
        "topic": "general",
        "text": (
            "Break a job into steps, identify hazards associated with each step, determine controls, "
            "and brief affected workers before starting. Chemical hazards require accessible hazard "
            "information, appropriate labels/SDS, worker training, and suitable controls. PPE selection "
            "should follow hazard assessment rather than be the sole control."
        ),
    },
    {
        "source": "OSHA 29 CFR 1910.146 — Permit-required confined spaces",
        "topic": "confined space",
        "text": (
            "Before entry, determine whether the space meets the applicable confined-space criteria and "
            "whether permit requirements apply. Establish an entry permit/program where required; identify "
            "hazards; isolate energy and process hazards; test atmosphere using suitable calibrated instruments; "
            "provide ventilation where needed; maintain an attendant and communications; establish rescue/emergency "
            "arrangements; and ensure entrants/attendants/supervisors are trained and authorized."
        ),
    },
    {
        "source": "OSHA 29 CFR 1910.147 — Control of hazardous energy (LOTO)",
        "topic": "lockout tagout",
        "text": (
            "Before servicing or maintenance, identify all energy sources, shut down equipment, isolate energy, "
            "apply locks/tags under the applicable procedure, release or restrain stored energy, and verify "
            "zero-energy state before work. Affected and authorized employees need appropriate training. "
            "Restoration requires inspection, personnel clearance and controlled removal of locks/tags."
        ),
    },
    {
        "source": "NFPA 51B — Standard for Fire Prevention During Welding, Cutting, and Other Hot Work",
        "topic": "hot work",
        "text": (
            "Hot work requires evaluation of fire/explosion hazards and controls appropriate to the work area. "
            "Use a hot-work permit system where required; remove or protect combustibles; control sparks, slag and "
            "heat transfer; provide suitable fire extinguishing equipment and fire watch when required; inspect "
            "adjacent areas and continue post-work fire monitoring for an appropriate period based on risk."
        ),
    },
    {
        "source": "NFPA 70E — Standard for Electrical Safety in the Workplace",
        "topic": "electrical",
        "text": (
            "Electrical work should be planned around shock and arc-flash hazards. Establish an electrically safe "
            "work condition when feasible through proper de-energization, isolation, lockout/tagout and verification. "
            "When energized work is justified and permitted, establish boundaries, shock/arc-flash risk controls, "
            "qualified personnel, appropriate PPE and insulated tools. Maintain approach boundaries and warning labels "
            "where applicable."
        ),
    },
    {
        "source": "API RP 54 — Occupational Safety for Oil and Gas Well Drilling and Servicing Operations",
        "topic": "oil and gas",
        "text": (
            "Oil and gas drilling/service work should use task-specific hazard assessment, competent supervision, "
            "energy/process isolation, dropped-object controls, well-control precautions, lifting controls, "
            "pressure-hazard controls, PPE, emergency response and effective communication. Personnel should be "
            "trained and competent for assigned duties."
        ),
    },
    {
        "source": "API RP 500 / API RP 505 — Classification of Locations for Electrical Installations",
        "topic": "hazardous area",
        "text": (
            "Where flammable gas, vapor or liquid may create a hazardous atmosphere, determine the applicable "
            "hazardous-area classification and ensure electrical/instrument equipment is suitable for the classified "
            "location. Control ignition sources and verify equipment certification/installation requirements."
        ),
    },
    {
        "source": "ILO — Safety and health in construction / occupational safety principles",
        "topic": "construction",
        "text": (
            "Construction activities should be planned with competent supervision, safe access/egress, housekeeping, "
            "fall prevention, lifting controls, excavation controls, electrical safety, PPE, emergency arrangements "
            "and worker consultation. Work areas should be kept orderly and hazards corrected promptly."
        ),
    },
    {
        "source": "Best industry practice — Work at height",
        "topic": "work at height",
        "text": (
            "Avoid work at height where reasonably practicable. Prefer collective protection such as compliant "
            "guardrails/platforms before personal fall-arrest systems. Inspect access equipment and fall-protection "
            "equipment, secure tools/materials against falling, establish exclusion zones where necessary, and provide "
            "a rescue plan rather than relying only on emergency services."
        ),
    },
    {
        "source": "Best industry practice — Lifting and rigging",
        "topic": "lifting",
        "text": (
            "Plan lifting operations according to load, center of gravity, lifting points, equipment capacity, ground "
            "conditions and environmental limits. Use inspected/certified lifting equipment and accessories, competent "
            "operators/riggers, an agreed signaling method, exclusion zones and controlled load travel. Never stand "
            "under a suspended load."
        ),
    },
    {
        "source": "Best industry practice — Excavation and trenching",
        "topic": "excavation",
        "text": (
            "Before excavation, identify underground services and obtain required permits/clearances. Assess soil and "
            "stability, provide suitable sloping/shoring/trench protection, safe access/egress, edge protection and "
            "controls for plant near edges. Keep spoil and equipment back from edges as required by the risk assessment. "
            "Inspect excavations after events that may affect stability."
        ),
    },
    {
        "source": "Best industry practice — Pressure testing",
        "topic": "pressure testing",
        "text": (
            "Pressure testing requires an approved procedure, defined test medium and pressure, calibrated test "
            "instruments, rated components, controlled test boundaries, exclusion zones and communication. Isolate "
            "non-rated equipment, secure temporary hoses/fittings, control stored energy, and prohibit personnel from "
            "line-of-fire positions during pressurization and depressurization."
        ),
    },
    {
        "source": "Best industry practice — Gas testing and process safety",
        "topic": "gas testing",
        "text": (
            "For potential flammable, toxic or oxygen-deficient atmospheres, use suitable calibrated and bump-tested "
            "gas detectors. Define sampling points and frequency, alarm/action levels, ventilation requirements and "
            "evacuation criteria. Gas testing does not replace source isolation or engineering controls."
        ),
    },
    {
        "source": "Best industry practice — Permit to Work (PTW)",
        "topic": "permit to work",
        "text": (
            "A PTW system should define the exact job, location, hazards, isolations, required controls, responsible "
            "persons, validity period and simultaneous operations conflicts. Conduct a pre-job toolbox talk. Verify "
            "field conditions before authorization, suspend permits when conditions change, and close/cancel permits "
            "after the area is left in a safe condition."
        ),
    },
    {
        "source": "Best industry practice — PPE",
        "topic": "PPE",
        "text": (
            "Select PPE from the hazard assessment and ensure compatibility with the task and other PPE. Inspect before "
            "use, maintain/replace defective equipment, train users in limitations and correct use, and ensure PPE is "
            "stored hygienically. PPE is the last line of defense and should not replace feasible higher-level controls."
        ),
    },
]

# ---------- Model / RAG helpers ----------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    # Small, CPU-friendly multilingual model suitable for Streamlit Cloud.
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text, max_chars=900, overlap=120):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def docs_to_chunks(docs):
    chunks = []
    for d in docs:
        for i, c in enumerate(chunk_text(d["text"])):
            chunks.append({
                "source": d["source"],
                "topic": d.get("topic", ""),
                "text": c,
                "chunk_id": i,
            })
    return chunks

def extract_uploaded_files(files):
    docs = []
    for f in files:
        suffix = Path(f.name).suffix.lower()
        try:
            if suffix == ".pdf":
                reader = PdfReader(f)
                text = "\n".join((p.extract_text() or "") for p in reader.pages)
            else:
                text = f.read().decode("utf-8", errors="ignore")
            if text.strip():
                docs.append({
                    "source": f"User uploaded: {f.name}",
                    "topic": "uploaded",
                    "text": text,
                })
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")
    return docs

@st.cache_resource(show_spinner=False)
def build_index(doc_signature):
    # doc_signature is JSON so Streamlit can cache based on knowledge content.
    docs = json.loads(doc_signature)
    chunks = docs_to_chunks(docs)
    model = load_embedding_model()
    embeddings = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

def retrieve(job, index, chunks, k=12):
    model = load_embedding_model()
    q = model.encode([job], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    scores, ids = index.search(q, min(k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            item = dict(chunks[int(idx)])
            item["score"] = float(score)
            results.append(item)
    return results

def get_groq_client():
    key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    if not key:
        return None
    return Groq(api_key=key)

def generate_checklist(job, retrieved, site_context, risk_tolerance):
    client = get_groq_client()
    if client is None:
        return None

    context = "\n\n".join(
        f"[SOURCE: {r['source']} | similarity={r['score']:.3f}]\n{r['text']}"
        for r in retrieved
    )

    system = """You are a senior occupational health, safety and process-safety professional.
Generate a practical job safety checklist from the supplied RAG context.
Rules:
1. Do NOT invent standards, clause numbers, legal requirements, limits, or citations.
2. Use only requirements supported by the supplied context plus broadly accepted safety practice.
3. Distinguish "must/required" only when the source context clearly supports it; otherwise use "verify/ensure".
4. Produce checklist items covering BEFORE WORK, DURING WORK, and CLOSE-OUT.
5. Include a risk/criticality field: CRITICAL, HIGH, MEDIUM, or LOW.
6. Include a source field for every item, using the source names supplied in context.
7. If job-specific details are missing, make conservative assumptions and identify what should be confirmed.
8. For high-risk jobs, include stop-work triggers.
9. Return ONLY valid JSON matching the requested schema."""

    user = f"""
JOB:
{job}

SITE / CONTEXT:
{site_context or "Not provided"}

RISK TOLERANCE:
{risk_tolerance}

RAG CONTEXT:
{context}

Return JSON:
{{
  "job_summary": "brief summary",
  "assumptions": ["..."],
  "pre_work": [
    {{"id":"PRE-01","item":"...","criticality":"CRITICAL|HIGH|MEDIUM|LOW","source":"..."}}
  ],
  "during_work": [
    {{"id":"DUR-01","item":"...","criticality":"CRITICAL|HIGH|MEDIUM|LOW","source":"..."}}
  ],
  "close_out": [
    {{"id":"CLO-01","item":"...","criticality":"CRITICAL|HIGH|MEDIUM|LOW","source":"..."}}
  ],
  "stop_work_triggers": ["..."],
  "key_verifications": ["..."],
  "disclaimer": "..."
}}
"""

    model = st.session_state.get("groq_model", "openai/gpt-oss-120b")
    response = client.chat.completions.create(
        model=model,
        temperature=0.15,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(response.choices[0].message.content)

# ---------- Session state ----------
if "knowledge_docs" not in st.session_state:
    st.session_state.knowledge_docs = list(SEED_DOCS)

if "checklist" not in st.session_state:
    st.session_state.checklist = None

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.caption("RAG + FAISS + Groq")
    st.session_state.groq_model = st.selectbox(
        "Groq model",
        [
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
],
        index=0,
    )
    top_k = st.slider("RAG results", 5, 20, 12)
    risk_tolerance = st.selectbox(
        "Checklist posture",
        ["Conservative / safety-first", "Balanced", "Minimum practical controls"],
    )
    st.markdown("---")
    st.markdown("### 📚 Add your own standards")
    uploads = st.file_uploader(
        "Upload PDF/TXT/MD safety documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Examples: company HSE procedures, PTW manual, emergency plan, client standards, approved codes/guidelines.",
    )
    if uploads:
        uploaded_docs = extract_uploaded_files(uploads)
        if uploaded_docs:
            # Replace duplicates by source name.
            existing = {d["source"]: d for d in st.session_state.knowledge_docs}
            for d in uploaded_docs:
                existing[d["source"]] = d
            st.session_state.knowledge_docs = list(existing.values())
            st.success(f"Knowledge base: {len(st.session_state.knowledge_docs)} documents")

    if st.button("Reset to built-in knowledge", use_container_width=True):
        st.session_state.knowledge_docs = list(SEED_DOCS)
        st.session_state.checklist = None
        st.rerun()

    st.markdown("---")
    st.caption(
        "Important: This tool is a safety planning aid. It does not replace "
        "a competent HSE professional, approved PTW/JSA/JHA, applicable law, "
        "company procedures, or the authoritative edition of a standard/code."
    )

# ---------- Main UI ----------
st.markdown(
    """
    <div class="hero">
      <h1>🦺 Safety RAG — Job Safety Checklist</h1>
      <p>Describe the job. Retrieve relevant safety controls. Generate a practical,
      traceable checklist for pre-job, execution and close-out.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([2.2, 1])
with col1:
    job = st.text_area(
        "Job / Activity",
        placeholder=(
            "Example: Hot work welding on a natural-gas pipeline inside an operating gas facility"
        ),
        height=120,
    )
with col2:
    site_context = st.text_area(
        "Optional site / job context",
        placeholder=(
            "e.g., operating gas plant, hydrocarbon service, outdoor, night shift, "
            "contractor involved, nearby live line..."
        ),
        height=120,
    )

generate = st.button(
    "🔎 Retrieve requirements & generate checklist",
    type="primary",
    use_container_width=True,
)

if generate:
    if not job.strip():
        st.error("Please enter the job/activity first.")
        st.stop()

    docs_signature = json.dumps(st.session_state.knowledge_docs, sort_keys=True)
    with st.spinner("Building/searching the safety knowledge base with FAISS..."):
        index, chunks = build_index(docs_signature)
        retrieved = retrieve(job, index, chunks, top_k)

    st.session_state.retrieved = retrieved

    if get_groq_client() is None:
        st.error(
            "GROQ_API_KEY is not configured. Add it in Streamlit Cloud → "
            "App settings → Secrets, or set it in .streamlit/secrets.toml locally."
        )
        st.stop()

    with st.spinner("Groq is synthesizing the job-specific checklist..."):
        try:
            st.session_state.checklist = generate_checklist(
                job, retrieved, site_context, risk_tolerance
            )
        except Exception as e:
            st.error(f"Groq request failed: {e}")
            st.stop()

# ---------- Results ----------
if st.session_state.checklist:
    data = st.session_state.checklist
    retrieved = st.session_state.get("retrieved", [])

    st.markdown("## 📋 Generated Safety Checklist")
    st.info(
        "Tick each control only after it has actually been verified/completed. "
        "The checklist is generated from the retrieved knowledge and should be reviewed "
        "by the responsible competent person before authorization."
    )

    st.markdown(f"**Job:** {job}")
    if data.get("job_summary"):
        st.markdown(f"**Safety planning summary:** {data['job_summary']}")

    if data.get("assumptions"):
        with st.expander("Assumptions / items requiring confirmation"):
            for x in data["assumptions"]:
                st.warning(x)

    # Use stable session keys so checkboxes persist during interaction.
    sections = [
        ("PRE-WORK", "pre_work"),
        ("DURING WORK", "during_work"),
        ("CLOSE-OUT", "close_out"),
    ]

    all_items = []
    for title, key in sections:
        items = data.get(key, [])
        st.markdown(f"### {title}")
        if not items:
            st.caption("No items generated for this section.")
            continue

        for item in items:
            item_id = item.get("id", f"{key}-{len(all_items)+1}")
            criticality = item.get("criticality", "MEDIUM").upper()
            label = f"{item_id} — {item.get('item','')}"
            checked = st.checkbox(
                label,
                key=f"check_{item_id}",
            )
            c1, c2 = st.columns([1, 5])
            with c1:
                st.markdown(f"**{criticality}**")
            with c2:
                st.caption(f"Source: {item.get('source','Not specified')}")
            all_items.append((item_id, checked))

    st.markdown("### ⛔ Stop-work triggers")
    for x in data.get("stop_work_triggers", []):
        st.warning(x)

    st.markdown("### 🔍 Key verifications")
    for x in data.get("key_verifications", []):
        st.markdown(f"- {x}")

    completed = sum(1 for _, checked in all_items if checked)
    total = len(all_items)
    if total:
        pct = completed / total
        st.progress(pct)
        st.metric("Checklist completion", f"{completed}/{total}", f"{pct:.0%}")

        if completed == total:
            st.success("All generated checklist items are marked complete.")
        else:
            st.warning("Do not treat an unchecked item as completed.")

    # Download a simple text record of the generated checklist.
    lines = [
        "SAFETY RAG — JOB SAFETY CHECKLIST",
        f"Job: {job}",
        f"Site context: {site_context}",
        "",
    ]
    for title, key in sections:
        lines.append(title)
        lines.append("-" * len(title))
        for item in data.get(key, []):
            item_id = item.get("id", "")
            checked = st.session_state.get(f"check_{item_id}", False)
            lines.append(
                f"[{'X' if checked else ' '}] {item_id} | "
                f"{item.get('criticality','')} | {item.get('item','')} | "
                f"Source: {item.get('source','')}"
            )
        lines.append("")
    lines.append("STOP-WORK TRIGGERS")
    for x in data.get("stop_work_triggers", []):
        lines.append(f"- {x}")
    lines.append("")
    lines.append("DISCLAIMER")
    lines.append(data.get("disclaimer", ""))

    st.download_button(
        "⬇️ Download completed checklist (TXT)",
        data="\n".join(lines),
        file_name="safety_job_checklist.txt",
        mime="text/plain",
        use_container_width=True,
    )

    with st.expander("📚 RAG evidence retrieved"):
        for i, r in enumerate(retrieved, 1):
            st.markdown(
                f"**{i}. {r['source']}** — similarity {r['score']:.3f}"
            )
            st.write(r["text"])

else:
    st.markdown("### How it works")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1", "Describe job")
    c2.metric("2", "FAISS retrieval")
    c3.metric("3", "Groq synthesis")
    c4.metric("4", "Tick / verify")

    st.markdown(
        """
        **Recommended production workflow**
        1. Upload your organization's approved HSE procedures, PTW/JSA requirements,
           emergency procedures and client standards.
        2. Enter a specific job rather than a broad phrase.
        3. Review the retrieved evidence.
        4. Have the competent HSE/job supervisor validate the generated checklist.
        5. Complete the checklist in the field and retain the completed record according
           to your document-control requirements.
        """
    )
