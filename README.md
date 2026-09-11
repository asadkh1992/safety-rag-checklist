# Safety RAG — Job Safety Checklist Generator

A Streamlit application that accepts a **Job / Activity** as input, retrieves relevant safety requirements using **RAG + FAISS**, and uses the **Groq API** to generate a structured, job-specific safety checklist.

The user can tick/untick checklist controls, review the RAG evidence, upload additional company/client safety documents, and download the completed checklist as a TXT record.

## Architecture

```text
User enters JOB
       |
       v
Sentence-Transformers embedding
       |
       v
FAISS similarity search
       |
       v
Relevant safety knowledge chunks
       |
       v
Groq LLM (JSON checklist generation)
       |
       v
Streamlit interactive checklist
       |
       +--> Tick / Untick
       +--> Stop-work triggers
       +--> Key verifications
       +--> RAG evidence
       +--> Download TXT
```

## Included knowledge

The app contains concise, original safety-principle summaries covering topics such as:

- ISO 45001-style OH&S management principles
- OSHA job-hazard-analysis / hazard-communication principles
- Confined spaces
- Lockout/tagout
- Hot work
- Electrical safety / arc flash
- Oil & gas drilling/service safety
- Hazardous areas
- Work at height
- Lifting and rigging
- Excavation
- Pressure testing
- Gas testing
- Permit to Work
- PPE
- Construction safety

**Important:** The built-in content is deliberately a summary/knowledge layer, not a copy of copyrighted standards. It should not be treated as the authoritative text of a standard or as legal advice.

## Strongly recommended production enhancement

Upload your organization's approved documents through the sidebar, for example:

- HSE Manual
- Permit-to-Work Procedure
- JSA/JHA Procedure
- LOTO Procedure
- Confined Space Procedure
- Hot Work Procedure
- Electrical Safety Procedure
- Lifting Plan / Rigging Procedure
- Work-at-Height Procedure
- Gas Testing Procedure
- Emergency Response Plan
- Client HSE requirements
- Approved corporate standards
- Applicable regulatory guidance

This allows the RAG layer to retrieve company-specific controls in addition to the built-in knowledge.

## Files for Streamlit Community Cloud

Repository root:

```text
safety-rag-checklist/
├── app.py
├── requirements.txt
└── README.md
```

No `packages.txt` is required by this version.

## 1. Create the GitHub repository

Create a new GitHub repository, for example:

```text
safety-rag-checklist
```

Upload:

- `app.py`
- `requirements.txt`
- `README.md`

Keep the repository structure simple.

## 2. Get a Groq API key

Create/sign in to your Groq account and generate an API key.

Do **not** place the API key inside `app.py` or commit it to GitHub.

## 3. Test locally (recommended)

Install Python 3.12.

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install:

```powershell
pip install -r requirements.txt
```

Set the API key for the current terminal session:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Run:

```powershell
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## 4. Optional local secrets file

Instead of setting an environment variable, create:

```text
.streamlit/
└── secrets.toml
```

with:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

Add `.streamlit/secrets.toml` to `.gitignore`.

**Never commit your API key to GitHub.**

## 5. Deploy to Streamlit Community Cloud

1. Go to Streamlit Community Cloud.
2. Sign in with GitHub.
3. Click **Create app**.
4. Select your GitHub repository.
5. Select the branch, normally `main`.
6. Select `app.py` as the entrypoint.
7. Open **Advanced settings**.
8. Select Python 3.12 if available.
9. In **Secrets**, enter:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

10. Save/deploy.
11. Wait for dependency installation and application startup.
12. Open the generated `streamlit.app` URL.

## 6. First production test

Test with jobs such as:

```text
Hot work welding on a natural gas pipeline inside an operating gas facility
```

```text
Entry into a permit-required confined space
```

```text
Electrical maintenance on a 415 V MCC
```

```text
Lifting a gas valve assembly using a mobile crane
```

```text
Excavation near an existing underground gas pipeline
```

```text
Pressure testing of a gas pipeline section
```

After generation:

1. Check the job summary.
2. Review assumptions.
3. Review the RAG evidence.
4. Validate every control against the applicable company procedure and approved standard.
5. Tick only controls that have actually been verified/completed.
6. Review stop-work triggers.
7. Download the completed checklist if required.

## 7. How RAG works

The app does not simply ask the LLM to invent a checklist.

It first:

1. Splits the safety knowledge into chunks.
2. Converts each chunk into an embedding using `all-MiniLM-L6-v2`.
3. Stores embeddings in a FAISS inner-product index.
4. Embeds the user's job description.
5. Retrieves the most similar safety requirements.
6. Sends those retrieved requirements to Groq.
7. Groq generates the checklist in a controlled JSON schema.
8. Streamlit renders the checklist interactively.

## 8. Adding company documents

Use the sidebar:

**Add your own standards → Upload PDF/TXT/MD safety documents**

The uploaded documents are extracted and added to the current session's knowledge base.

For a serious organizational deployment, consider moving the knowledge base to persistent storage rather than relying on runtime uploads.

## 9. Recommended future production architecture

For a larger enterprise version:

```text
Streamlit UI
     |
     +--> Authentication / RBAC
     |
     +--> Job database
     |
     +--> Document management
     |
     +--> Persistent vector DB
     |       (FAISS / pgvector / Qdrant)
     |
     +--> RAG retrieval
     |
     +--> Groq
     |
     +--> Audit trail
     |
     +--> PDF/Excel checklist
     |
     +--> Management dashboard
```

Recommended additional features:

- User login / role-based access
- Job number / work order
- Site / plant / department
- Permit number
- Contractor name
- Supervisor / HSE officer
- Date/time
- Weather and environmental conditions
- SIMOPS screening
- Risk matrix
- JSA/JHA generation
- Permit-to-work linkage
- Digital signature
- PDF checklist
- Excel export
- Database/audit trail
- Version-controlled safety knowledge
- Document expiry/review dates
- Mandatory critical controls
- Stop-work escalation
- Management dashboard
- Analytics for recurring hazards
- Separate jurisdiction/company profiles

## 10. Safety and legal disclaimer

This application is a **decision-support and safety-planning tool**.

It does not replace:

- applicable law or regulation
- the current authoritative edition of a standard/code
- company HSE procedures
- permit-to-work requirements
- approved JSA/JHA
- competent-person judgment
- site-specific risk assessment
- emergency/rescue planning

For high-risk activities, the generated checklist must be reviewed and approved by the responsible competent person/HSE authority before work starts.

## Troubleshooting

### `GROQ_API_KEY is not configured`

Add the secret in Streamlit Community Cloud:

**App → Settings → Secrets**

Use:

```toml
GROQ_API_KEY = "YOUR_KEY"
```

### FAISS installation error

Make sure the repository is using the supplied `requirements.txt` and Python 3.12. If dependency resolution changes in the future, update the FAISS package to a compatible release.

### PDF produces no useful text

Some PDFs are scanned images rather than text PDFs. OCR can be added as a future enhancement.

### App becomes slow

The first run downloads the embedding model. Streamlit caches the model and FAISS index. For a large enterprise corpus, use a persistent vector database and pre-build embeddings.

## License / ownership

Adapt this project to your organization's policies. Review third-party standards' licensing/copyright terms before distributing their full text or storing proprietary copies.
