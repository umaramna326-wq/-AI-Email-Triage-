import streamlit as st
import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from groq import Groq

# --- SYSTEM SYSTEM PROMPT CONSTANT ---
TRIAGE_SYSTEM_PROMPT = """You are an advanced enterprise AI Email Triage and Operations Assistant. 
Your job is to analyze incoming emails, extract key operational metadata, evaluate business urgency, isolate explicit action items, and draft contextual replies.

You must evaluate business urgency based on these strict guidelines:
- Critical: active system failure, security threat, or severe revenue block.
- High: frustrated customer, deadline within 24 hours, or executive request.
- Medium: routine questions, scheduling updates.
- Low: newsletters, automated alerts, open-ended feedback.

You must extract key entities. If an entity is completely absent, return null. Never guess or hallucinate parameters.

Return your analysis strictly as a valid JSON object matching the defined schema below. Do not include any conversational preamble, markdown blocks, or explanation outside of the JSON structure.

REQUIRED JSON SCHEMA:
{
  "metadata": {
    "category": "Sales Inquiry | Technical Support | Billing & Invoicing | Partnership & Vendor | General / Informational",
    "urgency": "Critical | High | Medium | Low",
    "justification": "A single sentence explaining why this urgency tier was assigned."
  },
  "extracted_entities": {
    "sender_name": "string or null",
    "company": "string or null",
    "associated_ids": ["string"],
    "deadlines_or_dates": ["string"]
  },
  "task_list": ["short punchy action item"],
  "draft_response": {
    "subject": "string",
    "body": "string"
  }
}"""

# --- DEMO DATA CONSTANT ---
DEMO_EMAILS = [
    {
        "filename": "demo_support_outage.txt",
        "content": "Subject: EMERGENCY: Production database down!\nFrom: chief_ops@enterpriseclient.com\n\nOur system is throwing 500 errors and our entire checkout pipeline has been completely stuck since 10 minutes ago. This is costing us thousands every minute. Fix this immediately! Ticket ID: #98342."
    },
    {
        "filename": "demo_invoice_dispute.txt",
        "content": "Subject: Discrepancy on Invoice INV-2026-889\nFrom: accounts@vendorcorp.net\n\nHello, I noticed we were cross-charged twice for the server maintenance line item on our latest invoice INV-2026-889 dated Nov 4. Please adjust the credit balance before the auto-pay processes this Friday."
    },
    {
        "filename": "demo_sales_quote.txt",
        "content": "Subject: Enterprise Tier RFQ / Pricing Request\nFrom: sarah.jenkins@globalretail.com\n\nHi Sales Team, our division is looking to onboard 450 seats onto your cloud architecture next quarter. Could you provide a formal corporate price quote and a draft SLA by next Tuesday? Thanks, Sarah Jenkins."
    },
    {
        "filename": "demo_vendor_partnership.txt",
        "content": "Subject: Strategic API Integration Partnership Proposal\nFrom: dylan@connectapi.io\n\nDear Team, I represent ConnectAPI. We see an open opportunity to link our webhook modules with your native delivery core. Let me know if your tech ecosystem lead has 15 minutes for an intro call next Thursday."
    },
    {
        "filename": "demo_thank_you.txt",
        "content": "Subject: Great work on the onboarding session!\nFrom: tim@startupgrow.io\n\nJust wanted to send a quick note saying thanks to your onboarding agent for guiding us through the platform setup yesterday. The team loves the interface!"
    }
]

# --- CORE BUSINESS LOGIC FUNCTIONS ---

def get_groq_client() -> Optional[Groq]:
    """Retrieves API Key from secrets or environment variables safely."""
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

def call_llm_triage(client: Groq, model: str, email_body: str, retry: bool = True) -> Dict[str, Any]:
    """Requests structural JSON analysis from the chosen Groq inference engine model."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Please analyze and triage this email:\n\n{email_body}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices.message.content)
    except Exception as e:
        if retry:
            return call_llm_triage(client, model, email_body, retry=False)
        raise RuntimeError(f"LLM Processing Failure: {str(e)}")

def parse_uploaded_files(uploaded_files: List[Any]) -> List[Dict[str, str]]:
    """Normalizes uploaded raw byte structures into operational text fields."""
    parsed_emails = []
    for file in uploaded_files:
        filename = file.name
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(file)
                content_col = next((col for col in df.columns if "content" in col.lower() or "text" in col.lower() or "body" in col.lower()), df.columns[0])
                for idx, row in df.iterrows():
                    parsed_emails.append({
                        "filename": f"{filename}_row_{idx}",
                        "content": str(row[content_col])
                    })
            else:
                raw_text = file.read().decode("utf-8", errors="ignore")
                parsed_emails.append({"filename": filename, "content": raw_text})
        except Exception as e:
            st.error(f"Could not parse file {filename}: {e}")
    return parsed_emails

# --- APPLICATION INTERFACE CONFIGURATION ---

st.set_page_config(page_title="AI Email Triage Simulator", layout="wide", page_icon="📨")

if "triage_results" not in st.session_state:
    st.session_state.triage_results = {}
if "review_states" not in st.session_state:
    st.session_state.review_states = {}
if "edited_replies" not in st.session_state:
    st.session_state.edited_replies = {}

groq_client = get_groq_client()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.title("🛠️ Operations Center")
    
    if not groq_client:
        st.error("❌ GROQ_API_KEY Missing! Add it to Streamlit Secrets or your local environment vars.")
        st.stop()
        
    st.subheader("Model Parameters")
    selected_model = st.selectbox(
        "Inference Architecture Engine",
        ["llama-3.3-70b-versatile", "llama3-8b-8192"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("Dashboard Global Filters")
    
    categories_pool = ["Sales Inquiry", "Technical Support", "Billing & Invoicing", "Partnership & Vendor", "General / Informational"]
    urgencies_pool = ["Critical", "High", "Medium", "Low"]
    
    filter_categories = st.multiselect("Filter Categories", categories_pool, default=categories_pool)
    filter_urgencies = st.multiselect("Filter Urgencies", urgencies_pool, default=urgencies_pool)

# --- MAIN INGESTION LAYOUT ---
st.title("📨 AI Email Triage Simulator")
st.write("Route corporate mail vectors, run contextual entity extractions, and evaluate automated auto-replies seamlessly.")

ingest_tab, dash_tab = st.tabs(["📥 Ingestion & Intake Core", "📊 Live Operations Dashboard"])

with ingest_tab:
    col_input, col_preset = st.columns(2)
    
    with col_input:
        st.subheader("Batch File Uploads")
        uploaded_files = st.file_uploader(
            "Upload files (.txt, .eml, .csv)", 
            type=["txt", "eml", "csv"], 
            accept_multiple_files=True
        )
        
        st.subheader("Manual Data Entry Override")
        manual_subject = st.text_input("Manual Email Subject", "Inquiry regarding infrastructure options")
        manual_body = st.text_area("Manual Email Body Text", "Type or paste your unstructured email body contents here...")
        
        execute_manual = st.button("⚡ Process Manual Input Only")
        
    with col_preset:
        st.subheader("Simulation Presets")
        st.write("Inject our production test-suite pipeline into state vectors to run instant evaluations.")
        load_demo = st.button("🚀 Load Sample Test Emails Matrix")

    emails_to_process = []
    if load_demo:
        emails_to_process.extend(DEMO_EMAILS)
    if execute_manual and manual_body.strip():
        emails_to_process.append({
            "filename": f"manual_entry_{pd.Timestamp.now().strftime('%H%M%S')}.txt",
            "content": f"Subject: {manual_subject}\n\n{manual_body}"
        })
    if uploaded_files:
        emails_to_process.extend(parse_uploaded_files(uploaded_files))

    if emails_to_process:
        st.markdown("---")
        st.subheader("Execution Progress Log")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        success_count = 0
        for idx, email_meta in enumerate(emails_to_process):
            f_name = email_meta["filename"]
            status_text.write(f"⚙️ Triaging: `{f_name}` via {selected_model}...")
            unique_key = f"{f_name}_{hash(email_meta['content'])}"
            
            try:
                parsed_json = call_llm_triage(groq_client, selected_model, email_meta["content"])
                st.session_state.triage_results[unique_key] = {
                    "id": unique_key,
                    "filename": f_name,
                    "raw_content": email_meta["content"],
                    "analysis": parsed_json
                }
                if unique_key not in st.session_state.review_states:
                    st.session_state.review_states[unique_key] = "Pending"
                if unique_key not in st.session_state.edited_replies:
                    st.session_state.edited_replies[unique_key] = parsed_json.get("draft_response", {}).get("body", "")
                success_count += 1
            except Exception as exc:
                st.error(f"❌ Failed to extract metadata safe parameters for `{f_name}`: {exc}")
                
            progress_bar.progress((idx + 1) / len(emails_to_process))
            
        status_text.empty()
        progress_bar.empty()
        st.success(f"Successfully processed {success_count} item(s) through Groq structural pipelines!")

# --- DASHBOARD & ANALYTICS LAYOUT ---
with dash_tab:
    compiled_rows = []
    badge_colors = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
    
    for k, record in list(st.session_state.triage_results.items()):
        analysis = record["analysis"]
        meta = analysis.get("metadata", {})
        category = meta.get("category", "General / Informational")
        urgency = meta.get("urgency", "Low")
        
        if category in filter_categories and urgency in filter_urgencies:
            entities = analysis.get("extracted_entities", {})
            tasks = analysis.get("task_list", [])
            
            compiled_rows.append({
                "ID": record["id"],
                "File Indicator": record["filename"],
                "Category": category,
                "Urgency Tier": f"{badge_colors.get(urgency, '⚪')} {urgency}",
                "Inferred Sender": entities.get("sender_name") or "Unknown",
                "Action Items Count": len(tasks),
                "Workflow Status": st.session_state.review_states[record["id"]],
                "_raw_urgency_key": urgency
            })

    if not compiled_rows:
        st.info("No logs passed current visibility filter specifications or queue tracking arrays are empty.")
    else:
        df_dash = pd.DataFrame(compiled_rows)
        st.subheader("Operational Health Overview")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Total Items Managed", len(df_dash))
        metric_col2.metric("Critical Alerts Flagged", len(df_dash[df_dash["_raw_urgency_key"] == "Critical"]))
        metric_col3.metric("Actionable Core Tasks Required", int(df_dash["Action Items Count"].sum()))
        
        st.markdown("---")
        st.subheader("Master Verification Matrix")
        st.dataframe(df_dash.drop(columns=["ID", "_raw_urgency_key"]), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("Granular Queue Inspector & Human-In-The-Loop Validation")
        
        for idx, row in df_dash.iterrows():
            record_id = row["ID"]
            orig_record = st.session_state.triage_results[record_id]
            data_payload = orig_record["analysis"]
            expander_label = f"📝 Inspect File: {row['File Indicator']} | Tier: {row['Urgency Tier']} | Status: {row['Workflow Status']}"
            
            with st.expander(expander_label):
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown("**Original Email Message Payload:**")
                    st.code(orig_record["raw_content"], language="text")
                    st.markdown("**Extracted Identity Properties:**")
                    st.json(data_payload.get("extracted_entities", {}))
                with col_right:
                    st.markdown(f"**AI Urgency Assessment Rationale:**")
                    st.info(data_payload.get("metadata", {}).get("justification", "No justification array recorded."))
                    st.markdown("**Isolated Action Item Tasks List:**")
                    for t in data_payload.get("task_list", []):
                        st.markdown(f"- [ ] {t}")
                    st.markdown("**Contextual Draft Auto-Reply Review:**")
                    current_reply_text = st.text_area(
                        "Modify Generated Text Response Output",
                        value=st.session_state.edited_replies[record_id],
                        key=f"text_area_{record_id}",
                        height=180
                    )
                    st.session_state.edited_replies[record_id] = current_reply_text
                    
                    btn_col1, btn_col2 = st.columns(2)
                    if btn_col1.button("✅ Approve Dispatch Line", key=f"apprv_{record_id}"):
                        st.session_state.review_states[record_id] = "Approved"
                        st.rerun()
                    if btn_col2.button("❌ Reject / Flag Review Out", key=f"rjct_{record_id}"):
                        st.session_state.review_states[record_id] = "Rejected"
                        st.rerun()

        st.markdown("---")
        st.subheader("Data Export Pipeline")
        export_payload_list = []
        for row_item in compiled_rows:
            rec = st.session_state.triage_results[row_item["ID"]]
            payload = rec["analysis"].copy()
            payload["filename"] = rec["filename"]
            payload["human_workflow_status"] = st.session_state.review_states[row_item["ID"]]
            payload["final_approved_reply_body"] = st.session_state.edited_replies[row_item["ID"]]
            export_payload_list.append(payload)
            
        col_dl1, col_dl2 = st.columns(2)
        json_string = json.dumps(export_payload_list, indent=2)
        col_dl1.download_button(
            label="📥 Download Data Records Dataset as JSON",
            data=json_string,
            file_name="triage_export_matrix.json",
            mime="application/json"
        )
        
        flat_csv_list = []
        for item in export_payload_list:
            flat_csv_list.append({
                "filename": item.get("filename"),
                "category": item.get("metadata", {}).get("category"),
                "urgency": item.get("metadata", {}).get("urgency"),
                "sender_name": item.get("extracted_entities", {}).get("sender_name"),
                "company": item.get("extracted_entities", {}).get("company"),
                "task_count": len(item.get("task_list", [])),
                "workflow_status": item.get("human_workflow_status")
            })
        df_csv = pd.DataFrame(flat_csv_list)
        col_dl2.download_button(
            label="📥 Download Data Records Summary as CSV",
            data=df_csv.to_csv(index=False),
            file_name="triage_export_summary.csv",
            mime="text/csv"
        )
