import io
import json
import re
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PyPDF2 import PdfReader

try:
    import spacy
    SPACY_ENABLED = True
except ImportError:
    spacy = None
    SPACY_ENABLED = False

try:
    from fpdf import FPDF
    PDF_ENABLED = True
except ImportError:
    FPDF = None
    PDF_ENABLED = False

st.set_page_config(
    page_title="Sensitive Data Detection & Compliance Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Sensitive Data Detection & Compliance Assistant"
APP_SUBTITLE = "AI-powered compliance analysis for PDFs, CSVs and TXT documents."
ACCENT = "#4F46E5"
SECONDARY = "#7C3AED"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"
BACKGROUND = "#0f172a"
CARD = "#111827"
TEXT = "#E5E7EB"

FILE_TYPES = {
    ".pdf": "PDF Document",
    ".csv": "CSV Spreadsheet",
    ".txt": "Text File",
}

SENSITIVE_PATTERNS = [
    (
        "Credit Card",
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
        "Financial",
        "Critical",
        "Protect payment information and remove from shared documents.",
    ),
    (
        "Social Security Number",
        r"\b\d{3}-\d{2}-\d{4}\b",
        "Identity",
        "Critical",
        "Mask or redact Social Security Numbers in all reports.",
    ),
    (
        "Email Address",
        r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
        "Contact",
        "High",
        "Avoid publishing raw email addresses in external documents.",
    ),
    (
        "Phone Number",
        r"\b(?:\+?\d[\d .-]{8,}\d)\b",
        "Contact",
        "High",
        "Do not share phone numbers without explicit consent.",
    ),
    (
        "Passport Number",
        r"\b[A-Z]{1}[0-9]{7,8}\b",
        "Identity",
        "High",
        "Treat passport numbers as sensitive identity data.",
    ),
    (
        "Date of Birth",
        r"\b(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12][0-9]|3[01])\b",
        "Identity",
        "Medium",
        "Avoid sharing dates of birth in open reports.",
    ),
    (
        "IP Address",
        r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b",
        "Technical",
        "Medium",
        "Log files and server exports should conceal IP addresses.",
    ),
    (
        "Routing Number",
        r"\b\d{9}\b",
        "Financial",
        "Critical",
        "Bank routing numbers are sensitive financial identifiers.",
    ),
    (
        "IBAN",
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{13,30}\b",
        "Financial",
        "Critical",
        "International bank account numbers require strict handling.",
    ),
    (
        "Password Pattern",
        r"\b(password|pwd|pass|secret)\s*[:=]\s*[^\s]+\b",
        "Credentials",
        "Critical",
        "Never store or share passwords in plain text.",
    ),
]

CSV_HEADER_RISKS = {
    "password": ("Credentials", "Critical", "Passwords in headers may indicate insecure storage."),
    "ssn": ("Identity", "Critical", "SSN headers identify sensitive personal data."),
    "credit card": ("Financial", "Critical", "Payment card labels require PCI compliance."),
    "email": ("Contact", "High", "Email headers may expose user contact details."),
    "phone": ("Contact", "High", "Phone number headers can expose private details."),
    "dob": ("Identity", "Medium", "Date of birth fields should be handled carefully."),
}


@st.cache_resource
def load_spacy_model():
    if not SPACY_ENABLED:
        return None
    return spacy.blank("en")


def normalize_text(text: str) -> str:
    if not SPACY_ENABLED:
        return text
    nlp = load_spacy_model()
    if nlp is None:
        return text
    doc = nlp(text or "")
    return " ".join(token.text for token in doc)


@st.cache_data
def parse_document(file_bytes: bytes, filename: str) -> dict:
    extension = Path(filename).suffix.lower()
    preview = None
    text = ""
    pages = None
    rows = None
    file_type = FILE_TYPES.get(extension, "Document")

    if extension == ".pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = len(reader.pages)
        segments = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            segments.append(page_text)
        text = "\n\n".join(segments)
        preview = "\n\n".join(segments[:3])
    elif extension == ".csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
        rows = len(df)
        text = "\n".join(
            df.astype(str)
            .fillna("")
            .apply(lambda row: " ".join(row.values.astype(str)), axis=1)
            .tolist()
        )
        preview = df.head(8)
    else:
        text = file_bytes.decode("utf-8", errors="ignore")
        preview = "\n".join(text.splitlines()[:12])

    text = normalize_text(text)
    return {
        "file_name": filename,
        "file_type": file_type,
        "extension": extension,
        "file_size_kb": round(len(file_bytes) / 1024, 2),
        "pages": pages,
        "rows": rows,
        "text": text,
        "preview": preview,
    }


@st.cache_data
def detect_sensitive_entities(text: str, filename: str, file_type: str) -> pd.DataFrame:
    findings = []
    normalized_text = text or ""

    for label, pattern, category, risk, recommendation in SENSITIVE_PATTERNS:
        for match in re.finditer(pattern, normalized_text, flags=re.IGNORECASE):
            findings.append(
                {
                    "Entity": label,
                    "Detected Text": match.group().strip(),
                    "Risk Level": risk,
                    "Category": category,
                    "Recommendation": recommendation,
                }
            )

    if file_type == "CSV Spreadsheet":
        header_line = normalized_text.splitlines()[0] if normalized_text else ""
        header_tokens = [token.strip().lower() for token in re.split(r"[,;|\\t]", header_line)]
        for token in header_tokens:
            if token in CSV_HEADER_RISKS:
                category, risk, recommendation = CSV_HEADER_RISKS[token]
                findings.append(
                    {
                        "Entity": f"Header: {token.title()}",
                        "Detected Text": token,
                        "Risk Level": risk,
                        "Category": category,
                        "Recommendation": recommendation,
                    }
                )

    if not findings:
        return pd.DataFrame(
            [
                {
                    "Entity": "No sensitive entities detected",
                    "Detected Text": "N/A",
                    "Risk Level": "Low",
                    "Category": "Safe",
                    "Recommendation": "The document appears clean, but continue to validate manually.",
                }
            ]
        )

    entity_df = pd.DataFrame(findings)
    entity_df = entity_df.drop_duplicates(subset=["Entity", "Detected Text", "Risk Level"])
    entity_df = entity_df.reset_index(drop=True)
    return entity_df


@st.cache_data
def calculate_scores(entities: pd.DataFrame, pages: int | None, rows: int | None) -> dict:
    counts = entities["Risk Level"].value_counts().to_dict()
    critical = int(counts.get("Critical", 0))
    high = int(counts.get("High", 0))
    medium = int(counts.get("Medium", 0))
    low = int(counts.get("Low", 0))
    total = len(entities)

    risk_score = min(100, critical * 30 + high * 16 + medium * 8 + low * 3)
    compliance_score = max(0, 100 - risk_score * 0.7)
    compliance_score = int(compliance_score)
    risk_score = int(risk_score)

    if total == 0 or (total == 1 and entities.iloc[0]["Entity"] == "No sensitive entities detected"):
        compliance_score = 100
        risk_score = 5

    risk_rating = "Low"
    if risk_score >= 70:
        risk_rating = "Critical"
    elif risk_score >= 50:
        risk_rating = "High"
    elif risk_score >= 30:
        risk_rating = "Medium"

    return {
        "total_entities": total,
        "critical": critical,
        "high": high,
        "medium": medium,
        "low": low,
        "compliance_score": compliance_score,
        "risk_score": risk_score,
        "risk_rating": risk_rating,
        "pages": pages,
        "rows": rows,
    }


def create_gauge(value: int, title: str, subtitle: str, color: str) -> go.Figure:
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": f"{title}<br><span style='font-size:0.8em;color:#cbd5e1'>{subtitle}</span>"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": color},
                "bgcolor": "#1f2937",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#10b981"},
                    {"range": [30, 60], "color": "#f59e0b"},
                    {"range": [60, 100], "color": "#ef4444"},
                ],
            },
            number={"font": {"color": "#f8fafc", "size": 26}},
        )
    ).update_layout(
        paper_bgcolor=BACKGROUND,
        font={"color": "#f8fafc"},
        margin=dict(t=10, b=10, l=10, r=10),
    )


def build_timeline_chart() -> go.Figure:
    timeline_df = pd.DataFrame(
        [
            {"Stage": "Loading document", "Progress": 20},
            {"Stage": "Extracting text", "Progress": 40},
            {"Stage": "Detecting sensitive entities", "Progress": 60},
            {"Stage": "Risk assessment", "Progress": 80},
            {"Stage": "Generating compliance report", "Progress": 100},
        ]
    )
    fig = px.bar(
        timeline_df,
        x="Progress",
        y="Stage",
        orientation="h",
        color="Progress",
        color_continuous_scale=["#4F46E5", "#7C3AED"],
        title="Processing Workflow",
        labels={"Progress": "Completion"},
    )
    fig.update_layout(
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font_color=TEXT,
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def build_pdf_report(report_text: str) -> bytes | None:
    if not PDF_ENABLED or FPDF is None:
        return None

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_font("Helvetica", size=12)
    pdf.set_text_color(15, 23, 42)
    pdf.set_fill_color(247, 250, 252)

    for paragraph in report_text.split("\n\n"):
        pdf.set_font("Helvetica", style="B", size=12)
        title, _, body = paragraph.partition(":\n")
        if body:
            pdf.cell(0, 8, title, ln=True, fill=True)
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, body.strip())
            pdf.ln(2)
        else:
            pdf.multi_cell(0, 6, paragraph)
            pdf.ln(2)

    return pdf.output(dest="S").encode("latin-1", "replace")


def build_report(metrics: dict, preview_text: str) -> str:
    summary = (
        f"Executive Summary:\n"
        f"The document contains {metrics['total_entities']} sensitive items across {metrics['critical']} critical, {metrics['high']} high, {metrics['medium']} medium, and {metrics['low']} low severity findings."
    )
    compliance_status = "Compliant" if metrics["compliance_score"] >= 75 else "Needs Review"
    verdict = "Safe for distribution with standard controls." 
    if metrics["risk_score"] >= 70:
        verdict = "Immediate remediation required before distribution."
    elif metrics["risk_score"] >= 50:
        verdict = "Review the sensitive findings and apply controls."

    return "\n\n".join(
        [
            f"Executive Summary:\n{summary}",
            f"Detected Risks:\n- Critical: {metrics['critical']}\n- High: {metrics['high']}\n- Medium: {metrics['medium']}\n- Low: {metrics['low']}",
            f"Business Impact:\nSensitive data exposure can lead to compliance violations, legal risk, and customer trust loss. Review highlighted items and protect the document immediately.",
            f"Compliance Status:\n{compliance_status} with a compliance score of {metrics['compliance_score']}%.",
            f"Recommended Actions:\nConduct focused remediation on critical findings, redact or encrypt sensitive values, and validate the final document before sharing.",
            f"Overall Risk Rating:\n{metrics['risk_rating']} (Risk Score: {metrics['risk_score']}%)",
            f"Final Verdict:\n{verdict}",
        ]
    )


def style_entity_table(df: pd.DataFrame):
    color_map = {
        "Critical": "background-color: rgba(248, 113, 113, 0.25);",
        "High": "background-color: rgba(251, 191, 36, 0.2);",
        "Medium": "background-color: rgba(251, 191, 36, 0.12);",
        "Low": "background-color: rgba(34, 197, 94, 0.12);",
        "Safe": "background-color: rgba(59, 130, 246, 0.1);",
    }

    def row_style(value):
        return color_map.get(value, "")

    return df.style.format(precision=0).applymap(row_style, subset=["Risk Level"])


def inject_custom_styles():
    css = f"""
    <style>
    .css-1d391kg {{ background-color: {BACKGROUND}; }}
    .stApp {{ background: linear-gradient(180deg, #020617 0%, #0f172a 100%); color: {TEXT}; }}
    .hero-banner {{
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(79,70,229,0.95), rgba(124,58,237,0.88));
        box-shadow: 0 24px 60px rgba(79,70,229,0.18);
        color: #ffffff;
    }}
    .hero-banner h1 {{ font-size: 2.5rem; margin-bottom: 0.5rem; }}
    .hero-banner p {{ color: rgba(255,255,255,0.88); font-size: 1.05rem; }}
    .panel-card {{
        background: {CARD};
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 22px;
        padding: 1.35rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.35);
    }}
    .metric-label {{ color: #94a3b8; font-size: 0.9rem; }}
    .metric-value {{ font-size: 2rem; font-weight: 700; color: #f8fafc; }}
    .badge-pill {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #ffffff;
    }}
    .badge-success {{ background: #10b981; }}
    .badge-warning {{ background: #f59e0b; }}
    .badge-danger {{ background: #ef4444; }}
    .badge-info {{ background: #2563eb; }}
    .upload-card {{
        border-radius: 24px;
        background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(30,41,59,0.95));
        border: 1px dashed rgba(148, 163, 184, 0.42);
        padding: 2rem;
    }}
    .sidebar-content p, .sidebar-content li {{ color: #cbd5e1; }}
    .sidebar-title {{ color: #fff; margin-bottom: 0.5rem; }}
    .footer-note {{ color: #94a3b8; margin-top: 1.5rem; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_sidebar():
    st.sidebar.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    st.sidebar.markdown("# 📄 Compliance Assistant")
    st.sidebar.markdown("AI-driven document scanning for sensitive information and regulatory risk.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("## About Project")
    st.sidebar.markdown(
        "This application detects sensitive entities in PDF, CSV and TXT documents, calculates risk scores, and generates compliance-ready insights."
    )
    st.sidebar.markdown("## Features")
    st.sidebar.markdown(
        "- Secure file upload\n- Sensitive data detection\n- Risk scoring and compliance reporting\n- Downloadable CSV/JSON/TXT/PDF reports\n- Legacy sales analytics tab"
    )
    st.sidebar.markdown("## Technology Stack")
    st.sidebar.markdown("Python · Streamlit · spaCy · Regex · Pandas · Plotly · LLM")
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Developer")
    st.sidebar.markdown("**Created by:** Aarsh Pavashiya")
    st.sidebar.markdown("**Role:** AI Research Internship Project")
    st.sidebar.markdown("<div class='footer-note'>Build a modern AI portfolio-ready compliance dashboard with polished UX.</div>", unsafe_allow_html=True)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)


def analyze_document(file_bytes: bytes, filename: str):
    file_data = parse_document(file_bytes, filename)
    entity_df = detect_sensitive_entities(file_data["text"], file_data["file_name"], file_data["file_type"])
    metrics = calculate_scores(entity_df, file_data["pages"], file_data["rows"])
    report = build_report(metrics, file_data["text"])
    pdf_bytes = build_pdf_report(report)
    return {
        "file_data": file_data,
        "entities": entity_df,
        "metrics": metrics,
        "report": report,
        "pdf_bytes": pdf_bytes,
    }


def render_compliance_dashboard(results: dict):
    file_data = results["file_data"]
    metrics = results["metrics"]
    entity_df = results["entities"]
    report_text = results["report"]
    pdf_report = results["pdf_bytes"]

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("### Document Summary")
    cols = st.columns([1.5, 1, 1, 1])
    cols[0].markdown(f"**Filename**\n{file_data['file_name']}")
    cols[1].markdown(f"**Type**\n{file_data['file_type']}")
    cols[2].markdown(f"**Size**\n{file_data['file_size_kb']} KB")
    cols[3].markdown(
        f"**Status**\n<span class='badge-pill badge-success'>Ready to analyze</span>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>")

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Sensitive Entities", metrics["total_entities"], delta=None)
    metric_cols[1].metric("Compliance Score", f"{metrics['compliance_score']}%", delta=None)
    metric_cols[2].metric("Risk Score", f"{metrics['risk_score']}%", delta=None)
    metric_cols[3].metric("Processing Time", "Instant", delta=None)
    st.markdown("</div>")

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    chart_cols = st.columns(2)
    risk_breakdown = pd.DataFrame(
        {
            "Risk Level": ["Critical", "High", "Medium", "Low"],
            "Count": [
                metrics["critical"],
                metrics["high"],
                metrics["medium"],
                metrics["low"],
            ],
        }
    )
    chart_cols[0].plotly_chart(
        px.pie(
            risk_breakdown,
            names="Risk Level",
            values="Count",
            color_discrete_map={
                "Critical": DANGER,
                "High": WARNING,
                "Medium": SECONDARY,
                "Low": SUCCESS,
            },
            hole=0.55,
            title="Risk Distribution",
        ).update_layout(paper_bgcolor=BACKGROUND, font_color=TEXT, margin=dict(t=40, b=0, l=0, r=0)),
        use_container_width=True,
    )

    categories = entity_df["Category"].value_counts().reset_index()
    categories.columns = ["Category", "Count"]
    chart_cols[1].plotly_chart(
        px.bar(
            categories,
            x="Count",
            y="Category",
            orientation="h",
            color="Category",
            title="Sensitive Data Categories",
            color_discrete_sequence=[ACCENT, SECONDARY, DANGER, WARNING, SUCCESS],
        ).update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND, font_color=TEXT, margin=dict(t=40, b=10, l=0, r=0)),
        use_container_width=True,
    )
    st.markdown("</div>")

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    chart_cols = st.columns(3)
    chart_cols[0].plotly_chart(
        create_gauge(metrics["compliance_score"], "Compliance", "Higher is better", SUCCESS),
        use_container_width=True,
    )
    chart_cols[1].plotly_chart(
        create_gauge(metrics["risk_score"], "Risk", "Lower is safer", DANGER),
        use_container_width=True,
    )
    chart_cols[2].plotly_chart(
        build_timeline_chart(),
        use_container_width=True,
    )
    st.markdown("</div>")

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.subheader("Sensitive Entity Findings")
    styled_table = style_entity_table(entity_df)
    st.dataframe(styled_table, use_container_width=True)
    st.markdown("</div>")

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    with st.expander("Compliance Report Summary", expanded=True):
        st.markdown(report_text.replace("\n", "  \n"))
    st.markdown("</div>")

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.subheader("Download Compliance Reports")
    downloads = st.columns(4)
    downloads[0].download_button(
        "CSV Report",
        entity_df.to_csv(index=False).encode("utf-8"),
        "compliance_report.csv",
        "text/csv",
    )
    downloads[1].download_button(
        "JSON Report",
        json.dumps(entity_df.to_dict(orient="records"), indent=2).encode("utf-8"),
        "compliance_report.json",
        "application/json",
    )
    downloads[2].download_button(
        "TXT Report",
        report_text.encode("utf-8"),
        "compliance_report.txt",
        "text/plain",
    )
    if pdf_report is not None:
        downloads[3].download_button(
            "PDF Report",
            pdf_report,
            "compliance_report.pdf",
            "application/pdf",
        )
    else:
        downloads[3].markdown("PDF export unavailable (missing fpdf package)")
    st.markdown("</div>")

    try:
        st.toast("Analysis complete", icon="✅")
    except Exception:
        st.success("Analysis complete")


def render_hero():
    st.markdown(
        "<div class='hero-banner'>"
        f"<h1>{APP_TITLE}</h1>"
        f"<p>{APP_SUBTITLE}</p>"
        "<div style='display:flex;flex-wrap:wrap;gap:1rem;margin-top:1.5rem;'>"
        "<span class='badge-pill badge-info'>📄 Upload</span>"
        "<span class='badge-pill badge-info'>🔍 Detect Sensitive Data</span>"
        "<span class='badge-pill badge-info'>🛡 Risk Classification</span>"
        "<span class='badge-pill badge-info'>📊 Analytics</span>"
        "<span class='badge-pill badge-info'>📑 Compliance Report</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_legacy_sales_tab() -> None:
    st.markdown("# Legacy Sales Forecasting Dashboard")
    sample_path = Path("Dataset/Sample - Superstore.csv")
    if not sample_path.exists():
        st.warning("Legacy sales dataset not found. Please place the file in Dataset/Sample - Superstore.csv.")
        return

    df = pd.read_csv(sample_path)
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce", infer_datetime_format=True)
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Sales"])

    top_metrics = st.columns(4)
    top_metrics[0].metric("Total Sales", f"${df['Sales'].sum():,.2f}")
    top_metrics[1].metric("Orders", df["Order ID"].nunique())
    top_metrics[2].metric("Customers", df["Customer ID"].nunique())
    top_metrics[3].metric("Avg Order", f"${df['Sales'].mean():,.2f}")

    st.markdown("---")
    chart_cols = st.columns(2)
    category_sales = df.groupby("Category", as_index=False)["Sales"].sum()
    chart_cols[0].plotly_chart(
        px.bar(category_sales, x="Category", y="Sales", title="Sales by Category").update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND, font_color=TEXT),
        use_container_width=True,
    )
    region_sales = df.groupby("Region", as_index=False)["Sales"].sum()
    chart_cols[1].plotly_chart(
        px.pie(region_sales, names="Region", values="Sales", hole=0.45, title="Sales by Region").update_layout(paper_bgcolor=BACKGROUND, font_color=TEXT),
        use_container_width=True,
    )

    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
        monthly_df = df.dropna(subset=["Order Date", "Sales"]).copy()
        if not monthly_df.empty:
            monthly = (
                monthly_df.set_index("Order Date")["Sales"]
                .resample("M")
                .sum()
                .reset_index()
            )
            st.plotly_chart(
                px.line(monthly, x="Order Date", y="Sales", markers=True, title="Monthly Sales Trend").update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND, font_color=TEXT),
                use_container_width=True,
            )

    with st.expander("Show Raw Legacy Data", expanded=False):
        st.dataframe(df.head(250), use_container_width=True)


def main():
    inject_custom_styles()
    render_sidebar()

    tab1, tab2 = st.tabs(["Compliance AI", "Legacy Sales Insights"])

    with tab1:
        render_hero()
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.subheader("Upload Document")
        upload_col, preview_col = st.columns([2, 1])

        with upload_col:
            uploaded_file = st.file_uploader(
                "Select a PDF, CSV, or TXT file for compliance analysis",
                type=["pdf", "csv", "txt"],
                label_visibility="visible",
            )
            analyze_button = st.button("🚀 Analyze Document")

        with preview_col:
            st.markdown("**Supported Formats**")
            st.markdown("- PDF\n- CSV\n- TXT")
            st.markdown("**Upload Status**")
            if uploaded_file is not None:
                st.markdown(f"**Filename:** {uploaded_file.name}")
                st.markdown(f"**Type:** {Path(uploaded_file.name).suffix.upper()[1:]} file")
                st.markdown(f"**Size:** {round(len(uploaded_file.getvalue()) / 1024, 2)} KB")
            else:
                st.info("Upload a document to begin sensitive data detection.")

        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            file_data = parse_document(file_bytes, uploaded_file.name)
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            st.subheader("File Information")
            info_cols = st.columns(4)
            info_cols[0].markdown(f"**Type**\n{file_data['file_type']}")
            info_cols[1].markdown(f"**Size**\n{file_data['file_size_kb']} KB")
            info_cols[2].markdown(f"**Pages**\n{file_data['pages'] or 'N/A'}")
            info_cols[3].markdown(f"**Rows**\n{file_data['rows'] or 'N/A'}")
            st.markdown("</div>", unsafe_allow_html=True)

            if analyze_button:
                status = st.empty()
                progress_bar = st.progress(0)
                stages = [
                    "Loading document",
                    "Extracting text",
                    "Detecting sensitive entities",
                    "Risk assessment",
                    "Generating compliance report",
                    "Finalizing results",
                ]
                for index, stage in enumerate(stages):
                    status.info(f"{stage}...")
                    progress_bar.progress(int((index + 1) / len(stages) * 100))
                    time.sleep(0.1)

                with st.spinner("Analyzing the document with AI-powered compliance logic..."):
                    results = analyze_document(file_bytes, uploaded_file.name)
                    st.session_state["latest_analysis"] = results
                    st.session_state["latest_filename"] = uploaded_file.name
                status.success("Document analysis complete.")
                progress_bar.progress(100)

            if st.session_state.get("latest_analysis") and st.session_state.get("latest_filename") == uploaded_file.name:
                render_compliance_dashboard(st.session_state["latest_analysis"])

        else:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            st.subheader("Start your analysis")
            st.markdown(
                "Choose a PDF, CSV, or TXT document to scan for sensitive data and generate a compliance-ready report."
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        render_legacy_sales_tab()

    st.markdown("---")
    footer_cols = st.columns([3, 1, 1, 1])
    footer_cols[0].markdown(
        "**Created by Aarsh Pavashiya**  <br>"
        "AI Research Internship Project  <br>"
        "Python | Streamlit | AI"
        , unsafe_allow_html=True
    )
    footer_cols[1].markdown("[GitHub](https://github.com)")
    footer_cols[2].markdown("[LinkedIn](https://linkedin.com)")
    footer_cols[3].markdown("Built for portfolio and product showcase.")


if __name__ == "__main__":
    main()
