"""
Generates a one-page PDF summary of a single prediction: submitted
values, result, and (if available) the explanation — using the same
disclaimer language shown throughout the site, so the PDF is consistent
with what the web UI already communicates.
"""
import io
from datetime import datetime, timezone
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

from utils import get_feature_specs

_CONFIDENCE_NOTES = {
    "limited": "Based on a limited training dataset \u2014 treat this estimate with extra caution.",
    "moderate": "Based on a moderate-sized training dataset.",
    "adequate": "Based on a reasonably sized training dataset.",
}


def _display_value(spec, raw_value):
    if spec["type"] == "categorical":
        try:
            code = int(float(raw_value))
            if code in spec["option_labels"]:
                return spec["option_labels"][code]
        except (TypeError, ValueError):
            pass
    return str(raw_value)


def generate_prediction_pdf(disease_key, disease_cfg, input_data, result):
    specs = get_feature_specs(disease_key)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=18, spaceAfter=4)
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9, textColor=colors.grey)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=13,
                                    spaceBefore=14, spaceAfter=6)
    body_style = styles["Normal"]
    border_color = colors.HexColor("#DCE7E4")
    header_bg = colors.HexColor("#EAF1EF")
    ink = colors.HexColor("#142B29")

    story = [
        Paragraph(f"{disease_cfg['display_name']} Risk Assessment", title_style),
        Paragraph(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", meta_style),
        Spacer(1, 10),
        HRFlowable(width="100%", color=border_color),
        Spacer(1, 10),
        Paragraph("Result", section_style),
    ]

    risk_color = colors.HexColor("#B4392B") if result["prediction"] == 1 else colors.HexColor("#1E7B3C")
    result_style = ParagraphStyle("ResultLabel", parent=styles["Normal"], fontSize=14,
                                   textColor=risk_color, spaceAfter=4)
    story.append(Paragraph(f"<b>{result['risk_label']}</b>", result_style))
    story.append(Paragraph(f"Estimated likelihood: {result['probability']*100:.1f}%", body_style))

    conf = result.get("data_confidence")
    if conf in _CONFIDENCE_NOTES:
        story.append(Paragraph(_CONFIDENCE_NOTES[conf], meta_style))

    story.append(Paragraph("Submitted Values", section_style))
    table_data = [["Field", "Value"]]
    for feature, spec in specs.items():
        table_data.append([spec["label"], _display_value(spec, input_data.get(feature, ""))])

    table = Table(table_data, colWidths=[3 * inch, 3 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), ink),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F6")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)

    if isinstance(result.get("explanation"), list) and result["explanation"]:
        story.append(Paragraph("Factors That Influenced This Result", section_style))
        exp_data = [["Factor", "Impact"]]
        for item in result["explanation"][:5]:
            exp_data.append([item["feature"], f"{item['impact']:+.4f}"])
        exp_table = Table(exp_data, colWidths=[3 * inch, 3 * inch])
        exp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), header_bg),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, border_color),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(exp_table)

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", color=border_color))
    story.append(Spacer(1, 8))
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    story.append(Paragraph(
        "This report provides a statistical estimate based on historical data, not a medical "
        "diagnosis. Always consult a qualified healthcare professional about these results.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer
