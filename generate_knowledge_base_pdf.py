from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from knowledge_base import (
    SYSTEM_INFO, FEATURES, FAQ, USER_GUIDE, ERROR_MESSAGES,
    SYSTEM_COMMANDS, CHATBOT_RESPONSES, SUBJECT_KEYWORDS,
    MODULE_HELP, LEARNING_RESOURCES
)

def create_knowledge_base_pdf():
    # Create PDF document
    doc = SimpleDocTemplate("knowledge_base.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12
    )

    # Title
    story.append(Paragraph("EduTrace Knowledge Base", title_style))
    story.append(Spacer(1, 20))

    # System Information
    story.append(Paragraph("System Information", heading_style))
    info_data = [
        ["Name", SYSTEM_INFO["name"]],
        ["Version", SYSTEM_INFO["version"]],
        ["Description", SYSTEM_INFO["description"]],
        ["Purpose", SYSTEM_INFO["purpose"]]
    ]
    info_table = Table(info_data, colWidths=[1.5*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('LEADING', (0, 0), (-1, -1), 14),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 20))

    # Features
    story.append(Paragraph("System Features", heading_style))
    for feature in FEATURES:
        story.append(Paragraph(f"<b>{feature['name']}</b>", body_style))
        story.append(Paragraph(f"Description: {feature['description']}", body_style))
        story.append(Paragraph(f"How to use: {feature['how_to_use']}", body_style))
        story.append(Spacer(1, 10))
    story.append(Spacer(1, 10))

    # FAQ
    story.append(Paragraph("Frequently Asked Questions", heading_style))
    for faq in FAQ:
        story.append(Paragraph(f"<b>Q: {faq['question']}</b>", body_style))
        story.append(Paragraph(f"A: {faq['answer']}", body_style))
        story.append(Spacer(1, 10))
    story.append(Spacer(1, 10))

    # User Guide
    story.append(Paragraph("User Guide", heading_style))
    for section, items in USER_GUIDE.items():
        story.append(Paragraph(f"<b>{section.replace('_', ' ').title()}</b>", body_style))
        for item in items:
            story.append(Paragraph(f"• {item}", body_style))
        story.append(Spacer(1, 10))
    story.append(Spacer(1, 10))

    # Subject Keywords
    story.append(Paragraph("Subject-Specific Keywords", heading_style))
    for subject, keywords in SUBJECT_KEYWORDS.items():
        story.append(Paragraph(f"<b>{subject}</b>", body_style))
        story.append(Paragraph(", ".join(keywords), body_style))
        story.append(Spacer(1, 10))
    story.append(Spacer(1, 10))

    # Module Help
    story.append(Paragraph("Module-Specific Help", heading_style))
    for module, help_text in MODULE_HELP.items():
        story.append(Paragraph(f"<b>{module}</b>", body_style))
        story.append(Paragraph(help_text, body_style))
        story.append(Spacer(1, 10))
    story.append(Spacer(1, 10))

    # Learning Resources
    story.append(Paragraph("Learning Resources", heading_style))
    for resource, description in LEARNING_RESOURCES.items():
        story.append(Paragraph(f"<b>{resource.title()}</b>", body_style))
        story.append(Paragraph(description, body_style))
        story.append(Spacer(1, 10))
    story.append(Spacer(1, 10))

    # System Commands
    story.append(Paragraph("System Commands", heading_style))
    cmd_data = [[cmd, desc] for cmd, desc in SYSTEM_COMMANDS.items()]
    cmd_table = Table(cmd_data, colWidths=[1.5*inch, 4*inch])
    cmd_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('LEADING', (0, 0), (-1, -1), 14),
    ]))
    story.append(cmd_table)
    story.append(Spacer(1, 20))

    # Error Messages
    story.append(Paragraph("Error Messages", heading_style))
    error_data = [[error_type, message] for error_type, message in ERROR_MESSAGES.items()]
    error_table = Table(error_data, colWidths=[1.5*inch, 4*inch])
    error_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('LEADING', (0, 0), (-1, -1), 14),
    ]))
    story.append(error_table)

    # Build PDF
    doc.build(story)

if __name__ == "__main__":
    create_knowledge_base_pdf() 