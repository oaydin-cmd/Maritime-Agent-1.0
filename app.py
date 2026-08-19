def create_advanced_sms_docx(work_title, content_text):
    """SIRE 2.0 / SMS Standartlarında Tablolu Word Dokümanı Hazırlar."""
    doc = DocxDocument()
    
    # Sayfa Kenar Boşlukları (Daraltılmış - Daha fazla sığsın)
    for section in doc.sections:
        section.top_margin = Inches(0.6)
        section.bottom_margin = Inches(0.6)
        section.left_margin = Inches(0.6)
        section.right_margin = Inches(0.6)

    # Başlık ve Üst Bilgi
    title_p = doc.add_paragraph()
    title_run = title_p.add_run("SAFETY MANAGEMENT SYSTEM (SMS)\nRISK ASSESSMENT & PERMIT TO WORK")
    title_run.bold = True
    title_run.font.size = Pt(16)
    title_run.font.color.rgb = RGBColor(0, 51, 102) # Koyu Deniz Mavi
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Gemi / Operasyon Üst Bilgi Tablosu
    header_table = doc.add_table(rows=2, cols=4)
    header_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    header_table.style = 'Table Grid'
    
    fields = [
        ("Vessel Name:", "M/T "),
        ("IMO No:", ""), 
        ("Date & Time:", ""),
        ("Location / Tank:", ""),
        ("Work Description:", work_title[:40]),
        ("Permit No:", "RA-2026-"),
        ("Risk Level:", "HIGH / MEDIUM / LOW"),
        ("Status:", "APPROVED")
    ]
    
    for i, (label, val) in enumerate(fields):
        row_idx = i // 4
        col_idx = (i % 4)
        if row_idx < 2:
            cell = header_table.cell(row_idx, col_idx)
            cell.paragraphs[0].text = f"{label} {val}"
            cell.paragraphs[0].runs[0].font.bold = True
            cell.paragraphs[0].runs[0].font.size = Pt(9)

    doc.add_paragraph("\n")

    # Yapay Zekanın Ürettiği İçeriği İşle ve İncele
    lines = content_text.split('\n')
    table_data = []

    for line in lines:
        l = line.strip()
        if not l:
            continue

        # Başlık Kontrolleri
        if l.startswith('# '):
            doc.add_heading(l.replace('# ', ''), level=1)
        elif l.startswith('## '):
            doc.add_heading(l.replace('## ', ''), level=2)
        elif l.startswith('### '):
            doc.add_heading(l.replace('### ', ''), level=3)
        # Tablo Satırı tespiti (| ile başlayan satırlar)
        elif l.startswith('|'):
            cells = [c.strip() for c in l.split('|')[1:-1]]
            # Ayraç satırlarını (---|---) atla
            if cells and not all(set(c).issubset({'-', ':', ' '}) for c in cells):
                table_data.append(cells)
        else:
            # Eğer tablodan çıkıldıysa mevcut birikmiş tabloyu oluştur
            if table_data:
                _build_docx_table(doc, table_data)
                table_data = []
            
            if l.startswith('* ') or l.startswith('- '):
                doc.add_paragraph(l[2:], style='List Bullet')
            else:
                doc.add_paragraph(l)

    # Kalan tablo varsa bas
    if table_data:
        _build_docx_table(doc, table_data)

    # Onay Kutuları / Checklist Alanı
    doc.add_heading("Safety Checklist & Controls", level=2)
    p_check = doc.add_paragraph()
    p_check.add_run("[  ] Risk Assessment briefed to all team members (Toolbox Talk Completed)\n").font.size = Pt(9.5)
    p_check.add_run("[  ] Required PPE available and inspected\n").font.size = Pt(9.5)
    p_check.add_run("[  ] Isolation / LOTO applied (If applicable)\n").font.size = Pt(9.5)
    p_check.add_run("[  ] Gas test performed and readings logged (If applicable)\n").font.size = Pt(9.5)
    p_check.add_run("[  ] Communication established with Duty Officer / Bridge").font.size = Pt(9.5)

    # İmza Blokları
    doc.add_heading("Authorisation & Signatures", level=2)
    sig_table = doc.add_table(rows=2, cols=3)
    sig_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    sig_table.style = 'Table Grid'
    
    headers = ["Prepared By (Person in Charge)", "Checked By (Safety Officer)", "Approved By (Master / Ch.Eng)"]
    for idx, text in enumerate(headers):
        cell = sig_table.cell(0, idx)
        cell.paragraphs[0].text = text
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(9)
        
    for idx in range(3):
        cell = sig_table.cell(1, idx)
        cell.paragraphs[0].text = "\nName:\nRank:\nSignature:\nDate:"
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio
