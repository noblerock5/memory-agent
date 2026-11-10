"""批量灌库脚本: 把 docs/文档资料/ 下的 PDF/DOCX/XLSX 解析为文本并入 Milvus 知识库

用法: .venv/bin/python scripts/ingest_docs.py [--user-id kld_user] [--dir docs/文档资料]
图片文件跳过 (无文本层)。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import knowledge  # noqa: E402


def extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[第{i + 1}页]\n{text}")
    return "\n\n".join(pages)


def extract_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def extract_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        rows = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(v).strip() for v in row if v is not None and str(v).strip()]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            parts.append(f"[工作表: {ws.title}]\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(parts)


EXTRACTORS = {".pdf": extract_pdf, ".docx": extract_docx, ".xlsx": extract_xlsx}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", default="kld_user")
    parser.add_argument("--dir", default="docs/文档资料")
    args = parser.parse_args()

    root = Path(args.dir)
    files = sorted(f for f in root.iterdir() if f.suffix.lower() in EXTRACTORS)
    print(f"共 {len(files)} 个待处理文件, user_id={args.user_id}")

    ok, failed, total_chunks = 0, 0, 0
    for f in files:
        try:
            text = EXTRACTORS[f.suffix.lower()](f)
            if len(text.strip()) < 20:
                print(f"  跳过(无有效文本): {f.name}")
                failed += 1
                continue
            result = knowledge.ingest_text(args.user_id, f.name, text)
            stored = result.get("stored", 0)
            if stored > 0:
                ok += 1
                total_chunks += stored
                print(f"  入库 {stored} 块: {f.name}")
            else:
                failed += 1
                print(f"  失败: {f.name} -> {result.get('error')}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  异常: {f.name} -> {type(e).__name__}: {e}")

    print(f"\n完成: 成功 {ok} 个文件 / {total_chunks} 块; 失败 {failed}")
    print("状态:", knowledge.status())


if __name__ == "__main__":
    main()
