"""WRITEUP.md -> WRITEUP.pdf through headless Chrome (A4, two to four pages)."""
import os, subprocess, sys, markdown
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS = """@page { size: A4; margin: 16mm 17mm; }
body { font: 10pt/1.38 Georgia, serif; color: #111; }
h1 { font-size: 15.5pt; margin: 0 0 4pt; } h2 { font-size: 11.5pt; margin: 11pt 0 3pt; }
p, li { margin: 3pt 0; text-align: justify; } ul { padding-left: 15pt; }
table { border-collapse: collapse; font: 8.3pt/1.3 Helvetica, Arial, sans-serif; margin: 5pt 0; width: 100%; }
th, td { border-bottom: 0.5pt solid #bbb; padding: 2pt 4pt; text-align: left; } th { border-bottom: 1pt solid #333; }
pre { font-size: 8.8pt; background: #f4f4f4; padding: 4pt 8pt; } code { font-size: 8.8pt; }"""
html = markdown.markdown(open(f"{HERE}/WRITEUP.md").read(), extensions=["tables"])
open(f"{HERE}/WRITEUP.html", "w").write(f"<!doctype html><meta charset=utf-8><style>{CSS}</style>{html}")
chrome = sys.argv[1] if len(sys.argv) > 1 else "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={HERE}/WRITEUP.pdf",
                f"file://{HERE}/WRITEUP.html"], check=True, capture_output=True)
os.remove(f"{HERE}/WRITEUP.html")
