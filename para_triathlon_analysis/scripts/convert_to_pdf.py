"""
HTML to PDF Converter for Para Standards Reports
Uses Chrome/Edge in headless mode to convert HTML reports to PDF

Note: Headless browser PDF generation is unreliable. This script will verify PDFs are created.
If it fails, use manual conversion: Open HTML in browser → Ctrl+P → Save as PDF
"""
import subprocess
import sys
import time
from pathlib import Path


def find_chrome_or_edge():
    """Find Chrome or Edge executable on Windows."""
    possible_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    return None


def convert_html_to_pdf(html_path: Path, pdf_path: Path, browser_exe: str) -> tuple[bool, str]:
    """Convert HTML file to PDF using Chrome/Edge headless. Returns (success, error_message)."""
    try:
        # Resolve to absolute paths — Chrome resolves --print-to-pdf relative to its
        # own working directory, not the script's, so relative paths produce silent misses.
        html_abs = html_path.resolve()
        pdf_abs = pdf_path.resolve()

        # Remove a stale/old PDF so we can detect fresh creation below.
        if pdf_abs.exists():
            pdf_abs.unlink()

        cmd = [
            browser_exe,
            "--headless=new",
            "--allow-file-access-from-files",
            "--disable-web-security",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=20000",
            f"--print-to-pdf={pdf_abs}",
            f"file:///{html_abs.as_posix()}"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        # Chrome emits benign stderr (e.g. GCM registration errors) even on success.
        # The only reliable success signal is whether the PDF file was written with content.
        # Poll for up to 6 seconds to handle delayed flush on Windows.
        for _ in range(12):
            if pdf_abs.exists() and pdf_abs.stat().st_size > 0:
                # Copy to the originally requested path if they differ.
                if pdf_abs != pdf_path.resolve():
                    pdf_abs.replace(pdf_path)
                return True, ""
            time.sleep(0.5)

        # Nothing was written — capture a meaningful error if available.
        # Filter out the common benign GCM noise so real errors are visible.
        stderr_lines = [
            ln for ln in (result.stderr or "").splitlines()
            if "registration_request" not in ln and ln.strip()
        ]
        error_msg = "\n".join(stderr_lines[:5]) if stderr_lines else "PDF file not created or empty"
        return False, error_msg
    except Exception as e:
        return False, str(e)


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_reports_to_pdf.py <batch_output_directory>")
        print("Example: python convert_reports_to_pdf.py tri_analysis/outputs/para_standards_batch_20260128_142005")
        sys.exit(1)
    
    batch_dir = Path(sys.argv[1])
    
    if not batch_dir.exists():
        print(f"Error: Directory not found: {batch_dir}")
        sys.exit(1)
    
    browser = find_chrome_or_edge()
    if not browser:
        print("Error: Could not find Chrome or Edge browser.")
        print("Please install Chrome or Edge, or use manual conversion:")
        print("  1. Open report.html in browser")
        print("  2. Ctrl+P (Print)")
        print("  3. Save as PDF")
        sys.exit(1)
    
    print(f"Found browser: {browser}")
    print(f"Converting reports in: {batch_dir}\n")
    
    success_count = 0
    failed = []
    
    # Find all report.html files
    for category_dir in sorted(batch_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        
        html_file = category_dir / "report.html"
        if not html_file.exists():
            continue
        
        pdf_file = category_dir / "report.pdf"
        category_name = category_dir.name.replace('_', ' ')
        
        print(f"Converting: {category_name}...", end=" ")
        
        success, error = convert_html_to_pdf(html_file, pdf_file, browser)
        if success:
            success_count += 1
            print("✓ Done")
        else:
            failed.append((category_name, error))
            print(f"✗ Failed: {error[:80] if error else 'Unknown error'}")
    
    print(f"\n{'='*50}")
    
    if success_count > 0:
        print(f"✓ Successfully created {success_count} PDF files")
        print(f"\nPDF reports saved alongside HTML files in: {batch_dir}")
    else:
        print("✗ No PDFs were created")
        print("\nHeadless browser PDF conversion is unreliable.")
        print("Please use manual conversion instead:")
        print("  1. Open each report.html in Chrome/Edge")
        print("  2. Press Ctrl+P (Print)")
        print("  3. Select 'Save as PDF'")
        print("  4. Save in the same folder as the HTML file")
    
    if failed:
        print(f"\nFailed conversions ({len(failed)}):")
        for name, error in failed:
            print(f"  - {name}: {error[:100] if error else 'Unknown error'}")
    
    # Exit with error code if no PDFs were created
    if success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
