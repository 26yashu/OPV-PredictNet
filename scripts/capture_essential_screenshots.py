"""
OPV-PredictNet — Playwright Automated Screenshot Capture.
Runs live FastAPI backend and Vite frontend, connects via headless Chrome,
interacts with the actual application, and captures 8 genuine screenshots.
"""

from pathlib import Path
import subprocess
import sys
import time
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.security import create_access_token

SCREENSHOTS_DIR = PROJECT_ROOT / "docs" / "report_assets" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not Path(CHROME_EXE).exists():
    CHROME_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# Generate valid JWT token for seamless authenticated session
auth_token = create_access_token(subject="1")

print("Starting background backend FastAPI server on port 8000...")
backend_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"],
    cwd=str(PROJECT_ROOT)
)

print("Starting background frontend Vite dev server on port 5173...")
frontend_proc = subprocess.Popen(
    ["cmd.exe", "/c", "npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"],
    cwd=str(PROJECT_ROOT / "frontend")
)

try:
    print("Waiting 5 seconds for local servers to initialize...")
    time.sleep(5.0)

    with sync_playwright() as p:
        print(f"Launching Chrome from: {CHROME_EXE}")
        browser = p.chromium.launch(executable_path=CHROME_EXE, headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 920}, device_scale_factor=2)
        
        # Inject auth token into browser storage
        context.add_init_script(f"""
            localStorage.setItem('opv_predictnet_auth_token', '{auth_token}');
        """)
        
        page = context.new_page()

        # 1. Landing Page
        print("1. Navigating to Landing Page (http://127.0.0.1:5173)...")
        page.goto("http://127.0.0.1:5173", wait_until="networkidle")
        time.sleep(2.0)
        page.screenshot(path=str(SCREENSHOTS_DIR / "01_landing_page.png"))
        print("[OK] Captured 01_landing_page.png")

        # 2. Login Page / Modal
        print("2. Opening Auth Modal...")
        auth_btn = page.locator("button.nav-auth-btn")
        if auth_btn.count() > 0:
            auth_btn.first.click()
            time.sleep(1.0)
            page.screenshot(path=str(SCREENSHOTS_DIR / "02_login_page.png"))
            print("[OK] Captured 02_login_page.png")
            
            # Close modal via modal-close-btn
            close_btn = page.locator("button.modal-close-btn")
            if close_btn.count() > 0:
                close_btn.first.click()
                time.sleep(0.5)

        # 3. Main Prediction Dashboard
        print("3. Navigating to Prediction Workspace...")
        workspace_tab = page.locator("button.nav-link-btn:has-text('Prediction Workspace')")
        if workspace_tab.count() > 0:
            workspace_tab.first.click()
        else:
            launch_hero = page.locator("button:has-text('Launch Prediction Workspace')")
            if launch_hero.count() > 0:
                launch_hero.first.click()
        time.sleep(2.0)
        page.screenshot(path=str(SCREENSHOTS_DIR / "03_prediction_dashboard.png"))
        print("[OK] Captured 03_prediction_dashboard.png")

        # 4. Molecular Input with donor and acceptor SMILES
        print("4. Populating preset molecules (PM6 : Y6)...")
        preset_btn = page.locator("button:has-text('PM6 : Y6')")
        if preset_btn.count() > 0:
            preset_btn.first.click()
            time.sleep(1.0)
        page.screenshot(path=str(SCREENSHOTS_DIR / "04_molecular_input.png"))
        print("[OK] Captured 04_molecular_input.png")

        # 5. RDKit Molecular Analysis
        print("5. Running RDKit analysis...")
        analyze_btn = page.locator("button:has-text('Analyze Molecules')")
        if analyze_btn.count() > 0:
            analyze_btn.first.click()
            time.sleep(2.0)
        page.screenshot(path=str(SCREENSHOTS_DIR / "05_rdkit_analysis.png"))
        print("[OK] Captured 05_rdkit_analysis.png")

        # 6. Actual PCE Prediction Result
        print("6. Running PCE prediction for M3...")
        m3_tab = page.locator("button:has-text('M3')")
        if m3_tab.count() > 0:
            m3_tab.first.click()
            time.sleep(0.5)
        
        predict_btn = page.locator("button:has-text('Predict PCE')")
        if predict_btn.count() > 0:
            predict_btn.first.click()
            time.sleep(2.5)
        page.screenshot(path=str(SCREENSHOTS_DIR / "06_prediction_result.png"))
        print("[OK] Captured 06_prediction_result.png")

        # 7. OOD Warning or Rejection Example
        print("7. Testing OOD input (unseen norbornane scaffold)...")
        donor_inputs = page.locator("textarea")
        if donor_inputs.count() > 0:
            donor_inputs.first.fill("C1CC2CCC1C2")
            time.sleep(0.5)
            if predict_btn.count() > 0:
                predict_btn.first.click()
                time.sleep(2.0)
        page.screenshot(path=str(SCREENSHOTS_DIR / "07_ood_warning_rejection.png"))
        print("[OK] Captured 07_ood_warning_rejection.png")

        # 8. Prediction History
        print("8. Capturing Prediction History...")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1.0)
        page.screenshot(path=str(SCREENSHOTS_DIR / "08_prediction_history.png"))
        print("[OK] Captured 08_prediction_history.png")

        browser.close()

finally:
    print("Shutting down background demonstration servers...")
    backend_proc.terminate()
    frontend_proc.terminate()
    try:
        backend_proc.wait(timeout=5)
        frontend_proc.wait(timeout=5)
    except Exception:
        pass
    print("[OK] All servers stopped cleanly.")
