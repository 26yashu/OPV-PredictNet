param (
    [string]$docxPath = "c:\Users\DELL\OPV-PredictNet\OPV_PredictNet_Final_Report.docx",
    [string]$pdfPath = "c:\Users\DELL\OPV-PredictNet\OPV_PredictNet_Final_Report.pdf"
)

try {
    Write-Output "Starting Word COM automation..."
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    
    Write-Output "Opening $docxPath..."
    $doc = $word.Documents.Open($docxPath)
    
    # wdFormatPDF is 17
    Write-Output "Saving as PDF to $pdfPath..."
    $doc.SaveAs([ref]$pdfPath, [ref]17)
    $doc.Close([ref]0)
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    
    if (Test-Path $pdfPath) {
        $size = (Get-Item $pdfPath).Length
        Write-Output "[SUCCESS] Exported PDF: $pdfPath ($size bytes)"
    } else {
        Write-Output "[ERROR] PDF file was not created."
    }
} catch {
    Write-Output "[ERROR] Conversion failed: $($_.Exception.Message)"
}
