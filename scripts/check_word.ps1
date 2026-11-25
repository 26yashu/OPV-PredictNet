try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    Write-Output "Word COM is available: Version $($word.Version)"
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
} catch {
    Write-Output "Word COM not available: $($_.Exception.Message)"
}
