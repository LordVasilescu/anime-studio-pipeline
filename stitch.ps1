# © 2026 Mathew. All rights reserved.
# AnimeStudio stitcher — concatenates generated clips into one movie file.
# Usage:
#   .\stitch.ps1                          -> stitches every mp4/webm in .\clips (alphabetical) to movie.mp4
#   .\stitch.ps1 -ClipDir <dir> -Out <file> [-Music <audio file>]
# Name clips with sortable prefixes (010_intro.mp4, 020_fight.mp4) to control scene order.

param(
    [string]$ClipDir = "$PSScriptRoot\clips",
    [string]$Out = "$PSScriptRoot\movie.mp4",
    [string]$Music = ""
)

$clips = Get-ChildItem $ClipDir -File | Where-Object { $_.Extension -match '\.(mp4|webm|mkv|mov)$' } | Sort-Object Name
if (-not $clips) { Write-Error "No clips found in $ClipDir"; exit 1 }

Write-Host "Stitching $($clips.Count) clips:"
$clips | ForEach-Object { Write-Host "  $($_.Name)" }

# Re-encode pass keeps mixed resolutions/codecs from breaking concat.
$listFile = Join-Path $env:TEMP "anime_stitch_list.txt"
$tempDir = Join-Path $env:TEMP "anime_stitch_norm"
New-Item -ItemType Directory -Force $tempDir | Out-Null
Remove-Item "$tempDir\*" -Force -ErrorAction SilentlyContinue

$i = 0
$lines = foreach ($c in $clips) {
    $norm = Join-Path $tempDir ("{0:D4}.mp4" -f $i); $i++
    ffmpeg -y -loglevel error -i $c.FullName -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=24" -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -an $norm
    if ($LASTEXITCODE -ne 0) { Write-Error "ffmpeg failed on $($c.Name)"; exit 1 }
    "file '$($norm.Replace('\','/'))'"
}
Set-Content $listFile $lines -Encoding ascii

if ($Music -and (Test-Path $Music)) {
    ffmpeg -y -loglevel error -f concat -safe 0 -i $listFile -i $Music -c:v copy -c:a aac -shortest $Out
} else {
    ffmpeg -y -loglevel error -f concat -safe 0 -i $listFile -c:v copy $Out
}
if ($LASTEXITCODE -eq 0) { Write-Host "Done -> $Out" } else { Write-Error "Final concat failed" }
