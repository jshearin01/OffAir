; OffAir installer — Inno Setup 6 script.
;
; Build (on the Windows build machine, after `pyinstaller OffAir.spec`):
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" OffAir.iss
;
; Output: installer\OffAir-Setup.exe — a per-user installer (no admin
; rights needed) that copies the onedir bundle to %LOCALAPPDATA%\
; Programs\OffAir, adds a Start-menu entry and an optional desktop
; shortcut, and registers an uninstaller. The installed app is the
; same frozen build as dist\OffAir\OffAir.exe: no Python, pip, or
; MSVC tools needed on the target machine. Whisper/sherpa models
; still download once on first use into %USERPROFILE%\.offair\models.
;
; Unsigned build: Windows SmartScreen will warn on first run
; ("Windows protected your PC" → More info → Run anyway) until the
; project gets a code-signing certificate. See docs/PACKAGING.md.
;
; Paths below are relative to this file (repo root), so the script
; works from a clean checkout after the PyInstaller build.

#define AppVersion "1.0.6"

[Setup]
AppName=OffAir
AppVersion={#AppVersion}
AppPublisher=OffAir
DefaultDirName={autopf}\OffAir
DefaultGroupName=OffAir
OutputDir=installer
OutputBaseFilename=OffAir-Setup-{#AppVersion}
Compression=lzma2/ultra
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
WizardStyle=modern
UninstallDisplayIcon={app}\OffAir.exe
SetupIconFile=assets\offair.ico

[Files]
Source: "dist\OffAir\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\OffAir"; Filename: "{app}\OffAir.exe"
Name: "{autodesktop}\OffAir"; Filename: "{app}\OffAir.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\OffAir.exe"; Description: "Launch OffAir"; Flags: nowait postinstall skipifsilent
