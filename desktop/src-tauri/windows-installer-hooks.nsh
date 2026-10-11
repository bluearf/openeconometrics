; NSIS copies the new resource inventory, but old versioned metadata can survive
; an in-place upgrade. Replace only the application's bundled Python runtime.
; User projects and the KnownFolder profile are outside this installation tree.
!macro NSIS_HOOK_PREINSTALL
  IfFileExists "$INSTDIR\runtime\openecon-runtime\*.*" 0 oe_runtime_clean_done
  IfFileExists "$INSTDIR\openecon-desktop.exe" 0 oe_runtime_clean_refuse
  IfFileExists "$INSTDIR\runtime\openecon-runtime\openecon-runtime.exe" 0 oe_runtime_clean_refuse
  ; Refuse redirected directories rather than traversing an unexpected target.
  System::Call 'kernel32::GetFileAttributesW(w "$INSTDIR") i .r0'
  IntOp $0 $0 & 0x400
  IntCmp $0 0 +2
    Goto oe_runtime_clean_refuse
  System::Call 'kernel32::GetFileAttributesW(w "$INSTDIR\runtime") i .r0'
  IntOp $0 $0 & 0x400
  IntCmp $0 0 +2
    Goto oe_runtime_clean_refuse
  System::Call 'kernel32::GetFileAttributesW(w "$INSTDIR\runtime\openecon-runtime") i .r0'
  IntOp $0 $0 & 0x400
  IntCmp $0 0 +2
    Goto oe_runtime_clean_refuse
  ClearErrors
  RMDir /r "$INSTDIR\runtime\openecon-runtime"
  IfErrors oe_runtime_clean_refuse
  Goto oe_runtime_clean_done
oe_runtime_clean_refuse:
  Abort "The previous bundled runtime could not be replaced safely. Close OpenEconometrics and retry."
oe_runtime_clean_done:
!macroend
