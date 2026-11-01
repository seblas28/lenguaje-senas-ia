Option Explicit
Dim fs, sh, root, python, entry
Set fs = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
root = fs.GetParentFolderName(WScript.ScriptFullName)
python = fs.BuildPath(root, ".venv\Scripts\pythonw.exe")
entry = fs.BuildPath(root, "entrenador.pyw")
If Not fs.FileExists(python) Then
  MsgBox "No se encontro .venv\Scripts\pythonw.exe. Prepara el entorno Python 3.12 como indica README.md.", 48, "Kai Senas - Entorno pendiente"
  WScript.Quit 1
End If
sh.CurrentDirectory = root
sh.Run Chr(34) & python & Chr(34) & " " & Chr(34) & entry & Chr(34), 1, False

