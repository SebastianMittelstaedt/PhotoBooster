pyinstaller.exe PhotoBooster.py
xcopy /e /v .\src\opencl .\dist\PhotoBooster\src\opencl\
"C:\Program Files\7-Zip\7z.exe" a -r .\builds\PhotoBooster.7z .\dist\PhotoBooster\*