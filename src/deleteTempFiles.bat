@echo off
cd /d "%~dp0"
cd ..
set TempFiles=%cd%\PyomoTempFiles
@echo on 
cd %TempFiles%
del *.lp
del *.log
del *.script
del *.sol
del *.dat
del *.txt

@echo off
cd ..
set OutputFiles=%cd%\OutputFiles\T-DOPFFiles
@echo on
cd %OutputFiles%
del *.dat

