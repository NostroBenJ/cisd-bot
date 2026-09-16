@echo off
REM Monthly momentum rebalance -- Alpaca PAPER.
REM
REM Run by Windows Task Scheduler EVERY WEEKDAY. The script itself decides
REM whether a rebalance is due (--if-due), so holidays, weekends, a sleeping
REM laptop and a missed month all resolve without the scheduler knowing
REM anything about market calendars.
REM
REM --refresh pulls fresh prices from Alpaca first and REFUSES to trade if the
REM feed disagrees with the history on disk.
REM
REM To stop it:   schtasks /Delete /TN "CISD Momentum Rebalance" /F
REM To run now:   schtasks /Run /TN "CISD Momentum Rebalance"

cd /d "%~dp0"
if not exist logs mkdir logs

for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set STAMP=%%c-%%a-%%b

echo. >> "logs\rebalance.log"
echo ================================================== >> "logs\rebalance.log"
echo RUN %date% %time% >> "logs\rebalance.log"

python run_momentum.py --submit --if-due --refresh --equity 1000 >> "logs\rebalance.log" 2>&1

echo EXIT %ERRORLEVEL% >> "logs\rebalance.log"
exit /b %ERRORLEVEL%
