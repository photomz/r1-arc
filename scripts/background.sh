nohup ./scripts/autorun.sh > autorun.log 2>&1 &
ps aux | grep autorun.sh
tail -f autorun.log

# status: ps aux | grep autorun.sh
# tail: tail -f autorun.log
# SIGINT: kill -2 $(pgrep -f autorun.sh)