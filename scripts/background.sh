nohup ./scripts/autorun.sh > autorun.log 2>&1 &

# status: ps aux | grep autorun.sh
# tail: tail -f autorun.log
# kill: kill $(pgrep -f autorun.sh)