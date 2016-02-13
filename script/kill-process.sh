ps aux | grep "marble" | awk '{print "kill "$2}' | bash
