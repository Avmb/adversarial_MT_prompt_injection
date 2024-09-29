find pia/ -type f -exec bash -c 'if [ `cat "{}" |wc -w` -eq 0 ]; then echo "{}";fi' \; 
