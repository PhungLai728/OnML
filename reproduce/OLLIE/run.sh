#!/bin/sh
for i in {1..10}; do 
	java -Xmx512m -jar ollie-app-latest.jar --split OLLIE_cc_in/"in$i" --output OLLIE_cc_out/"out$i"; 
done