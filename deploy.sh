#!/bin/bash

project="exposure-fairness"

rm -rf /root/$project/*
mkdir -p /root/$project/

echo "copying project to /root"
cp -r /mnt/primary/$project/* /root/$project/
echo "copied"

