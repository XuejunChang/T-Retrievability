#!/bin/bash

for dr in bm25 splade tctcolbert bm25_tctcolbert bm25_monot5 ; do
    cd ../retrieved_trec_files/
    rm -rf ./tmp

    echo decompressing "${dr}" split files
    cp -r "${dr}" ./tmp/ && cd ./tmp/ &&
    merged_res="${dr}_msmarco-passage_dev_100.res"
    xz -d *.xz && cat * > "${merged_res}"
    mv "${merged_res}" ../
    cd .. && rm -rf ./tmp

    echo finished

done