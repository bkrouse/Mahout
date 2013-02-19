#!/bin/bash
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#INTRO explanations

function fileExists() {
  if ([ "$MAHOUT_LOCAL" != "" ] && [ ! -e "$1" ]) || ([ "$MAHOUT_LOCAL" == "" ] && ! $HADOOP_HOME/bin/hadoop fs -test -e /user/$USER/$1); then
    return 1 # file doesn't exist
  else
    return 0 # file exists
  fi
}

function removeFolder() {
  if [ "$MAHOUT_LOCAL" == "" ]; then
    rm -rf $1
  else
    if fileExists "$1"; then
      hadoop fs -rmr /user/$USER/$1
    fi
  fi	
}

if [ "$1" = "--help" ] || [ "$1" = "--?" ]; then
  echo "This script runs SVT on a handful of different datasets."
  exit
fi

if [ -z "$2" ]; then
  echo "Usage: svt-examples.sh input_path output_path  (where these paths are on hdfs)"
  exit
fi



SCRIPT_PATH=${0%/*}
if [ "$0" != "$SCRIPT_PATH" ] && [ "$SCRIPT_PATH" != "" ]; then
  cd $SCRIPT_PATH
fi
START_PATH=`pwd`
echo "START_PATH: $START_PATH"
MAHOUT="../../bin/mahout"
IN=$1
OUT=$2


algorithm=( csv netflix yahoo clean )
if [ -n "$3" ]; then
  choice=$3
else
  echo "Please select a number to choose the dataset for SVT matrix completion"
  echo "1. ${algorithm[0]}"
  echo "2. ${algorithm[1]}"
  echo "3. ${algorithm[2]}"
  echo "4. ${algorithm[3]} -- cleans up the work area -- all files under the work area will be deleted"
  read -p "Enter your choice : " choice
fi
echo "ok. You chose $choice and we'll use ${algorithm[$choice-1]}"
alg=${algorithm[$choice-1]}


if [ "x$alg" == "xcsv" ]; then
  # convert the csvs to Sequence File and run SVTEvaluator
  M_CSV="$IN/M.csv"
  OMEGA_CSV="$IN/OmegaRowBasedSorted.csv"
  M_SEQ="$OUT/csv/tmp/M.seq"
  OMEGA_SEQ="$OUT/csv/tmp/OmegaRowBasedSorted.seq"
  SVTOUTPUT_SEQ="$OUT/csv/SVTOutput.seq"
  SVTOUTPUT_CSV="$IN/SVTOutput.csv"
  CSV_TMP="$OUT/csv/tmp"

#original line had the /chunk-0...but trying without. I may need to revisit this with a large csv...not sure. 
#  if ! fileExists "$MATRIX_IN/chunk-0"; then
  if ! fileExists "$M_SEQ"; then
    echo "Converting $M_CSV to DistrubutedMatrix format"

    $MAHOUT org.apache.mahout.completion.svt.conversion.DistributedMatrixFromCsv --input $M_CSV --output $M_SEQ
  fi

  if ! fileExists "$OMEGA_SEQ"; then
    echo "Converting $OMEGA_CSV to SequenceFile format"

    $MAHOUT org.apache.mahout.completion.svt.conversion.DistributedMatrixFromCsv --input $OMEGA_CSV --output $OMEGA_SEQ
 
  fi

  # run the SVT
  echo "Running SVT matrix completion"
#  $MAHOUT org.apache.mahout.completion.svt.SVTEvaluator --input $M_SEQ --output $SVTOUTPUT_SEQ --tempDir $CSV_TMP --numRows 150 --numCols 300 --omega $OMEGA_SEQ
#  removeFolder "$CSV_TMP"

  
  # commenting out for the moment..don't need this quite yet
    echo "Converting output back to csv"
    $MAHOUT org.apache.mahout.completion.svt.conversion.DistributedMatrixToCsv --input $SVTOUTPUT_SEQ --output $SVTOUTPUT_CSV



#netflix
elif [ "x$alg" == "xnetflix" ]; then

  #DO STUFF
  exit

#yahoo kddcup
elif [ "x$alg" == "xyahoo" ]; then

  #DO STUFF
  exit

elif [ "x$alg" == "xclean" ]; then
  echo "Are you sure you really want to remove all files under $OUT:"
  read -p "Enter your choice (y/n): " answer
  if [ "x$answer" == "xy" ] || [ "x$answer" == "xY" ]; then
    echo "Cleaning out $OUT";
	removeFolder "$OUT"
  fi
fi
