/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.completion.svt.conversion;

import com.google.common.io.Closeables;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Converts a CSV file into a SequenceFile format compatible with {@link DistributedMatrix}.
 */
public final class DistributedMatrixFromCsv extends AbstractJob {

  private static final Pattern SPLITTER = Pattern.compile("[ ,\t]*[,|\t][ ,\t]*");

  private static final Logger log = LoggerFactory.getLogger(
  		DistributedMatrixFromCsv.class);
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new DistributedMatrixFromCsv(), args);
  }

  public void convertCSV(Path inputPath, Path outputPath) throws IOException {
    Configuration conf = new Configuration(getConf());
    FileSystem fs = FileSystem.get(inputPath.toUri(), conf);
    SequenceFile.Writer writer = null;
    File csvFile = new File(inputPath.toString());
    int row=0;
    Pattern splitter = Pattern.compile(this.SPLITTER.toString());
    
    try {  	
      writer = new SequenceFile.Writer(fs,
          conf,
          outputPath,
          IntWritable.class,
          VectorWritable.class);
      
      for (String line : new FileLineIterable(csvFile)) {
      	String[] entries = splitter.split(line);
	      Vector v = new RandomAccessSparseVector(entries.length);
	      for (int m = 0; m < entries.length; m++) {
	        v.setQuick(m, Double.parseDouble(entries[m]));
	      }
	      writer.append(new IntWritable(row++), new VectorWritable(v));     
      }
      
      log.info("Converted {} rows", row);

    } finally {
    	Closeables.closeQuietly(writer);
    }
  }


  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    
    Map<String, List<String>> argMap = parseArguments(args);
    if (argMap == null) {
      return -1;
    }

    convertCSV(getInputPath(), getOutputPath());
    
    return 0;
  }
}
