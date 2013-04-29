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

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.zip.GZIPOutputStream;

/**
 * Converts a SequenceFile format compatible with {@link DistributedMatrix} into a Dat file that can be imported into Matlab.
 * Input file -- a local file system path to a csv.  
 * Output file -- an HDFS path to a sequence file
 */
public final class DistributedMatrixToCsv extends AbstractJob {

  private static final String DELIMITER = ",";

  private static final Logger log = LoggerFactory.getLogger(
  		DistributedMatrixToCsv.class);
  
  public static void main(String[] args) throws Exception {
  	

    ToolRunner.run(new Configuration(), new DistributedMatrixToCsv(), args);
  }

  public void createCsv(Path inputPath, Path outputPath) throws IOException {

  	FileSystem fs = outputPath.getFileSystem(getConf());
  	fs.delete(outputPath);
  	OutputStream outStream = new FileOutputStream(new File(outputPath.toString()));
    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(outStream, Charsets.UTF_8));

    try {
    	
      SequenceFileIterator<IntWritable,VectorWritable> seqIterator =
        new SequenceFileIterator<IntWritable,VectorWritable>(inputPath, true, getConf());
      
      while(seqIterator.hasNext()) {
      	Pair<IntWritable,VectorWritable> record = seqIterator.next();
      	
      	//write the row index
      	IntWritable iw = record.getFirst();
      	Integer row = iw.get();
      	writer.write(String.valueOf(row));
      	writer.write(DELIMITER);
      	
      	VectorWritable vw = record.getSecond();      	
      	Vector vector = vw.get(); 
 
      	//this could be sparse...not sure it's efficient to access in this fashion, but oh well
      	int numCols = vector.size();
      	for ( int i = 0; i < numCols; i++ ) {
      		Vector.Element e = vector.getElement(i);
          writer.write(String.valueOf(e.get()));
          if( i < (numCols-1) )
          	writer.write(DELIMITER);

      	}
      	      	
      	if(seqIterator.hasNext())
      		writer.newLine();
      }

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

    createCsv(getInputPath(), getOutputPath());
    
    return 0;
  }
}
