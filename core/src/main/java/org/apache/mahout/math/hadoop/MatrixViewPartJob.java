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

package org.apache.mahout.math.hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.join.CompositeInputFormat;
import org.apache.hadoop.mapred.join.TupleWritable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class MatrixViewPartJob extends AbstractJob {

	//assuming 0-based indices
  private static final String ROW_IDX_START = "DistributedMatrix.MatrixViewPart.ROW_IDX_START";
  private static final String ROW_IDX_END = "DistributedMatrix.MatrixViewPart.ROW_IDX_END";
  private static final String COL_IDX_START = "DistributedMatrix.MatrixViewPart.COL_IDX_START";
  private static final String COL_IDX_END = "DistributedMatrix.MatrixViewPart.COL_IDX_END";

	public static Configuration createMatrixViewPartJobConf(Configuration initialConf, Path matrixPath, int rowIdxStart, int rowIdxEnd,
																									int colIdxStart, int colIdxEnd, Path outPath) {
    JobConf conf = new JobConf(initialConf, MatrixViewPartJob.class);
    conf.setInputFormat(SequenceFileInputFormat.class);
    FileInputFormat.setInputPaths(conf, matrixPath);    
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.set(ROW_IDX_START, Integer.toString(rowIdxStart));
    conf.set(ROW_IDX_END, Integer.toString(rowIdxEnd));
    conf.set(COL_IDX_START, Integer.toString(colIdxStart));
    conf.set(COL_IDX_END, Integer.toString(colIdxEnd));
    FileOutputFormat.setOutputPath(conf, outPath);
    conf.setMapperClass(MatrixViewPartMapper.class);
    conf.setMapOutputKeyClass(IntWritable.class);
    conf.setMapOutputValueClass(VectorWritable.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(VectorWritable.class);
    conf.setNumReduceTasks(0);
    return conf;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new MatrixViewPartJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    addOption("numRows", "nr", "Number of rows of the input matrix", true);
    addOption("numCols", "nc", "Number of columns of the input matrix", true);
    addOption("rowIdxStart", "rs", "Start row index - using 0-based index", true);
    addOption("rowIdxEnd", "re", "End row index - using 0-based index", true);
    addOption("colIdxStart", "cs", "Start col index - using 0-based index", true);
    addOption("colIdxEnd", "ce", "End col index - using 0-based index", true);

    Map<String, List<String>> argMap = parseArguments(strings);
    if (argMap == null) {
      return -1;
    }

    DistributedRowMatrix m = new DistributedRowMatrix(new Path(getOption("inputPath")),
                                                      new Path(getOption("tempDir")),
                                                      Integer.parseInt(getOption("numRows")),
                                                      Integer.parseInt(getOption("numCols")));
    
    int rowIdxStart = Integer.parseInt(getOption("rowIdxStart"));
    int rowIdxEnd = Integer.parseInt(getOption("rowIdxEnd"));
    int colIdxStart = Integer.parseInt(getOption("colIdxStart"));
    int colIdxEnd = Integer.parseInt(getOption("colIdxEnd"));
    
    m.setConf(new Configuration(getConf()));

    m.viewPart(rowIdxStart, rowIdxEnd, colIdxStart, colIdxEnd);
    return 0;
  }

  public static class MatrixViewPartMapper extends MapReduceBase
      implements Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private int rowIdxStart;
    private int rowIdxEnd;
    private int colIdxStart;
    private int colIdxEnd;

    @Override
    public void configure(JobConf conf) {
    	rowIdxStart = Integer.valueOf(conf.get(ROW_IDX_START));
    	rowIdxEnd = Integer.valueOf(conf.get(ROW_IDX_END));
    	colIdxStart = Integer.valueOf(conf.get(COL_IDX_START));
    	colIdxEnd = Integer.valueOf(conf.get(COL_IDX_END));
    }

    @Override
    public void map(IntWritable index,
    								VectorWritable v,
                    OutputCollector<IntWritable,VectorWritable> out,
                    Reporter reporter) throws IOException {

      int rowIdx = index.get();
      if(rowIdx >= rowIdxStart && rowIdx <= rowIdxEnd) {
      	Vector vNew = new SequentialAccessSparseVector(colIdxEnd-colIdxStart+1);
      	Vector vPart = v.get().viewPart(colIdxStart, colIdxEnd-colIdxStart+1);
      	
        Iterator<Element> vPartIterator = vPart.iterateNonZero();
        
        while(vPartIterator.hasNext()) {
        	Element e = vPartIterator.next();
        	vNew.setQuick(e.index(), e.get());
        }
      	
        out.collect( new IntWritable(rowIdx-rowIdxStart), new VectorWritable(vNew) );      	
      }
      
      
    }
  }




}
