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
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class MatrixScalarMultiplicationJob extends AbstractJob {

  private static final String SCALAR = "DistributedMatrix.MatrixScalarMultiply.Scalar";

  public static Configuration createMatrixScalarMultiplyJobConf(Configuration initialConf, 
                                                          Path matrixPath, 
                                                          double scalar, 
                                                          Path outPath) {
    JobConf conf = new JobConf(initialConf, MatrixScalarMultiplicationJob.class);
    conf.setJobName("MatrixScalarMultiplicationJob: " + matrixPath + " (scalar=" + scalar + ") -> " + outPath); 	
    conf.setInputFormat(SequenceFileInputFormat.class);
    FileInputFormat.setInputPaths(conf, matrixPath);    
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.set(SCALAR, Double.toString(scalar));
    FileOutputFormat.setOutputPath(conf, outPath);
    conf.setMapperClass(MatrixScalarMultiplyMapper.class);
    conf.setMapOutputKeyClass(IntWritable.class);
    conf.setMapOutputValueClass(VectorWritable.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(VectorWritable.class);

    return conf;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new MatrixScalarMultiplicationJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    addOption("numRows", "nr", "Number of rows of the input matrix", true);
    addOption("numCols", "nc", "Number of columns of the input matrix", true);
    addOption("scalar", "s", "Scalar value", true);

    Map<String, List<String>> argMap = parseArguments(strings);
    if (argMap == null) {
      return -1;
    }

    DistributedRowMatrix m = new DistributedRowMatrix(new Path(getOption("inputPath")),
                                                      new Path(getOption("tempDir")),
                                                      Integer.parseInt(getOption("numRows")),
                                                      Integer.parseInt(getOption("numCols")));

    double s = Double.parseDouble(getOption("scalar"));
    
    m.setConf(new Configuration(getConf()));

    m.times(s);
    return 0;
  }

  public static class MatrixScalarMultiplyMapper extends MapReduceBase
      implements Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private double scalar;

    @Override
    public void configure(JobConf conf) {
    	scalar = Double.valueOf(conf.get(SCALAR));
    }

    @Override
    public void map(IntWritable index,
    								VectorWritable v,
                    OutputCollector<IntWritable,VectorWritable> out,
                    Reporter reporter) throws IOException {

      VectorWritable scaledVector = new VectorWritable(v.get().times(scalar));
      out.collect(index, scaledVector);
    }
  }


}
