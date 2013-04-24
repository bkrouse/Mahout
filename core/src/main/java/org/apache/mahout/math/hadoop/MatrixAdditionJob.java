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

public class MatrixAdditionJob extends AbstractJob {
	
  private static final String MULTIPLIER = "DistributedRowMatrix.matrixAddition.multiplier";


  public static Configuration createMatrixAdditionJobConf(Configuration initialConf, 
  																												Path aPath, 
                                                          Path bPath, 
                                                          Path outPath, 
                                                          double multiplier) {
    JobConf conf = new JobConf(initialConf, MatrixAdditionJob.class);
    conf.setInputFormat(CompositeInputFormat.class);
    conf.set("mapred.join.expr", CompositeInputFormat.compose(
          "inner", SequenceFileInputFormat.class, aPath, bPath));
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.set(MULTIPLIER, Double.toString(multiplier));
    FileOutputFormat.setOutputPath(conf, outPath);
    conf.setMapperClass(MatrixAdditionMapper.class);
    conf.setMapOutputKeyClass(IntWritable.class);
    conf.setMapOutputValueClass(VectorWritable.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(VectorWritable.class);
    conf.setNumReduceTasks(0);
    return conf;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new MatrixAdditionJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    throw new Exception("Not implemented");
  }

  public static class MatrixAdditionMapper extends MapReduceBase
      implements Mapper<IntWritable,TupleWritable,IntWritable,VectorWritable> {

    
  	private double multiplier;
    

    @Override
    public void configure(JobConf conf) {
      multiplier = Double.parseDouble(conf.get(MULTIPLIER, "1.0"));
    }

    @Override
    public void map(IntWritable index,
                    TupleWritable v,
                    OutputCollector<IntWritable,VectorWritable> out,
                    Reporter reporter) throws IOException {
      Vector fragA = ((VectorWritable)v.get(0)).get();
      Vector fragB = ((VectorWritable)v.get(1)).get();

            
      Vector outVector = new RandomAccessSparseVector(fragA.size());
      for(int i=0; i<fragA.size(); i++)
      {
      	double outValue = fragA.getQuick(i) + fragB.getQuick(i)*multiplier;
      	outVector.setQuick(i, outValue);
      }
            
      out.collect(index, new VectorWritable(new SequentialAccessSparseVector(outVector)));
    }
  }


}
