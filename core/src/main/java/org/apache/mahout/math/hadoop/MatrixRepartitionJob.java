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

import java.io.IOException;

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
import org.apache.hadoop.mapred.lib.IdentityMapper;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VectorWritable;

public class MatrixRepartitionJob extends AbstractJob {
	

  public static Configuration createMatrixRepartitionJobConf(Configuration initialConf, 
  																												Path path,  
                                                          Path outPath, 
                                                          int numPartitions) {  	
  	JobConf conf = new JobConf(initialConf, MatrixRepartitionJob.class);
    conf.setJobName("MatrixRepartitionJob: " + path + " (numPartitions=" + numPartitions + ") -> " + outPath); 	
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    FileInputFormat.setInputPaths(conf, path);
    FileOutputFormat.setOutputPath(conf, outPath);
    conf.setMapperClass(IdentityMapper.class);
    conf.setReducerClass(IdentityReducer.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(VectorWritable.class);
    conf.setNumReduceTasks(numPartitions);
    return conf;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new MatrixRepartitionJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    throw new Exception("Not implemented");
  }

  public static class MatrixRepartitionMapper extends MapReduceBase
	  implements Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {
				
		@Override
		public void map(IntWritable index,
										VectorWritable v,
		                OutputCollector<IntWritable,VectorWritable> out,
		                Reporter reporter) throws IOException {
		
		  out.collect(index, v);
	}
}
}
