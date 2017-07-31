package run;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import mine.CountAndReverseMapper;
import mine.CountAndReverseReducer;
import mine.ParallelFPMineMapper;
import mine.ParallelFPMineReducer_multithread;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import util.Pair;
import util.Parameters;

public class Run_multithread {

	public static final String EDGE_WEIGHT_PARAMETERS = "edgeWeight.parameters";
	public static final String INPUT = "input";
	public static final String OUTPUT = "output";
	public static final String G_LIST = "gList";
	public static final String PARALLEL_COUNTING = "parallelcounting";
	public static final String TWO_FREQUENT_PATTERNS = "twofrequentpatterns";
	public static final String FILE_PATTERN = "part-*";
	public static final String NUM_REDUCES = "numreduces";
	public static final int NUM_GROUPS_DEFAULT = 100;
	public static final String NUM_GROUP = "numgroup";
	public static final String MAX_PER_GROUP = "maxPerGroup";
	public static final String GROUP_OPTION = "groupoption";
	public static final String PASCAL = "0.6f";
	public static long diskUsage = 0;
	public static final long MB = 1 << 20;

	// inputpath--> hdfs://hadoop-master:9000/usr/test/edgeweight/input

	public static void main(String[] args) throws IOException,
			InterruptedException, ClassNotFoundException {

		Configuration conf = new Configuration();

		Parameters params = new Parameters();
		params.set(INPUT, args[0]);
		params.set(OUTPUT, args[1]);
		params.set(NUM_GROUP, args[2]);
		params.set(NUM_REDUCES, args[3]);
		// option 1: group by /
		// option 2: group by %
		// option zipf: group by zipf
		// option other integer: group by pascal
		params.set(GROUP_OPTION, args[4]);

		Date startTime = new Date();

		// step 1: start parallel counting (MR job1)
		startCountingAndReverse(conf, params);

		// step 2: read, filter and save the parallel counting job1 result
		Date startTimeFList = new Date();
		List<Pair> fList = readParallelCountResult(params);
		
		// set parameter to control group size in MR job2
		int numGroups = params.getInt(NUM_GROUP, NUM_GROUPS_DEFAULT);
		int maxPerGroup = fList.size() / numGroups;
		if (fList.size() % numGroups != 0) {
			maxPerGroup++;
		}
		params.set(MAX_PER_GROUP, Integer.toString(maxPerGroup));
		
		generalGListAndsave(fList, conf, params);
		Date end_timeFList = new Date();
		System.out.println(" read and save fList time: "
				+ (end_timeFList.getTime() - startTimeFList.getTime()) / 1000
				+ " seconds."); // end step2

		// step 3: start parallel mine two frequent itemset (MR job2)
		startTwoFrequentPatterns(conf, params);

		Date end_time = new Date();
		System.out.println("The job total took: "
				+ (end_time.getTime() - startTime.getTime()) / 1000
				+ " seconds.");

		System.out.println("total FILE_BYTES_WRITTEN--->> " + (diskUsage * 1.0)
				/ MB + " MB");
	}

	private static void generalGListAndsave(List<Pair> fList, Configuration conf,
			Parameters params) throws IOException {

		Path gListPath = new Path(params.get(OUTPUT), G_LIST);
		FileSystem fs = FileSystem.get(gListPath.toUri(), conf);
		fs.delete(gListPath);
		FSDataOutputStream writer = fs.create(gListPath);
		int groupOption = params.getInt(GROUP_OPTION, 0);
		int numGroup = params.getInt(NUM_GROUP, 0);
		int maxPerGroup = params.getInt(MAX_PER_GROUP, 0);
		for (int i = 0; i < fList.size(); i++) {
			if (groupOption == 1) {
				writer.writeBytes(fList.get(i).toString() + "\t"
						+ (i / maxPerGroup) + "\n");
			}
			if (groupOption == 2) {
				writer.writeBytes(fList.get(i).toString() + "\t"
						+ (i % numGroup) + "\n");
			}
		}
		writer.close();

	}

	/**
	 * read the feature frequency List which is built at the end of the Parallel
	 * counting MR job1
	 */
	public static List<Pair> readParallelCountResult(Parameters params)
			throws IOException {

		PriorityQueue<Pair> queue = new PriorityQueue<Pair>(11,
				new Comparator<Pair>() {

					@Override
					public int compare(Pair o1, Pair o2) {
						int compare = o2.getSecond().compareTo(o1.getSecond());
						if (compare == 0) {
							return o1.getFirst().compareTo(o2.getFirst());
						}
						return compare;
					}
				});

		// read parallel count result part-* files
		Configuration conf = new Configuration();
		Path parallelPath = new Path(params.get(OUTPUT), PARALLEL_COUNTING);
		FileSystem fs = FileSystem.get(parallelPath.toUri(), conf);
		FileStatus[] files = fs
				.globStatus(new Path(parallelPath, FILE_PATTERN));

		FSDataInputStream in;
		BufferedReader buff;
		// deduplication the same pair
		Set<String> pairSet = new HashSet<String>();
		for (FileStatus fileStatus : files) {
			in = fs.open(fileStatus.getPath());
			buff = new BufferedReader(new InputStreamReader(in));
			String line = null;
			while ((line = buff.readLine()) != null) {
				String[] pairString = line.split(" ");
				for (String pairstr : pairString) {
					pairSet.add(pairstr);
				}
			}
			buff.close();
			in.close();
		}

		// sort the pair list
		// 1) split
		// input: Set<String>, String=uid:support
		// output: Pair<uid,support>
		for (String pair : pairSet) {
			String[] pairString = pair.split(":");
			queue.add(new Pair(pairString[0], Integer.parseInt(pairString[1])));
		}
		// 2) sort
		List<Pair> fList = new ArrayList<Pair>();
		while (!queue.isEmpty()) {
			fList.add(queue.poll());
		}

		return fList;
	}

	/**
	 * read DistributeCache fList file
	 */
	public static Map<String, Integer> readDistributeCacheFListFile(Configuration conf)
			throws FileNotFoundException, IOException {

		Map<String, Integer> mapItemToGroupID = new HashMap<String, Integer>();

		// read fList file in distribute cache
		Path[] paths = DistributedCache.getLocalCacheFiles(conf);
		BufferedReader buff = new BufferedReader(new FileReader(
				paths[0].toString()));

		String line = null;
		while ((line = buff.readLine()) != null) {
			String[] split = line.split("[ |\t]");
			mapItemToGroupID.put(split[0], Integer.parseInt(split[1]));
		}
		buff.close();

		return mapItemToGroupID;
	}

	/**
	 * Count the frequencies of various features in parallel using Map/Reduce
	 */
	private static void startCountingAndReverse(Configuration conf,
			Parameters params) throws IOException, InterruptedException,
			ClassNotFoundException {

		Date startTime = new Date();

		conf.set("dfs.replication", "1"); // default: 3
		// conf.set("dfs.block.size", "134217728 "); // default: 64mb
		// conf.set("io.file.buffer.size", "131072"); // default: 4kb

		conf.set(EDGE_WEIGHT_PARAMETERS, params.toString());

		Job job = Job.getInstance(conf, "parallel counting");
		job.setJarByClass(Run_multithread.class);

		job.setNumReduceTasks(1);

		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Pair.class);

		job.setOutputKeyClass(NullWritable.class);
		job.setOutputValueClass(Text.class);

		FileInputFormat.addInputPath(job, new Path(params.get(INPUT)));
		Path outPath = new Path(params.get(OUTPUT), PARALLEL_COUNTING);
		FileOutputFormat.setOutputPath(job, outPath);

		FileSystem fs = outPath.getFileSystem(conf);
		fs.delete(outPath);

		job.setMapperClass(CountAndReverseMapper.class);
		job.setReducerClass(CountAndReverseReducer.class);

		boolean succeeded = job.waitForCompletion(true);
		if (!succeeded) {
			throw new IllegalStateException("Job failed!");
		} else {
			long writtenBytes = job.getCounters()
					.findCounter("FileSystemCounters", "FILE_BYTES_WRITTEN")
					.getValue();
			diskUsage += writtenBytes;
		}

		Date end_time = new Date();
		System.out.println("The parallel count job took: "
				+ (end_time.getTime() - startTime.getTime()) / 1000
				+ " seconds.");

	}

	/**
	 * calculate the two itemset frequency of group dependent shards
	 */
	private static void startTwoFrequentPatterns(Configuration conf,
			Parameters params) throws IOException, InterruptedException,
			ClassNotFoundException {

		Date startTime = new Date();

		conf.set(EDGE_WEIGHT_PARAMETERS, params.toString());
		conf.set("dfs.replication", "1"); // default: 3 hdfs-site.xml
		// conf.set("mapred.compress.map.output", "true");
		// conf.set("mapred.output.compression.type", "BLOCK");

		// add aList file to distribute cache system
		Path fListPath = new Path(params.get(OUTPUT), G_LIST);
		DistributedCache.addCacheFile(fListPath.toUri(), conf);

		Job job = Job.getInstance(conf, "parallel mine");
		job.setJarByClass(Run_multithread.class);

		job.setNumReduceTasks(Integer.parseInt(params.get(NUM_REDUCES)));

		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(Text.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		FileInputFormat.addInputPath(job, new Path(new Path(params.get(OUTPUT),
				PARALLEL_COUNTING), FILE_PATTERN));
		Path outPath = new Path(params.get(OUTPUT), TWO_FREQUENT_PATTERNS);
		FileOutputFormat.setOutputPath(job, outPath);

		// FileSystem fs = FileSystem.get(conf);
		FileSystem fs = outPath.getFileSystem(conf);
		fs.delete(outPath);

		job.setInputFormatClass(TextInputFormat.class);
		job.setMapperClass(ParallelFPMineMapper.class);
		job.setReducerClass(ParallelFPMineReducer_multithread.class);
		// job.setOutputFormatClass(SequenceFileOutputFormat.class);

		boolean succeeded = job.waitForCompletion(true);
		if (!succeeded) {
			throw new IllegalStateException("Job failed!");
		} else {
			long writtenBytes = job.getCounters()
					.findCounter("FileSystemCounters", "FILE_BYTES_WRITTEN")
					.getValue();
			diskUsage += writtenBytes;
		}

		Date end_time = new Date();
		System.out.println("The parallel mine job took: "
				+ (end_time.getTime() - startTime.getTime()) / 1000
				+ " seconds.");
	}

	public static Map<String, Integer> readGList(Parameters params)
			throws IOException {

		Configuration conf = new Configuration();
		// read file
		Path inputPath = new Path(params.get(OUTPUT), G_LIST);
		FileSystem fs = FileSystem.get(inputPath.toUri(), conf);
		FSDataInputStream in = fs.open(inputPath);
		InputStreamReader reader = new InputStreamReader(in);
		BufferedReader buff = new BufferedReader(reader);

		String line = null;
		Map<String, Integer> mapItemToGroupID = new HashMap<String, Integer>();
		while ((line = buff.readLine()) != null) {
			String[] split = line.split("[ |\t]");
			mapItemToGroupID.put(split[0], Integer.parseInt(split[1]));
		}

		buff.close();
		return mapItemToGroupID;
	}

}
