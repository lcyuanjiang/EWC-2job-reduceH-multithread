package mine;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import run.Run_multithread;
import util.Parameters;

/**
 * maps each transaction to all unique items groups in the transaction. mapper
 * outputs the group id as key and the transaction as value
 * 
 */
public class ParallelFPMineMapper extends
		Mapper<Object, Text, IntWritable, Text> {
	// output key
	private IntWritable outputGroupId = new IntWritable();
	// output value
	private Text outputTransaction = new Text();

	// Map<uid:support, groupID>
	private Map<String, Integer> mapItemToGroupID = new HashMap<String, Integer>();

	@Override
	protected void map(Object key, Text value, Context context)
			throws IOException, InterruptedException {

		String[] itemtoken = (value.toString()).split("[ |\t]");
		Set<Integer> groups = new HashSet<Integer>();
		for (int i = itemtoken.length - 1; i >= 0; i--) {

			int groupId = mapItemToGroupID.get(itemtoken[i]);

			// generate group dependent shards
			if (!groups.contains(groupId)) {

				// copy the string from 0 to i index
				StringBuilder buff = new StringBuilder(i);
				for (int j = 0; j <= i; j++) {
					buff.append(itemtoken[j] + ' ');
				}
				outputGroupId.set(groupId);
				outputTransaction.set(buff.toString());
				context.write(outputGroupId, outputTransaction);
			}
			groups.add(groupId);
		}

	}

	@Override
	protected void setup(Context context) throws IOException,
			InterruptedException {

		Parameters params = new Parameters(context.getConfiguration().get(
				Run_multithread.EDGE_WEIGHT_PARAMETERS, ""));

		// read the distributed cache fList file
		mapItemToGroupID = Run_multithread.readDistributeCacheFListFile(context
				.getConfiguration());

		// for eclipse local test
//		mapItemToGroupID=Run_multithread.readGList(params);
	}

}
