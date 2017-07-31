package mine;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import mine.CountAndReverseMapper.Counters;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hadoop.mapreduce.Reducer;

import util.Pair;

/**
 * sort the value order by: 1) according to the support in descending order, 2)
 * if the same support,then according to the user id ascending example: input:
 * <a2 ,{u1:3,u4:2,u2:3,u3:4}> output:<null,{u3:4,u1:3,u2:3,u4:2}>
 */
public class CountAndReverseReducer extends
		Reducer<Text, Pair, NullWritable, Text> {

	private NullWritable outputKey = NullWritable.get();
	private Text outputValue = new Text();
	protected int failedRecords;

	public static enum Counters {
		R1_FAILED_RECORDS
	}

	@Override
	protected void reduce(Text key, Iterable<Pair> pairs, Context context)
			throws IOException, InterruptedException {

		try {
			Configuration conf = context.getConfiguration();
			// clone the pairs
			List<Pair> pairList = new ArrayList<Pair>();
			for (Pair pair : pairs) {
				pairList.add(WritableUtils.clone(pair, conf));
			}

			// sort the pair list
			Collections.sort(pairList);

			// output the sorted pair list
			StringBuilder buff = new StringBuilder(pairList.size());
			for (Pair pair : pairList) {
				buff.append(pair.toString() + ' ');
			}

			outputValue.set(buff.toString());
			context.write(outputKey, outputValue);
		} catch (Exception e) {
			processError(context, e);
		}
	}

	private void processError(Context context, Exception e) {
		// increase the failed records
		context.getCounter(Counters.R1_FAILED_RECORDS).increment(1);
		context.setStatus("M1 Records with failures= " + (++failedRecords));

	}
}
