package mine;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import util.Pair;

/**
 * input: u1 a1 a2 a4 output: <a1,u1:3> <a2,u1:3> <a4,u1:3>
 */
public class CountAndReverseMapper extends Mapper<Object, Text, Text, Pair> {

	protected Text outputKey = new Text();
	protected Pair outputValue = new Pair();
	protected int failedRecords;

	public static enum Counters {
		M1_FAILED_RECORDS
	}

	@Override
	protected void map(Object key, Text value, Context context)
			throws IOException, InterruptedException {

		try {
			StringTokenizer stringToken = new StringTokenizer(value.toString());

			outputValue.setSecond(stringToken.countTokens() - 1);
			outputValue.setFirst(stringToken.nextToken());

			while (stringToken.hasMoreTokens()) {
				outputKey.set(stringToken.nextToken());
				context.write(outputKey, outputValue);

			}
		} catch (Exception e) {
			processError(context, e);
		}
	}

	private void processError(Context context, Exception e) {
		// increase the failed records
		context.getCounter(Counters.M1_FAILED_RECORDS).increment(1);
		context.setStatus("M1 Records with failures= " + (++failedRecords));
	}

}
