package mine;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;

import run.Run_multithread;
import util.Pair;
import util.Parameters;
import fptree.FPTree;
import fptree.FPTreeNode;

public class ParallelFPMineReducer_multithread extends
		Reducer<IntWritable, Text, Text, Text> {

	public static String NUM_THREADS = "mapreduce.reduce.multithreadedreduce.threads";

	// Map<uid:support, groupID>
	private Map<String, Integer> mapItemToGroupID = new HashMap<String, Integer>();

	private FPTree fptree;
	private int groupID;
	private Context outer;
	private List<SubReduceRunner> runners;

	public static int getNumberOfThreads(Context job) {
		return job.getConfiguration().getInt(NUM_THREADS, 18);
	}

	public static void setNumberOfThreads(Job job, int threads) {
		job.getConfiguration().setInt(NUM_THREADS, threads);
	}

	protected void reduce(IntWritable key, Iterable<Text> values,
			Context context) throws IOException, InterruptedException {

		groupID = key.get();
		outer = context;

		Date startCreateFPTree = new Date();

		// 1> create the local fp tree
		fptree = new FPTree();
		Map<String, Integer> mapItemSupport = createFPtree(mapItemToGroupID,
				values);

		// 3> sort the local header table
		fptree.sortHeaderTable(mapItemSupport);
		Date endCreateFPTree = new Date(); // end create local fp tree

		// print groupId statistics *********
		System.out.println("groupId: " + key.get() + " statistics ----->>");

		Date startMineFPTree = new Date();
		// 4> mine two frequent itemset
		mineTwoFrequentPatterns();
		Date endMineFPTree = new Date();

		// print create local tree time
		System.out.println("create fptree took: "
				+ (endCreateFPTree.getTime() - startCreateFPTree.getTime())
				/ 1000 + " seconds.");

		// print mine took time
		System.out.println("mine fptree took: "
				+ (endMineFPTree.getTime() - startMineFPTree.getTime()) / 1000
				+ " seconds.");
	}

	private void mineTwoFrequentPatterns() throws InterruptedException,
			IOException {

		int numberOfThreads = getNumberOfThreads(outer);
		if (numberOfThreads > fptree.getHeaderItemList().size()) {
			numberOfThreads = fptree.getHeaderItemList().size();
		}
		runners = new ArrayList<SubReduceRunner>(numberOfThreads);

		int[][] threadGroupList = generateThreadGroupList(numberOfThreads);

		for (int i = 0; i < numberOfThreads; i++) {
			SubReduceRunner thread = new SubReduceRunner(threadGroupList[i]);
			thread.start();
			runners.add(i, thread);
		}

		for (int i = 0; i < numberOfThreads; i++) {
			SubReduceRunner thread = runners.get(i);
			thread.join();
		}

		// print
		// List<String> headerTable = fptree.getHeaderItemList();
		// System.out.println("header table ++++++++++++++++++++");
		// int count = 0;
		// for (String h : headerTable) {
		// System.out.println(count + " " + h);
		// count++;
		// }
		//
		// System.out.println("thread array ++++++++++++++++++++");
		// for (int i = 0; i < threadGroupList.length; i++) {
		// System.out.println("thread " + i);
		// for (int j = 0; j < threadGroupList[i].length; j++) {
		// System.out.print(threadGroupList[i][j] + " ");
		// }
		// System.out.println();
		// for (int j = 0; j < threadGroupList[i].length; j++) {
		// System.out.print(headerTable.get(threadGroupList[i][j])+" ");
		// }
		// System.out.println();
		// }
		// end print
	}

	private int[][] generateThreadGroupList(int numberOfThreads) {
		int headerTableSize = fptree.getHeaderItemList().size();
		int itemNumPerThread = (headerTableSize / numberOfThreads) + 1;
		int threadGroupList[][] = new int[numberOfThreads][itemNumPerThread];
		for (int i = 0; i < numberOfThreads; ++i) {
			int k = 0;
			for (int j = 0; j < headerTableSize; ++j) {
				if (i == j % numberOfThreads) {
					threadGroupList[i][k++] = j + 1;
				}
			}
		}
		return threadGroupList;
	}

	private Map<String, Integer> createFPtree(
			Map<String, Integer> mapItemToGroupID2, Iterable<Text> values) {
		// Map <item, item support> based on group
		Map<String, Integer> mapItemSupport = new HashMap<String, Integer>();
		// for each local transaction
		for (Text value : values) {
			List<String> transaction = new ArrayList<String>();
			StringTokenizer itemtoken = new StringTokenizer(value.toString());

			while (itemtoken.hasMoreTokens()) {
				String item = itemtoken.nextToken();
				// check the item is in the group list
				if (mapItemToGroupID2.get(item) == groupID) {
					// calculate the item and its support
					Integer count = mapItemSupport.get(item);
					if (count == null) {
						mapItemSupport.put(item, 1);
					} else {
						// increase the count
						mapItemSupport.put(item, ++count);
					}
				}
				transaction.add(item);
			}
			// insert the transaction to the fp tree
			fptree.addTransaction(transaction, mapItemToGroupID2, groupID);
		}
		return mapItemSupport;
	}

	private class SubReduceRunner extends Thread {

		private Throwable throwable;
		int[] threadGroupList;

		public SubReduceRunner(int[] threadGroupList) {
			this.threadGroupList = threadGroupList;
		}

		@Override
		public void run() {

			for (int i = 0; i < threadGroupList.length; i++) {

				if (threadGroupList[i] == 0) {
					continue;
				}
				// a) calculate the frequency of each node in the
				// prefixPaths
				Map<String, Integer> mapItemSupportOfSuffix = new HashMap<String, Integer>();

				// current suffix item
				String suffixItem = fptree.getHeaderItemList().get(
						threadGroupList[i] - 1);
				FPTreeNode pathNode = fptree.getMapItemNode().get(suffixItem);

				// find all prefixPath of the suffix item
				while (pathNode != null) {

					// the support of the prefixPath's first node
					int pathSupport = pathNode.getCount();

					// check the node is not just the root
					if (pathNode.getParent().getName() != null) {

						// recursive add all the parent of this node
						FPTreeNode parent = pathNode.getParent();
						while (parent.getName() != null) {

							// calculate the frequency of each node in the
							// prefixPaths
							if (mapItemSupportOfSuffix.get(parent.getName()) == null) {
								mapItemSupportOfSuffix.put(parent.getName(),
										pathSupport);
							} else {
								// make the sum
								int count = mapItemSupportOfSuffix.get(parent
										.getName());
								mapItemSupportOfSuffix.put(parent.getName(),
										count + pathSupport);
							}
							parent = parent.getParent();
						}
					}
					// look for the next prefixPath
					pathNode = pathNode.getNodeLink();
				}

				// b) context write
				Set<String> itemset = mapItemSupportOfSuffix.keySet();
				if (itemset.size() > 0) {

					// suffix item
					String[] suffixItemSplit = suffixItem.split(":");
					String suffixItemName = suffixItemSplit[0];
					float suffixItemSupport = Integer
							.parseInt(suffixItemSplit[1]);

					StringBuilder buff = new StringBuilder(itemset.size());

					// for each prefixPaths item of the current suffix item
					for (String prefixItem : itemset) {

						// prefix item
						String[] prefixItemSplit = prefixItem.split(":");
						String prefixItemName = prefixItemSplit[0];
						float prefixItemSupport = Integer
								.parseInt(prefixItemSplit[1]);

						// two itemset support
						float twoItemsetSupport = mapItemSupportOfSuffix
								.get(prefixItem);

						// calculate the two itemset's weight
						float weight = (float) (Math
								.round((twoItemsetSupport / (prefixItemSupport
										+ suffixItemSupport - twoItemsetSupport)) * 100)) / 100;

						buff.append(prefixItemName + ',' + weight + ' ');
					}

					try {
						synchronized (outer) {
							outer.write(new Text(suffixItemName),
									new Text(buff.toString()));
						}
					} catch (IOException e) {
						e.printStackTrace();
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
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
//		mapItemToGroupID = Run_multithread.readGList(params);
	}
}
