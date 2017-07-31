package fptree;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import fptree.FPTreeNode;

public class FPTree {

	List<String> headerItemList = null;

	Map<String, FPTreeNode> mapItemNode = new HashMap<String, FPTreeNode>();

	Map<String, FPTreeNode> mapItemLastnode = new HashMap<String, FPTreeNode>();

	FPTreeNode root = new FPTreeNode();

	public FPTree() {

	}

	public List<String> getHeaderItemList() {
		return headerItemList;
	}

	public void setHeaderItemList(List<String> headerItemList) {
		this.headerItemList = headerItemList;
	}

	public Map<String, FPTreeNode> getMapItemNode() {
		return mapItemNode;
	}

	public void setMapItemNode(Map<String, FPTreeNode> mapItemNode) {
		this.mapItemNode = mapItemNode;
	}

	public Map<String, FPTreeNode> getMapItemLastnode() {
		return mapItemLastnode;
	}

	public void setMapItemLastnode(Map<String, FPTreeNode> mapItemLastnode) {
		this.mapItemLastnode = mapItemLastnode;
	}

	public FPTreeNode getRoot() {
		return root;
	}

	public void setRoot(FPTreeNode root) {
		this.root = root;
	}

	public void addTransaction(List<String> transaction, Map<String, Integer> mapItemToGroupID, int groupID) {
		FPTreeNode currentNode = root;
		for (String item : transaction) {
			FPTreeNode child = currentNode.getChildWithName(item);
			if (child == null) {
				// create a new tree node
				FPTreeNode newNode = new FPTreeNode();
				newNode.setName(item);
				newNode.setCount(1);
				newNode.setParent(currentNode);
				currentNode.getChildren().add(newNode);

				currentNode = newNode;

				fixNodeLink(item, newNode, mapItemToGroupID,groupID);
			} else {
				// increase the count
				child.setCount(child.getCount() + 1);
				currentNode = child;
			}
		}
	}

	private void fixNodeLink(String item, FPTreeNode newNode, Map<String, Integer> mapItemToGroupID, int groupID) {

		FPTreeNode lastNode = mapItemLastnode.get(item);
		if (lastNode != null) {
			lastNode.setNodeLink(newNode);
		}
		mapItemLastnode.put(item, newNode);

		// update local header table (g list)
		if (mapItemToGroupID.get(item)==groupID) {
			FPTreeNode headerNode = mapItemNode.get(item);
			if (headerNode == null) {
				mapItemNode.put(item, newNode);
			}
		}
	}

	public void sortHeaderTable(final Map<String, Integer> mapPairSupport) {

		headerItemList = new ArrayList<String>(mapItemNode.keySet());
		Collections.sort(headerItemList, new Comparator<String>() {

			@Override
			public int compare(String o1, String o2) {
				int compare = mapPairSupport.get(o2).compareTo(
						mapPairSupport.get(o1));
				if (compare == 0) {
					return o1.compareTo(o2);
				}
				return compare;
			}
		});

	}

}
