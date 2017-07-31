package fptree;

import java.util.ArrayList;
import java.util.List;

public class FPTreeNode {

	String name;
	int count;
	FPTreeNode parent;
	List<FPTreeNode> children = new ArrayList<FPTreeNode>();
	FPTreeNode nodeLink = null;

	public FPTreeNode() {

	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public int getCount() {
		return count;
	}

	public void setCount(int count) {
		this.count = count;
	}

	public FPTreeNode getParent() {
		return parent;
	}

	public void setParent(FPTreeNode parent) {
		this.parent = parent;
	}

	public List<FPTreeNode> getChildren() {
		return children;
	}

	public void setChildren(List<FPTreeNode> children) {
		this.children = children;
	}

	public FPTreeNode getNodeLink() {
		return nodeLink;
	}

	public void setNodeLink(FPTreeNode nodeLink) {
		this.nodeLink = nodeLink;
	}

	public FPTreeNode getChildWithName(String item) {
		for(FPTreeNode child:this.getChildren()){
			if(child.getName().equals(item)){
				return child;
			}
		}
		return null;
	}

}
