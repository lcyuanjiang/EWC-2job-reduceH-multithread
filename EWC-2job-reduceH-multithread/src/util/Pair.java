package util;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;

public class Pair implements WritableComparable<Pair> {

	private String first;
	private Integer second;

	public Pair() {

	}

	public Pair(String first, Integer second) {
		this.first = first;
		this.second = second;
	}

	public String getFirst() {
		return first;
	}

	public void setFirst(String first) {
		this.first = first;
	}

	public Integer getSecond() {
		return second;
	}

	public void setSecond(Integer second) {
		this.second = second;
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		this.first = in.readUTF();
		this.second = in.readInt();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeUTF(first);
		out.writeInt(second);
	}

	@Override
	public String toString() {
		return String.valueOf(first) + ':' + second;
	}

	@Override
	public int compareTo(Pair o) {
		int compare = o.second.compareTo(this.second);
		if (compare == 0) {
			return this.first.compareTo(o.first);
		}
		return compare;
	}

	@Override
	public int hashCode() {
		int firstHash = hashCodeNull(first);
		// Flip top and bottom 16 bits; this makes the hash function probably
		// different
		// for (a,b) versus (b,a)
		return (firstHash >>> 16 | firstHash << 16) ^ hashCodeNull(second);
	}

	private static int hashCodeNull(Object obj) {
		return obj == null ? 0 : obj.hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		if (!(obj instanceof Pair)) {
			return false;
		}
		Pair otherPair = (Pair) obj;
		return isEqualOrNulls(first, otherPair.getFirst())
				&& isEqualOrNulls(second, otherPair.getSecond());
	}

	private static boolean isEqualOrNulls(Object obj1, Object obj2) {
		return obj1 == null ? obj2 == null : obj1.equals(obj2);
	}

}
