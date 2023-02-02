
import edu.princeton.cs.algs4.WeightedQuickUnionUF;

public class Percolation {
	private boolean grid[][];
	private int size;
	private WeightedQuickUnionUF qf;
	
	private int Virtualtop;
	private int Virtualbottom;
	private int count;

	public Percolation(int N) {
		// number of rows
		if (N <= 0) {
			throw new java.lang.IllegalArgumentException();
		}
		size = N;
		grid = new boolean[size][size];
		count = 0;
		Virtualtop = size*size;
		Virtualbottom = size*size+1;
		qf = new WeightedQuickUnionUF(size * size + 2);
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				grid[i][j] = false; // blockes all sites
			}
		}
		/*
		 * for(int i = 0 ; i< size ;i++ ){ qf.union( Virtualtop, onedim(i, size)
		 * ); }
		 */
		// full = new WeightedQuickUnionUF (size *size);
	}
	// create N-by-N grid, with all sites initially blocked

	private int onedim(int row, int col) {
		return (size * row) +col ;

	}

	private boolean invalid(int row, int col) {
		if (row < 0 || col < 0 || row > size-1 || col > size-1){	
		return false;}
		else return true;
	}

	public void open(int row, int col) {
		if(grid[row][col] == true){
			return;
		}
		grid[row][col] = true;
		count++;
		if (invalid(row, col) == false) {
		throw new java.lang.IndexOutOfBoundsException();
	    }else if (isOpen(row,col)){
		
		if (row == 0 ) {
			qf.union(onedim(row,col),Virtualtop);
			if(isOpen(row+1,col)){
				qf.union(onedim(row+1,col),(onedim(row,col)));
			}
		}
		if(row == size-1){
		qf.union(onedim( row,col), Virtualbottom);
		if(isOpen(row-1,col)){
			qf.union(onedim(row-1,col),(onedim(row,col)));
		}}
		
		
		if (row+1 < size && isOpen(row+1,col)) { // connect
			// bottom
			qf.union(onedim(row + 1, col), onedim(row,col));

		}
		if (row -1 >= 0&& isOpen(row-1,col)) { // connect
			// top
			qf.union( onedim(row - 1,col), onedim(row,col));

		}
		if (col-1 >= 0 && isOpen(row,col-1))  { // connect
			// left
			qf.union(onedim(row, col - 1), onedim(row,col));

		}
		if (col +1 < size && isOpen(row, col+1))  { // connect
			// right
			qf.union(onedim(row, col + 1),onedim(row,col));

		}
	    }
		
		

	}

	// open the site (row, col) if it is not open already
	public boolean isOpen(int row, int col) {
		if (invalid(row, col) == false) {
			throw new java.lang.IndexOutOfBoundsException();}
		//if (row > col || col > row || row < 0 || col < 0){
			//throw new java.lang.IndexOutOfBoundsException();
		
		return grid[row][col] ;
	}

	// is the site (row, col) open?
	public boolean isFull(int row, int col) {
		if (invalid(row, col) == false) {
			throw new java.lang.IndexOutOfBoundsException();
		}
		
			return qf.connected(onedim(row,col),Virtualtop);
		
	} // is the site (row, col) full?

	public int numberOfOpenSites() {
		return count;

	} // number of open sites

	public boolean percolates() {
		return qf.connected(Virtualtop, Virtualbottom);
	} // does the system percolate?

	public static void main(String[] args) {

	}
}
