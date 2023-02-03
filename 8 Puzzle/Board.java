
public class Board {
private int [][] board;
private int N;
	public Board(int[][] tiles) {
		// construct a board from an N-by-N array of tiles
		board = new int [N][N];
		for(int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){
				board[i][j] = tiles[i][j];
			}
		}
	}

	// (where tiles[i][j] = tile at row i, column j)
	public int tileAt(int i, int j) {
		// return tile at row i, column j (or 0 if blank)
		if(board[i][j] == 0){
			return 0;
		}else{
			return board[i][j];
		}
		
	}

	public int size () {
		// board size N
		return N;
	}

	public int hamming() {
		// number of tiles out of place
		int wrongBlocks = 0;
		for(int i = 0; i < N; i++){
			for(int j = 0; j < N; j++){
				if (board [i][j] != N*i+j+1 && board[i][j] != 0){
					wrongBlocks++;
					
				}
			}
		}
		return wrongBlocks;
	}

	public int manhattan() {
		// sum of Manhattan distances between tiles and goal
	}

	public boolean isGoal() {
		// is this board the goal board?
		if( hamming() == 0){
			return true;
		}else return false;
		/*boolean goal;
		//for(int i = 0; i < N; i++){
			//for(int j = 0; j < N; i++){
				//if (board [i][j] == N*i+j+1){
					//return true;
				//}else {
					//return false;
				}
			}
		}*/
		
	
	}

	public boolean isSolvable() {
		// is this board solvable?
	}

	public boolean equals(Object y) {
		// does this board equal y?
	}

	public Iterable<Board> neighbors() {
		// all neighboring boards
	}

	public String toString() {
		// string representation of this board (in the output format specified
		// below)
	}

	public static void main(String[] args) {
		// unit testing (not graded)
	}
}