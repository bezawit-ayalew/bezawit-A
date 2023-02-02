

public class PercolationStats {

	private double[] tests;//array for experiments
	private int count = 0;
	int openSites = 0;//number of open sites
	double frac = 0;

	public PercolationStats(int N, int T) {
		if (N <= 0 || T <= 0) {
			throw new java.lang.IllegalArgumentException();
		}
		count = T;//100
		tests = new double[T];//100
		for (int i = 0; i < T; i++) {//number or tests
			Percolation p = new Percolation(N);//new percolation//200
			
			while (!p.percolates()) {
				int row = StdRandom.uniform(0, N);
				int col = StdRandom.uniform(0, N);
				p.open(row, col);
				
			}
			double frac = (double)openSites / (N*N);
			tests[i] = frac;
			openSites++;
		}
				
	}

	public double mean() {
		// sample mean of percolation threshold

		return StdStats.mean(tests);
	}

	public double stddev() {
		// sample standard deviation of percolation threshold
		return StdStats.stddev(tests);
	}

	public double confidenceLow() {
		// low endpoint of 95% confidence interval
		return mean() - (1.96 * stddev()) / (Math.sqrt(count));
	}

	public double confidenceHigh() {
		// high endpoint of 95% confidence interval
		return mean() + (1.96 * stddev()) / (Math.sqrt(count));
	}

	public static void main(String[] args) {
		int N = 200;
		int T = 100;
		PercolationStats ps = new PercolationStats(N, T);
		StdOut.println("mean =" + ps.mean());
		StdOut.println("stddev =" + ps.stddev());
		StdOut.println("95% confidence interval = " + ps.confidenceHigh());

	}
}