
//import edu.princeton.cs.algs4.StdDraw;
import edu.princeton.cs.algs4.StdOut;

public class InteractivePercolationVisualizer {
    private static final int DELAY = 20;

    public static void main(String[] args) {
        // n-by-n percolation system (read from command-line, default = 10)
        int n = 10;          
        if (args.length == 1) n = Integer.parseInt(args[0]);

        // turn on animation mode
        StdDraw.enableDoubleBuffering();

        // repeatedly open site specified my mouse click and draw resulting system
        StdOut.println(n);

        Percolation perc = new Percolation(n);
        PercolationVisualizer.draw(perc, n);
        StdDraw.show();

        while (true) {

            // detected mouse click
            if (StdDraw.mousePressed()) {

                // screen coordinates
                double x = StdDraw.mouseX();
                double y = StdDraw.mouseY();

                // convert to row i, column j
                int i = (int) (n - Math.floor(y) - 1);
                int j = (int) (Math.floor(x));

                // open site (i, j) provided it's in bounds
                if (i >= 0 && i < n && j >= 0 && j < n) {
                    if (!perc.isOpen(i, j)) { 
                        StdOut.println(i + " " + j);
                    }
                    perc.open(i, j);
                }

                // draw n-by-n percolation system
                PercolationVisualizer.draw(perc, n);
            }
            StdDraw.show();
            StdDraw.pause(DELAY);
        }
    }
}
