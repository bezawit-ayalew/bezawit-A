
import java.util.Comparator;

public class Point implements Comparable<Point> {

    // compare points by slope
    public final Comparator<Point> SLOPE_ORDER;
    	
    private class SlopeComparator implements Comparator<Point>{
    	public int compare(Point one, Point two){
    		double slopeOne = slopeTo(one);
    		double slopeTwo = slopeTo(two);
    		if( slopeOne < slopeTwo){
    			return -1;
    		}else if(slopeOne == slopeTwo){
    			return 0;
    		}else{
    			return 1;
    		}
    		
    	}
    }

    private final int x;                              // x coordinate
    private final int y;                              // y coordinate

    // create the point (x, y)
    public Point(int x, int y) {
        this.x = x;
        this.y = y;
        SLOPE_ORDER = new SlopeComparator();
    }

    // plot this point to standard drawing
    public void draw() {
    	StdDraw.setPenColor(StdDraw.BOOK_RED);
        StdDraw.point(x, y);
    }

    // draw line between this point and that point to standard drawing
    public void drawTo(Point that) {
        StdDraw.line(this.x, this.y, that.x, that.y);
    }
    public String toString() {
        return "(" + x + ", " + y + ")";
    }
    // return string representation of this point
    
    public int compareTo(Point that) {    	
    	if ((this.y < that.y) || (this.y == that.y && this.x < that.x)) {
    		return -1;
    	}else if (this.y == that.y && this.x == that.x){
    		return 0;
    	}else{
    		return 1;
    	}
    	
    }

    // slope between this point and that point
    public double slopeTo(Point that) {
		//double slope;
			
		if (that.y == this.y && that.x == this.x){
			return Double.NEGATIVE_INFINITY;
		}else if (that.y != this.y && that.x == this.x){
			return Double.POSITIVE_INFINITY;
		}if(that.y - this.y == 0){
			return 0;
		}else{
			return ((double)that.y - this.y)/((double)that.x - this.x);
		}
		 
    }


    // is this point lexicographically smaller than that one?
    // comparing y-coordinates and breaking ties by x-coordinates
    
    

   
   
}