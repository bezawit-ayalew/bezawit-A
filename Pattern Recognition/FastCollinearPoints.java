import java.util.*;

import edu.princeton.cs.algs4.Queue;

public class FastCollinearPoints {
	private int minLength = 3;//initialize minlength
	private Queue<PointSequence> PointQueue;
	private Point[] point;

	public FastCollinearPoints(Point[] points) {
		// makes a defensive copy of the array of points
		//Queue<PointSequence> PointQueue = new Queue<PointSequence>();
		Point[] pointsCopy = Arrays.copyOf(points, points.length);
		point = pointsCopy;//copy so you can access in computeSegments
		
		if(points.length < minLength){
			return;
		}
		
		for(Point oP : pointsCopy){
			Arrays.sort(pointsCopy, oP.SLOPE_ORDER);//sort based on slope order
			CompSegments(minLength); //call the compute method to find collinear points
		}
	}

	public int numberOfPoints() {
		// returns the number of total points in the array
		int N = numberOfSegments(minLength);
		int total_points = 0;
		Queue<PointSequence> PointQueue = new Queue<PointSequence>();

		for (int i = 0; i < N; i++) {
			total_points = PointQueue.dequeue().numberOfPoints();//number of points is the size of the queue
		}
		return total_points;
	}

	public int numberOfSegments(int minLength) {
		// returns the number of segments of length minLength or more
		if (this.minLength != minLength) {
			CompSegments(minLength);
		}
		return this.minLength;//number of segments is the minlength
	}

	private void CompSegments(int minLength) {

			int adjacentPts = 0;
			Point startPoint = point[0];//first point in the array
			Point[] lines = new Point[point.length];//new array with the size of point array
			lines[0] = startPoint;//first point in the array
			double slopeP1 = startPoint.slopeTo(point[1]);//slope of first and second point in the sorted array

			for (int i = 1; i < point.length; i++) {
				//keep checking the slopes of the adjacent points

				Point newPoint = point[i];
				double slope2 = startPoint.slopeTo(newPoint);//slope of first point and all other points in the array

				if (slope2 == slopeP1) {//they are collinear
					lines[++adjacentPts] = newPoint;//increment adjacentPts and set equal to next point
				} else {
					if (adjacentPts >= minLength){
						appendLines(lines, adjacentPts + 1);//connect lines array with adjacent point
					}
					adjacentPts = 1;
					lines[1]= newPoint;
				}

				slopeP1 = slope2;
			}

			
			if (adjacentPts >= minLength){
				appendLines(lines, adjacentPts + 1);////connect lines array with adjacent point
			}
	}
	
	private static void appendLines(Point[] lines, int size){
		Arrays.sort(lines, 1, size);//sort the array of collinear points
					
			//StdOut.printf("%s", lines[0]);//print out the first point
//			for(int k = 1; k < size; k++){
				//Point p3 = lines[k];//go through array 
//				if (lines[k] == null){
//					//System.out.println("asdads:");
//				}
//				
//				//StdOut.printf(" -> %s", p3);//print out p3
//			}
//			
//			//StdOut.println();
//			lines[0].drawTo(lines[size - 1]);

	}

 	public Iterable<PointSequence> segments(int minLength) {
		return new Queue<PointSequence>();}}

	

	// public static void main(String[] args) { 
// 		// draws all maximal length segments of length 4 or more
// 		StdDraw.setXscale(0, 32768);
// 		StdDraw.setYscale(0, 32768);
// 		StdDraw.setPenRadius(0.001);
// 
// 		In file = new In(args[0]);//read in file
// 		int N = file.readInt();
// 		Point[] point = new Point[N];//make array to store file
// 
// 		for (int i = 0; i < N; i++) {
// 			int x = file.readInt();
// 			int y = file.readInt();
// 
// 			Point p = new Point(x, y);//points
// 			point[i] = p;
// 			p.draw();//draw points
// 		}
// 		//Stopwatch stop = new Stopwatch();
// 		FastCollinearPoints fast = new FastCollinearPoints(point);
// 
// 		//StdOut.println("Time: FastCollinearPoints(" + N + ") ~ " + stop.elapsedTime() + "seconds");
// 
// 	}
// }