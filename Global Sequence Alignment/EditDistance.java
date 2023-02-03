//import java.util.*;
public class EditDistance{

private String a;
private String b;
// constructor, saves two arguments in instance variables
public EditDistance(String seq1, String seq2){
	if(seq1 == null || seq2 == null){
	throw new IllegalArgumentException();}
 this.a  = seq1;
 this.b = seq2;
}


// computes a global alignment, if needed; 
// returns the edit distance score
public int score( ) {                         
	String [] s = new String [] { a };
	String [] t = new String [] { b };
	 int M = s.length;
	 int N = t.length;
	 int[][] opt = new int[M+1][N+1];
	 for (int i = M-1; i >= 0; i--)
	 for (int j = N-1; j >= 0; j--)
	 if (s[i] == t[i])
	 opt[i][j] = opt[i+1][j+1]+1;
	 else
	 opt[i][j] = min( opt[i+1][j],opt[i][j+1],opt[i+2][i+2]);
	 return opt[0][0];
	
}
	
// computes a global alignment, if needed; 
// returns an optimal alignment
public String[] optAlignment(){
	String [] s = new String [] { a };
	String [] t = new String [] { b };
	 int M = s.length;
	 int N = t.length;
	 for (int i = 0; i<= M; i++)
		 for (int j = 0; j <= N;)
			 if(s[i] == " " )
				 return new String [] {"-",t[j]};
			 else  if (t[j]=="-" )
				return new String [] {s[i],"-"};
			 else
				 return new String []{ s[i], t[j]};
	return new String [] {a,b};}
  
	 // String [] first = a;
	
	
	


// returns the penalty for aligning char a and char b
public int penalty(char a, char b){
	if( a == b){
		return 0;} //first penality
		else
	
     if  ( a == ' '  || b ==' '){
			return 2; } //third penality
		return 1; }	
		


// returns the min of 3 integers
public int min(int a, int b, int c){
	if( a < b && a < c)
		return a;
	else if( b <= a && b <= c )
		return b;
	else
	   return c;
}


// do some testing/demo of this class
public static void main(String[] args) {
	String l = args[0];
	String q = args[1];
	EditDistance n = new EditDistance(l, q);
	n.score();
	System.out.println(n.score());
	
}}
                               